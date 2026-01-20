// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Langevin thermostat for NEP-SPIN RK4 method

   Key features:
   - Uses dts = dt/2 for RK4 half-step scaling
   - Provides two methods for RK4 stages:
     1. compute_single_langevin_store_noise: generates and stores noise
     2. compute_single_langevin_reuse_noise: reuses stored noise

   The stochastic LLG equation in the RK4 method requires the same noise
   to be used for all k1-k4 stages for correct Stratonovich interpretation.
------------------------------------------------------------------------- */

#include "fix_langevin_spin_rk4.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixLangevinSpinRK4::FixLangevinSpinRK4(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), random(nullptr)
{
  if (narg < 6) error->all(FLERR, "Illegal fix langevin/spin/rk4 command");

  temp = utils::numeric(FLERR, arg[3], false, lmp);
  alpha_t = utils::numeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);

  // Check validity
  if (alpha_t < 0.0)
    error->all(FLERR, "Fix langevin/spin/rk4 damping must be >= 0");
  if (temp < 0.0)
    error->all(FLERR, "Fix langevin/spin/rk4 temperature must be >= 0");
  if (seed <= 0)
    error->all(FLERR, "Illegal fix langevin/spin/rk4 seed");

  // Set flags
  tdamp_flag = 0;
  temp_flag = 0;
  if (alpha_t > 0.0) tdamp_flag = 1;
  if (temp > 0.0) temp_flag = 1;

  // Initialize random number generator
  random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */

FixLangevinSpinRK4::~FixLangevinSpinRK4()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixLangevinSpinRK4::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinRK4::init()
{
  // Check if fix nve/spin/rk4 is defined
  auto rk4_fixes = modify->get_fix_by_style("^nve/spin/rk4$");
  if (rk4_fixes.empty())
    error->warning(FLERR, "Fix langevin/spin/rk4 should be used with fix nve/spin/rk4");

  // Verify this fix comes after force fixes
  int flag_force = 0;
  int flag_lang = 0;
  for (int i = 0; i < modify->nfix; i++) {
    if (utils::strmatch(modify->fix[i]->style, "^precession/spin"))
      flag_force = i;
    if (strcmp(modify->fix[i]->style, "langevin/spin/rk4") == 0)
      flag_lang = i;
  }

  if (flag_force >= flag_lang)
    error->all(FLERR, "Fix langevin/spin/rk4 has to come after all other spin fixes");

  // Gilbert factor
  gil_factor = 1.0 / (1.0 + alpha_t * alpha_t);

  // Use dt/2 for RK4 half-steps (noise generated once per half-step)
  dts = 0.5 * update->dt;

  // Calculate noise strength
  // D = alpha * k_B * T / (mu_s * hbar)
  // sigma = sqrt(2 * D)
  double hbar = force->hplanck / MY_2PI;  // eV/(rad.THz)
  double kb = force->boltz;               // eV/K

  // We do NOT include 1/mu_s here because pair_nep_spin already
  // includes mu_s in fm (fm = mu_s * dE/dM / hbar).
  D = (alpha_t * (1.0 + alpha_t * alpha_t) * kb * temp);
  D /= (hbar * dts);
  sigma = sqrt(2.0 * D);

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix langevin/spin/rk4: Using dts = dt/2 = {} for RK4 half-step\n", dts);
    utils::logmesg(lmp, "Fix langevin/spin/rk4: sigma = {}, D = {}\n", sigma, D);
    utils::logmesg(lmp, "Fix langevin/spin/rk4: Same noise used for k1-k4 in each half-step\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinRK4::setup(int vflag)
{
  // Nothing to do
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinRK4::add_tdamping(double spi[3], double fmi[3])
{
  // Transverse damping: fmi -= alpha * (fmi × spi)
  double cpx = fmi[1] * spi[2] - fmi[2] * spi[1];
  double cpy = fmi[2] * spi[0] - fmi[0] * spi[2];
  double cpz = fmi[0] * spi[1] - fmi[1] * spi[0];

  fmi[0] -= alpha_t * cpx;
  fmi[1] -= alpha_t * cpy;
  fmi[2] -= alpha_t * cpz;
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinRK4::add_noise(double fmi[3], double noise[3])
{
  // Add stochastic field to effective field
  fmi[0] += sigma * noise[0];
  fmi[1] += sigma * noise[1];
  fmi[2] += sigma * noise[2];
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinRK4::apply_gil_factor(double fmi[3])
{
  // Apply Gilbert factor: fmi *= 1/(1+alpha^2)
  fmi[0] *= gil_factor;
  fmi[1] *= gil_factor;
  fmi[2] *= gil_factor;
}

/* ----------------------------------------------------------------------
   Standard Langevin computation (for compatibility)
   Generates new noise each time

   Order follows original LAMMPS fix_langevin_spin:
   1. Add noise (if enabled)
   2. Add damping using the noisy field
   3. Apply Gilbert factor (only when temp_flag is true)
------------------------------------------------------------------------- */

void FixLangevinSpinRK4::compute_single_langevin(int i, double spi[3], double fmi[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    if (temp_flag) {
      double noise[3];
      noise[0] = random->gaussian();
      noise[1] = random->gaussian();
      noise[2] = random->gaussian();
      add_noise(fmi, noise);
    }

    if (tdamp_flag) add_tdamping(spi, fmi);

    if (temp_flag) apply_gil_factor(fmi);
  }
}

/* ----------------------------------------------------------------------
   RK4 k1 stage: generate noise and store it

   Order follows original LAMMPS fix_langevin_spin:
   1. Add noise (if enabled)
   2. Add damping using the noisy field
   3. Apply Gilbert factor (only when temp_flag is true)
------------------------------------------------------------------------- */

void FixLangevinSpinRK4::compute_single_langevin_store_noise(int i, double spi[3],
                                                             double fmi[3], double noise_out[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    if (temp_flag) {
      noise_out[0] = random->gaussian();
      noise_out[1] = random->gaussian();
      noise_out[2] = random->gaussian();
      add_noise(fmi, noise_out);
    } else {
      noise_out[0] = 0.0;
      noise_out[1] = 0.0;
      noise_out[2] = 0.0;
    }

    if (tdamp_flag) add_tdamping(spi, fmi);

    if (temp_flag) apply_gil_factor(fmi);
  }
}

/* ----------------------------------------------------------------------
   RK4 k2-k4 stages: reuse stored noise
------------------------------------------------------------------------- */

void FixLangevinSpinRK4::compute_single_langevin_reuse_noise(int i, double spi[3],
                                                             double fmi[3], double noise_in[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    if (temp_flag) {
      add_noise(fmi, noise_in);
    }

    if (tdamp_flag) add_tdamping(spi, fmi);

    if (temp_flag) apply_gil_factor(fmi);
  }
}
