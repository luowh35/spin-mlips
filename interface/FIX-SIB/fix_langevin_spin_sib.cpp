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
   Langevin thermostat for SPIN-STEP SIB method
------------------------------------------------------------------------- */

#include "fix_langevin_spin_sib.h"

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

FixLangevinSpinSIB::FixLangevinSpinSIB(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), random(nullptr)
{
  if (narg < 6) error->all(FLERR, "Illegal fix langevin/spin/sib command");

  temp = utils::numeric(FLERR, arg[3], false, lmp);
  alpha_t = utils::numeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);

  // Check validity
  if (alpha_t < 0.0)
    error->all(FLERR, "Fix langevin/spin/sib damping must be >= 0");
  if (temp < 0.0)
    error->all(FLERR, "Fix langevin/spin/sib temperature must be >= 0");
  if (seed <= 0)
    error->all(FLERR, "Illegal fix langevin/spin/sib seed");

  // Set flags
  tdamp_flag = 0;
  temp_flag = 0;
  if (alpha_t > 0.0) tdamp_flag = 1;
  if (temp > 0.0) temp_flag = 1;

  // Initialize random number generator
  random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */

FixLangevinSpinSIB::~FixLangevinSpinSIB()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixLangevinSpinSIB::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinSIB::init()
{
  // Check if fix nve/spin/sib is defined
  auto ema_fixes = modify->get_fix_by_style("^nve/spin/sib$");
  if (ema_fixes.empty())
    error->warning(FLERR, "Fix langevin/spin/sib should be used with fix nve/spin/sib");

  // Verify this fix comes after force fixes
  int flag_force = 0;
  int flag_lang = 0;
  for (int i = 0; i < modify->nfix; i++) {
    if (utils::strmatch(modify->fix[i]->style, "^precession/spin"))
      flag_force = i;
    if (strcmp(modify->fix[i]->style, "langevin/spin/sib") == 0)
      flag_lang = i;
  }

  if (flag_force >= flag_lang)
    error->all(FLERR, "Fix langevin/spin/sib has to come after all other spin fixes");

  // Gilbert factor
  gil_factor = 1.0 / (1.0 + alpha_t * alpha_t);

  // Use dt/2 for SIB method - each half-step applies noise once
  dts = 0.5 * update->dt;

  // Calculate noise strength
  // From Mentink et al. (2010), Eq. (6):
  // D = alpha * k_B * T / (mu_s * hbar)
  // sigma = sqrt(2 * D)
  double hbar = force->hplanck / MY_2PI;  // eV/(rad.THz)
  double kb = force->boltz;               // eV/K

  // Note: The factor (1 + alpha^2) comes from the transformation to
  // the effective field representation.
  D = (alpha_t * (1.0 + alpha_t * alpha_t) * kb * temp);
  D /= (hbar * dts);
  sigma = sqrt(2.0 * D);

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix langevin/spin/sib: Using dts = dt/2 = {} for SIB half-step\n", dts);
    utils::logmesg(lmp, "Fix langevin/spin/sib: sigma = {}, D = {}\n", sigma, D);
    utils::logmesg(lmp, "Fix langevin/spin/sib: Same noise used in predictor and corrector\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinSIB::setup(int vflag)
{
  // Nothing to do
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinSIB::add_tdamping(double spi[3], double fmi[3])
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

void FixLangevinSpinSIB::add_noise(double fmi[3], double noise[3])
{
  // Add stochastic field to effective field
  fmi[0] += sigma * noise[0];
  fmi[1] += sigma * noise[1];
  fmi[2] += sigma * noise[2];
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinSIB::apply_gil_factor(double fmi[3])
{
  // Apply Gilbert factor: fmi *= 1/(1+alpha^2)
  fmi[0] *= gil_factor;
  fmi[1] *= gil_factor;
  fmi[2] *= gil_factor;
}

/* ----------------------------------------------------------------------
   Standard Langevin computation (for compatibility)
   Generates new noise each time
------------------------------------------------------------------------- */

void FixLangevinSpinSIB::compute_single_langevin(int i, double spi[3], double fmi[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    // Step 1: Add noise (only if temp_flag)
    if (temp_flag) {
      double noise[3];
      noise[0] = random->gaussian();
      noise[1] = random->gaussian();
      noise[2] = random->gaussian();
      add_noise(fmi, noise);
    }

    // Step 2: Add damping using noisy field
    if (tdamp_flag) add_tdamping(spi, fmi);

    // Step 3: Apply Gilbert factor (only if temp_flag)
    if (temp_flag) apply_gil_factor(fmi);
  }
}

/* ----------------------------------------------------------------------
   SIB predictor step: generate noise and store it
------------------------------------------------------------------------- */

void FixLangevinSpinSIB::compute_single_langevin_store_noise(int i, double spi[3],
                                                             double fmi[3], double noise_out[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    // Step 1: Generate and store noise, add to force
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

    // Step 2: Add damping using noisy field
    if (tdamp_flag) add_tdamping(spi, fmi);

    // Step 3: Apply Gilbert factor
    if (temp_flag) apply_gil_factor(fmi);
  }
}

/* ----------------------------------------------------------------------
   SIB corrector step: reuse stored noise
   CRITICAL: Must use the same noise as the predictor step!
------------------------------------------------------------------------- */

void FixLangevinSpinSIB::compute_single_langevin_reuse_noise(int i, double spi[3],
                                                             double fmi[3], double noise_in[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    // Step 1: Reuse stored noise
    if (temp_flag) {
      add_noise(fmi, noise_in);
    }

    // Step 2: Add damping at midpoint state using noisy field
    if (tdamp_flag) add_tdamping(spi, fmi);

    // Step 3: Apply Gilbert factor
    if (temp_flag) apply_gil_factor(fmi);
  }
}
