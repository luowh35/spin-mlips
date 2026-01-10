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
   Langevin thermostat for NEP-SPIN SIB method

   Key features:
   - Uses dts = dt (full timestep) for noise scaling
   - Provides two methods for SIB predictor-corrector:
     1. compute_single_langevin_store_noise: generates and stores noise
     2. compute_single_langevin_reuse_noise: reuses stored noise

   The stochastic LLG equation in the SIB method requires the same noise
   to be used in both predictor and corrector steps for correct
   Stratonovich interpretation.

   Reference:
   J.H. Mentink, M.V. Tretyakov, A. Fasolino, M.I. Katsnelson, T. Rasing,
   "Stable and fast semi-implicit integration of the stochastic
   Landau-Lifshitz equation", J. Phys.: Condens. Matter 22, 176001 (2010)
   DOI: 10.1088/0953-8984/22/17/176001
------------------------------------------------------------------------- */

#include "fix_langevin_spin_sib.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "modify.h"
#include "random_mars.h"
#include "respa.h"
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
  auto sib_fixes = modify->get_fix_by_style("^nve/spin/sib$");
  if (sib_fixes.empty())
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

  // Use dt/2 for SIB method with spin-lattice coupling
  // Each half-step uses dts = dt/2, and noise is generated once per half-step
  // Total: 2 half-steps × 2 NN calls = 4 NN calls per timestep
  dts = 0.5 * update->dt;

  // Calculate noise strength
  // From Mentink et al. (2010), Eq. (6):
  // D = alpha * k_B * T / (mu_s * hbar)
  // sigma = sqrt(2 * D)
  double hbar = force->hplanck / MY_2PI;  // eV/(rad.THz)
  double kb = force->boltz;               // eV/K

  // Note: The factor (1 + alpha^2) comes from the transformation to
  // the effective field representation
  D = (alpha_t * (1.0 + alpha_t * alpha_t) * kb * temp);
  D /= (hbar * dts);
  sigma = sqrt(2.0 * D);

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix langevin/spin/sib: Using dts = dt/2 = {} for SIB half-step\n", dts);
    utils::logmesg(lmp, "Fix langevin/spin/sib: sigma = {}, D = {}\n", sigma, D);
    utils::logmesg(lmp, "Fix langevin/spin/sib: Same noise used in predictor and corrector of each half-step\n");
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
  // Transverse damping: -alpha_t * (fm × s)
  // This corresponds to the Gilbert damping term in LLG equation
  double cpx = fmi[1] * spi[2] - fmi[2] * spi[1];
  double cpy = fmi[2] * spi[0] - fmi[0] * spi[2];
  double cpz = fmi[0] * spi[1] - fmi[1] * spi[0];

  fmi[0] -= alpha_t * cpx;
  fmi[1] -= alpha_t * cpy;
  fmi[2] -= alpha_t * cpz;
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpinSIB::add_noise_to_force(double fmi[3], double noise[3])
{
  // Add stochastic field contribution (same form as fix langevin/spin)
  // Random field = sigma * noise, then apply Gilbert factor to full fmi
  fmi[0] += sigma * noise[0];
  fmi[1] += sigma * noise[1];
  fmi[2] += sigma * noise[2];

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
    if (tdamp_flag) add_tdamping(spi, fmi);
    if (temp_flag) {
      double noise[3];
      noise[0] = random->gaussian();
      noise[1] = random->gaussian();
      noise[2] = random->gaussian();
      add_noise_to_force(fmi, noise);
    }
  }
}

/* ----------------------------------------------------------------------
   SIB predictor step: generate noise and store it
   This is called in the predictor step of SIB
------------------------------------------------------------------------- */

void FixLangevinSpinSIB::compute_single_langevin_store_noise(int i, double spi[3],
                                                             double fmi[3], double noise_out[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    // Add damping
    if (tdamp_flag) add_tdamping(spi, fmi);

    // Generate and store noise
    if (temp_flag) {
      noise_out[0] = random->gaussian();
      noise_out[1] = random->gaussian();
      noise_out[2] = random->gaussian();

      // Add noise contribution to force
      add_noise_to_force(fmi, noise_out);
    } else {
      noise_out[0] = 0.0;
      noise_out[1] = 0.0;
      noise_out[2] = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   SIB corrector step: reuse stored noise
   This is called in the corrector step of SIB
   CRITICAL: Must use the same noise as the predictor step!
------------------------------------------------------------------------- */

void FixLangevinSpinSIB::compute_single_langevin_reuse_noise(int i, double spi[3],
                                                             double fmi[3], double noise_in[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    // Add damping (evaluated at midpoint state)
    if (tdamp_flag) add_tdamping(spi, fmi);

    // Reuse stored noise (do NOT generate new noise!)
    if (temp_flag) {
      add_noise_to_force(fmi, noise_in);
    }
  }
}
