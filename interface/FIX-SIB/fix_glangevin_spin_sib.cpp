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
   Generalized Langevin thermostat for variable-length spin dynamics

   This fix implements a generalized Langevin thermostat that allows
   spin magnitude to fluctuate (variable-length spin mode).

   Based on SPILADY (Spin-Lattice Dynamics) implementation:
   - Ma & Dudarev, PRB 83, 134418 (2011)

   The stochastic LLG equation with longitudinal fluctuations:
   ds/dt = -γ s × H_eff - α s × (s × H_eff) + ξ_T(t) + γ_L (H_eff · ŝ) ŝ + ξ_L(t) ŝ

   where:
   - ξ_T(t): transverse random field
   - ξ_L(t): longitudinal random field
   - γ_L: longitudinal damping coefficient

   For fixed-length spin dynamics, use fix langevin/spin/sib instead.

   References:
   [1] P.-W. Ma, S.L. Dudarev, PRB 83, 134418 (2011) - SPILADY
   [2] J.H. Mentink et al., J. Phys.: Condens. Matter 22, 176001 (2010) - SIB
------------------------------------------------------------------------- */

#include "fix_glangevin_spin_sib.h"

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

FixGLangevinSpinSIB::FixGLangevinSpinSIB(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), random(nullptr)
{
  // Required arguments: temp alpha_t tau_L seed
  if (narg != 7) error->all(FLERR, "Illegal fix glangevin/spin/sib command: expected 7 arguments");

  temp = utils::numeric(FLERR, arg[3], false, lmp);
  alpha_t = utils::numeric(FLERR, arg[4], false, lmp);
  tau_L = utils::numeric(FLERR, arg[5], false, lmp);
  seed = utils::inumeric(FLERR, arg[6], false, lmp);

  // Check validity
  if (alpha_t < 0.0)
    error->all(FLERR, "Fix glangevin/spin/sib alpha_t must be >= 0");
  if (temp < 0.0)
    error->all(FLERR, "Fix glangevin/spin/sib temperature must be >= 0");
  if (tau_L <= 0.0)
    error->all(FLERR, "Fix glangevin/spin/sib tau_L must be > 0");
  if (seed <= 0)
    error->all(FLERR, "Illegal fix glangevin/spin/sib seed");

  // Convert relaxation time to damping coefficient
  // gamma_L = 1/tau_L (in ps^-1)
  gamma_L = 1.0 / tau_L;

  // Set flags
  tdamp_flag = 0;
  temp_flag = 0;
  if (alpha_t > 0.0) tdamp_flag = 1;
  if (temp > 0.0) temp_flag = 1;

  // Initialize random number generator
  random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */

FixGLangevinSpinSIB::~FixGLangevinSpinSIB()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixGLangevinSpinSIB::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGLangevinSpinSIB::init()
{
  // Check if fix nve/spin/sib is defined
  auto sib_fixes = modify->get_fix_by_style("^nve/spin/sib$");
  if (sib_fixes.empty())
    error->warning(FLERR, "Fix glangevin/spin/sib should be used with fix nve/spin/sib");
  auto lang_fixes = modify->get_fix_by_style("^langevin/spin/sib$");
  if (!lang_fixes.empty())
    error->all(FLERR, "Fix glangevin/spin/sib cannot be used with fix langevin/spin/sib");

  // Verify this fix comes after force fixes
  int flag_force = 0;
  int flag_lang = 0;
  for (int i = 0; i < modify->nfix; i++) {
    if (utils::strmatch(modify->fix[i]->style, "^precession/spin"))
      flag_force = i;
    if (strcmp(modify->fix[i]->style, "glangevin/spin/sib") == 0)
      flag_lang = i;
  }

  if (flag_force >= flag_lang)
    error->all(FLERR, "Fix glangevin/spin/sib has to come after all other spin fixes");

  // Gilbert factor for transverse damping
  gil_factor = 1.0 / (1.0 + alpha_t * alpha_t);

  // Use dt/2 for SIB method - each half-step applies noise once
  dts = 0.5 * update->dt;

  // Calculate transverse noise strength
  double hbar = force->hplanck / MY_2PI;  // eV/(rad.THz)
  double kb = force->boltz;               // eV/K

  D_T = (alpha_t * (1.0 + alpha_t * alpha_t) * kb * temp);
  D_T /= (hbar * dts);
  sigma_T = sqrt(2.0 * D_T);

  // Calculate longitudinal noise strength
  // Based on SPILADY: sigma_L = sqrt(2 * k_B * T * gamma_L / dt)
  // gamma_L = 1/tau_L (in ps^-1)
  D_L = gamma_L * kb * temp / dts;
  sigma_L = sqrt(2.0 * D_L);

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix glangevin/spin/sib: Variable-length spin thermostat\n");
    utils::logmesg(lmp, "Fix glangevin/spin/sib: Using dts = dt/2 = {} ps for SIB half-step\n", dts);
    utils::logmesg(lmp, "Fix glangevin/spin/sib: Transverse: alpha_t = {} (dimensionless), sigma_T = {}\n", alpha_t, sigma_T);
    utils::logmesg(lmp, "Fix glangevin/spin/sib: Longitudinal: tau_L = {} ps, gamma_L = {} ps^-1, sigma_L = {}\n", tau_L, gamma_L, sigma_L);
  }
}

/* ---------------------------------------------------------------------- */

void FixGLangevinSpinSIB::setup(int vflag)
{
  // Nothing to do
}

/* ---------------------------------------------------------------------- */

void FixGLangevinSpinSIB::add_tdamping(double spi[3], double fmi[3])
{
  // Transverse damping: fmi -= alpha_t * (fmi × spi)
  double cpx = fmi[1] * spi[2] - fmi[2] * spi[1];
  double cpy = fmi[2] * spi[0] - fmi[0] * spi[2];
  double cpz = fmi[0] * spi[1] - fmi[1] * spi[0];

  fmi[0] -= alpha_t * cpx;
  fmi[1] -= alpha_t * cpy;
  fmi[2] -= alpha_t * cpz;
}

/* ---------------------------------------------------------------------- */

void FixGLangevinSpinSIB::add_noise(double fmi[3], double noise[3])
{
  // Add transverse stochastic field
  fmi[0] += sigma_T * noise[0];
  fmi[1] += sigma_T * noise[1];
  fmi[2] += sigma_T * noise[2];
}

/* ---------------------------------------------------------------------- */

void FixGLangevinSpinSIB::apply_gil_factor(double fmi[3])
{
  // Apply Gilbert factor: fmi *= 1/(1+alpha_t^2)
  fmi[0] *= gil_factor;
  fmi[1] *= gil_factor;
  fmi[2] *= gil_factor;
}

/* ---------------------------------------------------------------------- */

void FixGLangevinSpinSIB::add_longitudinal_damping(double spi[3], double fmi[3], double sp_mag)
{
  // Longitudinal damping: adds γ_L * (H_eff · ŝ) * ŝ to the effective field
  if (sp_mag < 1e-10) return;

  double spi_norm = sqrt(spi[0]*spi[0] + spi[1]*spi[1] + spi[2]*spi[2]);
  if (spi_norm < 1e-10) return;

  // Calculate ŝ (unit vector along spin direction)
  double s_hat[3];
  s_hat[0] = spi[0] / spi_norm;
  s_hat[1] = spi[1] / spi_norm;
  s_hat[2] = spi[2] / spi_norm;

  // Calculate H_eff · ŝ (longitudinal component of effective field)
  double H_parallel = fmi[0] * s_hat[0] + fmi[1] * s_hat[1] + fmi[2] * s_hat[2];

  // Add longitudinal damping term: γ_L * H_parallel * ŝ
  fmi[0] += gamma_L * H_parallel * s_hat[0];
  fmi[1] += gamma_L * H_parallel * s_hat[1];
  fmi[2] += gamma_L * H_parallel * s_hat[2];
}

/* ---------------------------------------------------------------------- */

void FixGLangevinSpinSIB::add_longitudinal_noise(double spi[3], double fmi[3], double noise_L)
{
  // Add longitudinal stochastic field: σ_L * ξ_L * ŝ
  double spi_norm = sqrt(spi[0]*spi[0] + spi[1]*spi[1] + spi[2]*spi[2]);
  if (spi_norm < 1e-10) return;

  // Calculate ŝ (unit vector along spin direction)
  double s_hat[3];
  s_hat[0] = spi[0] / spi_norm;
  s_hat[1] = spi[1] / spi_norm;
  s_hat[2] = spi[2] / spi_norm;

  // Add longitudinal noise: σ_L * ξ_L * ŝ
  fmi[0] += sigma_L * noise_L * s_hat[0];
  fmi[1] += sigma_L * noise_L * s_hat[1];
  fmi[2] += sigma_L * noise_L * s_hat[2];
}

/* ----------------------------------------------------------------------
   SIB predictor step: generate noise and store it (transverse only)
   Same as langevin/spin/sib for direction dynamics
------------------------------------------------------------------------- */

void FixGLangevinSpinSIB::compute_single_langevin_store_noise(int i, double spi[3],
                                                              double fmi[3], double noise_out[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    // Step 1: Generate and store transverse noise, add to force
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

    // Step 2: Add transverse damping using noisy field
    if (tdamp_flag) add_tdamping(spi, fmi);

    // Step 3: Apply Gilbert factor for transverse components
    if (temp_flag) apply_gil_factor(fmi);
  }
}

/* ----------------------------------------------------------------------
   SIB corrector step: reuse stored noise (transverse only)
------------------------------------------------------------------------- */

void FixGLangevinSpinSIB::compute_single_langevin_reuse_noise(int i, double spi[3],
                                                              double fmi[3], double noise_in[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    // Step 1: Reuse stored transverse noise
    if (temp_flag) {
      add_noise(fmi, noise_in);
    }

    // Step 2: Add transverse damping
    if (tdamp_flag) add_tdamping(spi, fmi);

    // Step 3: Apply Gilbert factor for transverse components
    if (temp_flag) apply_gil_factor(fmi);
  }
}

/* ----------------------------------------------------------------------
   Longitudinal dynamics: update spin magnitude sp[3]

   Equation: d|S|/dt = γ_L * (H_eff · ŝ) + ξ_L

   Discretized: |S|^{n+1} = |S|^n + dt * [γ_L * (H_eff · ŝ) + σ_L * ξ_L]

   where σ_L = sqrt(2 * γ_L * k_B * T / dt) satisfies FDT
------------------------------------------------------------------------- */

void FixGLangevinSpinSIB::compute_longitudinal_step(int i, double spi[3], double fmi[3],
                                                    double &noise_L_out, double dt_step)
{
  int *mask = atom->mask;
  double **sp = atom->sp;

  if (mask[i] & groupbit) {
    // Calculate ŝ (unit vector along spin direction)
    double spi_norm = sqrt(spi[0]*spi[0] + spi[1]*spi[1] + spi[2]*spi[2]);
    if (spi_norm < 1e-10) {
      noise_L_out = 0.0;
      return;
    }

    double s_hat[3];
    s_hat[0] = spi[0] / spi_norm;
    s_hat[1] = spi[1] / spi_norm;
    s_hat[2] = spi[2] / spi_norm;

    // Calculate H_eff · ŝ (longitudinal component of effective field)
    double H_parallel = fmi[0] * s_hat[0] + fmi[1] * s_hat[1] + fmi[2] * s_hat[2];

    // Generate longitudinal noise
    if (temp_flag) {
      noise_L_out = random->gaussian();
    } else {
      noise_L_out = 0.0;
    }

    // Longitudinal noise strength for this timestep
    // σ_L = sqrt(2 * γ_L * k_B * T / dt)
    double kb = force->boltz;
    double sigma_L_dt = sqrt(2.0 * gamma_L * kb * temp / dt_step);

    // Update spin magnitude: d|S|/dt = γ_L * H_parallel + σ_L * ξ_L
    double d_sp_mag = dt_step * gamma_L * H_parallel + sqrt(dt_step) * sigma_L_dt * noise_L_out;

    // Update sp[3] (spin magnitude)
    sp[i][3] += d_sp_mag;

    // Ensure spin magnitude stays positive
    if (sp[i][3] < 0.01) sp[i][3] = 0.01;
  }
}

/* ----------------------------------------------------------------------
   Longitudinal dynamics corrector: reuse stored noise
------------------------------------------------------------------------- */

void FixGLangevinSpinSIB::compute_longitudinal_step_reuse(int i, double spi[3], double fmi[3],
                                                          double noise_L_in, double dt_step)
{
  int *mask = atom->mask;
  double **sp = atom->sp;

  if (mask[i] & groupbit) {
    // Calculate ŝ (unit vector along spin direction)
    double spi_norm = sqrt(spi[0]*spi[0] + spi[1]*spi[1] + spi[2]*spi[2]);
    if (spi_norm < 1e-10) return;

    double s_hat[3];
    s_hat[0] = spi[0] / spi_norm;
    s_hat[1] = spi[1] / spi_norm;
    s_hat[2] = spi[2] / spi_norm;

    // Calculate H_eff · ŝ (longitudinal component of effective field)
    double H_parallel = fmi[0] * s_hat[0] + fmi[1] * s_hat[1] + fmi[2] * s_hat[2];

    // Longitudinal noise strength for this timestep
    double kb = force->boltz;
    double sigma_L_dt = sqrt(2.0 * gamma_L * kb * temp / dt_step);

    // Update spin magnitude using same noise as predictor
    double d_sp_mag = dt_step * gamma_L * H_parallel + sqrt(dt_step) * sigma_L_dt * noise_L_in;

    // Update sp[3] (spin magnitude)
    sp[i][3] += d_sp_mag;

    // Ensure spin magnitude stays positive
    if (sp[i][3] < 0.01) sp[i][3] = 0.01;
  }
}
