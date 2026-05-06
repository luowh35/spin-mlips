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
   SPIN-STEP SIB Method with Spin-Lattice Coupling
   Adapted from NEP-SPIN implementation for use with pair_style spin/step
------------------------------------------------------------------------- */

#include "fix_nve_spin_sib.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "fix_langevin_spin_sib.h"
#include "fix_glangevin_spin_sib.h"
#include "fix_landau_spin.h"
#include "fix_precession_spin.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "pair_spin_ml.h"
#include "pair_spin.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_nve_spin_sib[] =
  "fix nve/spin/sib command:\n\n"
  "@article{mentink2010stable,\n"
  "title={Stable and fast semi-implicit integration of the stochastic "
  "   Landau-Lifshitz equation},\n"
  "author={Mentink, J. H. and Tretyakov, M. V. and Fasolino, A. and "
  "   Katsnelson, M. I. and Rasing, T.},\n"
  "journal={Journal of Physics: Condensed Matter},\n"
  "volume={22},\n"
  "pages={176001},\n"
  "year={2010},\n"
  "doi={10.1088/0953-8984/22/17/176001}\n"
  "}\n\n"
  "@article{tranchida2018massively,\n"
  "title={Massively Parallel Symplectic Algorithm for Coupled Magnetic Spin "
  "   Dynamics and Molecular Dynamics},\n"
  "author={Tranchida, J and Plimpton, S J and Thibaudeau, P and Thompson, A P},\n"
  "journal={Journal of Computational Physics},\n"
  "volume={372},\n"
  "pages={406--425},\n"
  "year={2018},\n"
  "doi={10.1016/j.jcp.2018.06.042}\n"
  "}\n\n";

/* ----------------------------------------------------------------------
   Helper: pad args with "lattice moving" if user omitted the keyword,
   so that the FixNVESpin parent constructor receives valid arguments.
------------------------------------------------------------------------- */

static int sib_pad_narg(int narg) { return (narg < 5) ? 5 : narg; }

static char **sib_pad_args(int narg, char **arg)
{
  static char *buf[5];
  static char kw[] = "lattice";
  static char val[] = "moving";
  if (narg >= 5) return arg;
  for (int i = 0; i < narg && i < 3; i++) buf[i] = arg[i];
  buf[3] = kw;
  buf[4] = val;
  return buf;
}

static bool pair_ml_has_longitudinal_force(PairSpinML *pair_spin_ml)
{
  return pair_spin_ml && pair_spin_ml->has_longitudinal_force();
}

/* ---------------------------------------------------------------------- */

FixNVESpinSIB::FixNVESpinSIB(LAMMPS *lmp, int narg, char **arg) :
  FixNVESpin(lmp, sib_pad_narg(narg), sib_pad_args(narg, arg)),
  pair_spin_ml(nullptr),
  locklangevinspin_sib(nullptr),
  lockglangevinspin_sib(nullptr),
  nlandauspin(0), locklandauspin(nullptr),
  s_save(nullptr), mag_save(nullptr),
  H_par_save(nullptr), noise_vec(nullptr),
  noise_L_vec(nullptr), fm_full(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_nve_spin_sib);

  if (narg < 3) error->all(FLERR, "Illegal fix nve/spin/sib command");

  // Parent constructor (FixNVESpin) already handled:
  //   time_integrate, lattice_flag, atom map check,
  //   lattice keyword parsing, atom/spin style check.
  // Re-parse lattice keyword from original args in case narg was padded.
  lattice_flag = 1;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "lattice") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix nve/spin/sib command");
      const std::string latarg = arg[iarg + 1];
      if ((latarg == "no") || (latarg == "off") || (latarg == "false") || (latarg == "frozen"))
        lattice_flag = 0;
      else if ((latarg == "yes") || (latarg == "on") || (latarg == "true") || (latarg == "moving"))
        lattice_flag = 1;
      else
        error->all(FLERR, "Illegal fix nve/spin/sib command");
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix nve/spin/sib command");
    }
  }
}

/* ---------------------------------------------------------------------- */

FixNVESpinSIB::~FixNVESpinSIB()
{
  memory->destroy(s_save);
  memory->destroy(mag_save);
  memory->destroy(H_par_save);
  memory->destroy(noise_vec);
  memory->destroy(noise_L_vec);
  memory->destroy(fm_full);
  // spin_pairs and lockprecessionspin are freed by parent ~FixNVESpin()
  delete[] locklandauspin;
}

/* ---------------------------------------------------------------------- */

int FixNVESpinSIB::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::init()
{
  // Set timesteps
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  // dts = dt/2 for half-step SIB updates (two half-steps per full timestep)
  dts = 0.5 * update->dt;

  // Find ML spin pair style (spin/step, spin/nep, or other PairSpinML derivatives)
  // First try spin/step
  pair_spin_ml = dynamic_cast<PairSpinML *>(force->pair_match("spin/step", 0, 0));
  // If not found, try spin/nep
  if (!pair_spin_ml)
    pair_spin_ml = dynamic_cast<PairSpinML *>(force->pair_match("spin/nep", 0, 0));

  // Find standard PairSpin styles (e.g. spin/exchange, spin/dmi, ...)
  // Exclude PairSpinML derivatives to avoid double-counting
  delete[] spin_pairs;
  spin_pairs = nullptr;
  npairspin = 0;

  // pair_match ignores nsub for non-hybrid pair styles (always returns the
  // same pointer), so we must distinguish hybrid vs non-hybrid explicitly.
  bool is_hybrid = (force->pair_match("^hybrid", 0) != nullptr);

  if (is_hybrid) {
    // Hybrid: nsub=1,2,... iterates over matching sub-styles correctly
    for (int i = 1; force->pair_match("^spin", 0, i); i++) {
      Pair *p = force->pair_match("^spin", 0, i);
      if (dynamic_cast<PairSpinML *>(p) == nullptr)
        npairspin++;
    }
    if (npairspin > 0) {
      spin_pairs = new PairSpin*[npairspin];
      int count = 0;
      for (int i = 1; force->pair_match("^spin", 0, i); i++) {
        Pair *p = force->pair_match("^spin", 0, i);
        if (dynamic_cast<PairSpinML *>(p) == nullptr)
          spin_pairs[count++] = dynamic_cast<PairSpin *>(p);
      }
    }
  } else {
    // Non-hybrid: at most one pair style
    Pair *p = force->pair_match("^spin", 0, 0);
    if (p && dynamic_cast<PairSpinML *>(p) == nullptr) {
      npairspin = 1;
      spin_pairs = new PairSpin*[1];
      spin_pairs[0] = dynamic_cast<PairSpin *>(p);
    }
  }

  // Require at least one spin pair style
  if (!pair_spin_ml && npairspin == 0)
    error->all(FLERR, "Fix nve/spin/sib requires at least one spin pair style "
               "(PairSpinML or standard PairSpin)");

  // Find fix precession/spin styles (multiple allowed)
  int iforce;
  nprecspin = 0;
  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (utils::strmatch(modify->fix[iforce]->style, "^precession/spin")) {
      nprecspin++;
    }
  }

  if (nprecspin > 0) {
    delete[] lockprecessionspin;
    lockprecessionspin = new FixPrecessionSpin *[nprecspin];

    int count = 0;
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (utils::strmatch(modify->fix[iforce]->style, "^precession/spin")) {
        lockprecessionspin[count] = dynamic_cast<FixPrecessionSpin *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Find fix langevin/spin/sib (at most one allowed)
  locklangevinspin_sib = nullptr;
  {
    int count = 0;
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (strcmp(modify->fix[iforce]->style, "langevin/spin/sib") == 0) {
        if (count > 0)
          error->all(FLERR, "Only one fix langevin/spin/sib is allowed");
        locklangevinspin_sib = dynamic_cast<FixLangevinSpinSIB *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Find fix glangevin/spin/sib (at most one allowed)
  lockglangevinspin_sib = nullptr;
  {
    int count = 0;
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (strcmp(modify->fix[iforce]->style, "glangevin/spin/sib") == 0) {
        if (count > 0)
          error->all(FLERR, "Only one fix glangevin/spin/sib is allowed");
        lockglangevinspin_sib = dynamic_cast<FixGLangevinSpinSIB *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Find fix landau/spin styles (multiple allowed, one per group)
  delete[] locklandauspin;
  locklandauspin = nullptr;
  nlandauspin = 0;
  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (strcmp(modify->fix[iforce]->style, "landau/spin") == 0)
      nlandauspin++;
  }
  if (nlandauspin > 0) {
    locklandauspin = new FixLandauSpin *[nlandauspin];
    int count = 0;
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (strcmp(modify->fix[iforce]->style, "landau/spin") == 0) {
        locklandauspin[count] = dynamic_cast<FixLandauSpin *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Allocate arrays
  nlocal_max = atom->nlocal;
  grow_arrays();

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix nve/spin/sib: SIB method with spin-lattice coupling\n");
    utils::logmesg(lmp, "Fix nve/spin/sib: dts = dt/2 = {} for half-step updates\n", dts);
    if (pair_spin_ml)
      utils::logmesg(lmp, "Fix nve/spin/sib: PairSpinML detected (spin/step or spin/nep)\n");
    if (npairspin > 0)
      utils::logmesg(lmp, "Fix nve/spin/sib: {} standard PairSpin style(s) detected\n", npairspin);
    if (lattice_flag)
      utils::logmesg(lmp, "Fix nve/spin/sib: Lattice dynamics enabled (spin-lattice coupling)\n");
    else
      utils::logmesg(lmp, "Fix nve/spin/sib: Lattice frozen (pure spin dynamics)\n");
    if (lockglangevinspin_sib)
      utils::logmesg(lmp, "Fix nve/spin/sib: Variable-length spin mode (glangevin/spin/sib detected)\n");
    if (nlandauspin > 0)
      utils::logmesg(lmp, "Fix nve/spin/sib: {} fix landau/spin style(s) detected\n", nlandauspin);
  }

  // Warn if glangevin/spin/sib is used without any longitudinal driving force
  if (lockglangevinspin_sib && nlandauspin == 0 &&
      !pair_ml_has_longitudinal_force(pair_spin_ml))
    error->warning(FLERR, "Fix nve/spin/sib: glangevin/spin/sib detected but no "
                   "longitudinal driving force is available. The active PairSpinML "
                   "style does not provide distribute_full_mag_forces(), and no "
                   "fix landau/spin was found. Longitudinal dynamics has no driving "
                   "force — |m| will diverge under thermal noise.");
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::setup(int vflag)
{
  // Initial force computation
  if (pair_spin_ml) pair_spin_ml->compute(1, 1);
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::grow_arrays()
{
  if (atom->nlocal > nlocal_max) {
    nlocal_max = atom->nlocal + 100;
  }
  memory->grow(s_save, nlocal_max, 3, "fix_nve_spin_sib:s_save");
  memory->grow(mag_save, nlocal_max, "fix_nve_spin_sib:mag_save");
  memory->grow(H_par_save, nlocal_max, "fix_nve_spin_sib:H_par_save");
  memory->grow(noise_vec, nlocal_max, 3, "fix_nve_spin_sib:noise_vec");
  memory->grow(noise_L_vec, nlocal_max, "fix_nve_spin_sib:noise_L_vec");
  memory->grow(fm_full, nlocal_max, 3, "fix_nve_spin_sib:fm_full");
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::initial_integrate(int /*vflag*/)
{
  double dtfm;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *type = atom->type;
  int *mask = atom->mask;

  // Grow arrays if needed
  if (nlocal > nlocal_max) grow_arrays();

  // ========== Step 1: Velocity half-step update ==========
  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (rmass)
          dtfm = dtf / rmass[i];
        else
          dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
  }

  // ========== Step 2: First SIB half-step for spin (dt/2) ==========
  sib_spin_half_step();

  // ========== Step 3: Position full-step update ==========
  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }
    }
  }

  // ========== Step 4: Second SIB half-step for spin (dt/2) ==========
  sib_spin_half_step();
}

/* ----------------------------------------------------------------------
   Perform one SIB half-step update (dt/2) using predictor-corrector
   This requires 2 NN calls per half-step
------------------------------------------------------------------------- */

void FixNVESpinSIB::sib_spin_half_step()
{
  double **sp = atom->sp;
  double **fm = atom->fm;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *mask = atom->mask;

  // --- Step A: Save current spins and initialize noise ---
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      s_save[i][0] = sp[i][0];
      s_save[i][1] = sp[i][1];
      s_save[i][2] = sp[i][2];

      // Save spin magnitude for longitudinal corrector (Heun resets from here)
      mag_save[i] = sp[i][3];
      H_par_save[i] = 0.0;

      // Initialize noise (will be filled by Langevin fix)
      noise_vec[i][0] = 0.0;
      noise_vec[i][1] = 0.0;
      noise_vec[i][2] = 0.0;
      noise_L_vec[i] = 0.0;
    }
  }

  // --- Step B: Compute omega(s_current) - force evaluation #1 at (s^n, |m|^n) ---
  comm->forward_comm();
  if (pair_spin_ml) pair_spin_ml->recompute_forces();

  // Distribute magnetic forces to fm array
  distribute_magnetic_forces();

  // Distribute full (unprojected) magnetic forces for longitudinal dynamics
  if (lockglangevinspin_sib) {
    // Clear fm_full first
    for (int i = 0; i < nlocal; i++)
      fm_full[i][0] = fm_full[i][1] = fm_full[i][2] = 0.0;
    // Only ML contributes to fm_full (standard PairSpin energy is direction-only)
    if (pair_spin_ml)
      pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);
    // Landau on-site longitudinal force
    for (int k = 0; k < nlandauspin; k++)
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          locklandauspin[k]->compute_single_landau(i, sp[i][3], fm_full[i]);
  }

  // Predictor step
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      // Longitudinal predictor: Euler step, sp[i][3] → |m|_pred
      // Stores H_∥_1 in H_par_save for Heun averaging
      if (lockglangevinspin_sib) {
        double spi_L[3] = {sp[i][0], sp[i][1], sp[i][2]};
        lockglangevinspin_sib->compute_longitudinal_predictor(
            i, spi_L, fm_full[i], H_par_save[i], dts);
      }

      double spi[3], fmi[3];
      spi[0] = sp[i][0];
      spi[1] = sp[i][1];
      spi[2] = sp[i][2];
      fmi[0] = fm[i][0];
      fmi[1] = fm[i][1];
      fmi[2] = fm[i][2];

      // Add precession contributions
      if (nprecspin > 0) {
        for (int k = 0; k < nprecspin; k++) {
          lockprecessionspin[k]->compute_single_precession(i, spi, fmi);
        }
      }

      // Add Langevin contributions and store noise
      if (locklangevinspin_sib)
        locklangevinspin_sib->compute_single_langevin_store_noise(i, spi, fmi, noise_vec[i]);

      // Add generalized Langevin transverse contributions and store noise
      if (lockglangevinspin_sib)
        lockglangevinspin_sib->compute_single_langevin_store_noise(i, spi, fmi, noise_vec[i]);

      double F_pred[3];
      F_pred[0] = fmi[0] * dts;
      F_pred[1] = fmi[1] * dts;
      F_pred[2] = fmi[2] * dts;

      double s_pred[3];

      // Solve implicit equation
      solve_implicit_sib(s_save[i], F_pred, s_pred);

      // Compute midpoint: s_mid = (s_save + s_pred) / 2
      double s_mid[3];
      s_mid[0] = 0.5 * (s_save[i][0] + s_pred[0]);
      s_mid[1] = 0.5 * (s_save[i][1] + s_pred[1]);
      s_mid[2] = 0.5 * (s_save[i][2] + s_pred[2]);

      // Temporarily update sp to midpoint for field calculation
      sp[i][0] = s_mid[0];
      sp[i][1] = s_mid[1];
      sp[i][2] = s_mid[2];
      // sp[i][3] is now |m|_pred (set by longitudinal predictor above)
    }
  }

  // --- Step C: Compute omega(s_mid, |m|_pred) - force evaluation #2 ---
  comm->forward_comm();
  if (pair_spin_ml) pair_spin_ml->recompute_forces();

  // Distribute magnetic forces to fm array
  distribute_magnetic_forces();

  // Distribute full (unprojected) magnetic forces for longitudinal dynamics
  if (lockglangevinspin_sib) {
    for (int i = 0; i < nlocal; i++)
      fm_full[i][0] = fm_full[i][1] = fm_full[i][2] = 0.0;
    if (pair_spin_ml)
      pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);
    // Landau on-site longitudinal force
    for (int k = 0; k < nlandauspin; k++)
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          locklandauspin[k]->compute_single_landau(i, sp[i][3], fm_full[i]);
  }

  // Corrector step
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double spi[3], fmi[3];
      spi[0] = sp[i][0];  // This is s_mid
      spi[1] = sp[i][1];
      spi[2] = sp[i][2];
      fmi[0] = fm[i][0];
      fmi[1] = fm[i][1];
      fmi[2] = fm[i][2];

      // Add precession contributions at midpoint
      if (nprecspin > 0) {
        for (int k = 0; k < nprecspin; k++) {
          lockprecessionspin[k]->compute_single_precession(i, spi, fmi);
        }
      }

      // Add Langevin contributions using SAME noise as predictor step
      if (locklangevinspin_sib)
        locklangevinspin_sib->compute_single_langevin_reuse_noise(i, spi, fmi, noise_vec[i]);

      // Add generalized Langevin transverse contributions using SAME noise
      if (lockglangevinspin_sib)
        lockglangevinspin_sib->compute_single_langevin_reuse_noise(i, spi, fmi, noise_vec[i]);

      double F_corr[3];
      F_corr[0] = fmi[0] * dts;
      F_corr[1] = fmi[1] * dts;
      F_corr[2] = fmi[2] * dts;

      double s_new[3];
      // Solve implicit equation from saved state
      solve_implicit_sib(s_save[i], F_corr, s_new);

      // Expose the effective field for diagnostics
      fm[i][0] = fmi[0];
      fm[i][1] = fmi[1];
      fm[i][2] = fmi[2];

      // Update spin direction to new value
      sp[i][0] = s_new[0];
      sp[i][1] = s_new[1];
      sp[i][2] = s_new[2];

      // Longitudinal corrector: Heun trapezoidal step + noise
      // Uses H_par_save (from predictor) and fm_full (from NN#2).
      // Project fm_full onto the predicted midpoint orientation spi=s_mid,
      // which is the state at which fm_full was evaluated.
      // Resets from mag_save to apply the averaged drift.
      if (lockglangevinspin_sib) {
        lockglangevinspin_sib->compute_longitudinal_corrector(
            i, spi, fm_full[i], H_par_save[i], mag_save[i],
            noise_L_vec[i], dts);
      }
    }
  }

  // Communicate updated spins
  comm->forward_comm();
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::final_integrate()
{
  double dtfm;

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *type = atom->type;
  int *mask = atom->mask;

  // Update half v for all particles
  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (rmass)
          dtfm = dtf / rmass[i];
        else
          dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Solve the implicit SIB equation:
     Y = X + (X + Y)/2 x F

   The solution using the analytical inverse is:
     M = (X + Y)/2 = [X + 0.5*(X x F) + 0.25*(F.X)*F] / (1 + |F|^2/4)
     Y = 2*M - X

   This exactly preserves |Y| = |X| = 1 for transverse dynamics.
   Spin magnitude changes are handled separately via sp[3].
------------------------------------------------------------------------- */

void FixNVESpinSIB::solve_implicit_sib(double *s_in, double *F_vec, double *s_out)
{
  double Fx = F_vec[0];
  double Fy = F_vec[1];
  double Fz = F_vec[2];

  double sx = s_in[0];
  double sy = s_in[1];
  double sz = s_in[2];

  // |F|^2
  double F_sq = Fx*Fx + Fy*Fy + Fz*Fz;

  // Denominator: 1 + |F|^2/4
  double denom = 1.0 + 0.25 * F_sq;

  // Cross product: X x F
  double cross_x = sy * Fz - sz * Fy;
  double cross_y = sz * Fx - sx * Fz;
  double cross_z = sx * Fy - sy * Fx;

  // Dot product: F . X
  double dot_FX = Fx * sx + Fy * sy + Fz * sz;

  // Compute midpoint M
  double Mx = (sx - 0.5 * cross_x + 0.25 * dot_FX * Fx) / denom;
  double My = (sy - 0.5 * cross_y + 0.25 * dot_FX * Fy) / denom;
  double Mz = (sz - 0.5 * cross_z + 0.25 * dot_FX * Fz) / denom;

  // Compute Y = 2*M - X
  s_out[0] = 2.0 * Mx - sx;
  s_out[1] = 2.0 * My - sy;
  s_out[2] = 2.0 * Mz - sz;

  // Renormalize for numerical stability (sp[0:2] is always unit vector)
  double norm = sqrt(s_out[0]*s_out[0] + s_out[1]*s_out[1] + s_out[2]*s_out[2]);
  if (norm > 1e-10) {
    s_out[0] /= norm;
    s_out[1] /= norm;
    s_out[2] /= norm;
  }
}

/* ----------------------------------------------------------------------
   Distribute cached magnetic forces from pair_spin_step to fm array
------------------------------------------------------------------------- */

void FixNVESpinSIB::distribute_magnetic_forces()
{
  // Clear fm array first (it accumulates)
  double **fm = atom->fm;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      fm[i][0] = 0.0;
      fm[i][1] = 0.0;
      fm[i][2] = 0.0;
    }
  }

  // ML forces (from cached NN output)
  if (pair_spin_ml)
    pair_spin_ml->distribute_cached_mag_forces();

  // Standard PairSpin forces (per-atom accumulation)
  for (int k = 0; k < npairspin; k++)
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        spin_pairs[k]->compute_single_pair(i, fm[i]);
}
