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
#include "fix_precession_spin.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "pair_spin_ml.h"
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

/* ---------------------------------------------------------------------- */

FixNVESpinSIB::FixNVESpinSIB(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  pair_spin_ml(nullptr), locklangevinspin_sib(nullptr),
  lockprecessionspin(nullptr), s_save(nullptr), noise_vec(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_nve_spin_sib);

  if (narg < 3) error->all(FLERR, "Illegal fix nve/spin/sib command");

  time_integrate = 1;
  lattice_flag = 1;
  nlocal_max = 0;

  // Initialize flags
  nprecspin = nlangspin_sib = 0;
  precession_spin_flag = 0;
  maglangevin_sib_flag = 0;

  // Check if atom map is defined
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Fix nve/spin/sib requires an atom map, see atom_modify");

  // Parse optional arguments
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

  // Check if atom/spin style is defined
  if (!atom->sp_flag)
    error->all(FLERR, "Fix nve/spin/sib requires atom_style spin");
}

/* ---------------------------------------------------------------------- */

FixNVESpinSIB::~FixNVESpinSIB()
{
  memory->destroy(s_save);
  memory->destroy(noise_vec);
  delete[] locklangevinspin_sib;
  delete[] lockprecessionspin;
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
  if (!pair_spin_ml)
    error->all(FLERR, "Fix nve/spin/sib requires pair_style spin/step or spin/nep");

  // Find fix precession/spin styles
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
        precession_spin_flag = 1;
        lockprecessionspin[count] = dynamic_cast<FixPrecessionSpin *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Find fix langevin/spin/sib styles
  nlangspin_sib = 0;
  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (strcmp(modify->fix[iforce]->style, "langevin/spin/sib") == 0) {
      nlangspin_sib++;
    }
  }

  if (nlangspin_sib > 0) {
    delete[] locklangevinspin_sib;
    locklangevinspin_sib = new FixLangevinSpinSIB *[nlangspin_sib];

    int count = 0;
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (strcmp(modify->fix[iforce]->style, "langevin/spin/sib") == 0) {
        maglangevin_sib_flag = 1;
        locklangevinspin_sib[count] = dynamic_cast<FixLangevinSpinSIB *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Allocate arrays
  nlocal_max = atom->nlocal;
  grow_arrays();

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix nve/spin/sib: SIB method with spin-lattice coupling\n");
    utils::logmesg(lmp, "Fix nve/spin/sib: Using PairSpinML base class interface\n");
    utils::logmesg(lmp, "Fix nve/spin/sib: dts = dt/2 = {} for half-step updates\n", dts);
    utils::logmesg(lmp, "Fix nve/spin/sib: 4 NN calls per timestep (2 per half-step)\n");
    if (lattice_flag)
      utils::logmesg(lmp, "Fix nve/spin/sib: Lattice dynamics enabled (spin-lattice coupling)\n");
    else
      utils::logmesg(lmp, "Fix nve/spin/sib: Lattice frozen (pure spin dynamics)\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::setup(int vflag)
{
  // Initial force computation
  pair_spin_ml->compute(1, 1);
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::grow_arrays()
{
  if (atom->nlocal > nlocal_max) {
    nlocal_max = atom->nlocal + 100;
  }
  memory->grow(s_save, nlocal_max, 3, "fix_nve_spin_sib:s_save");
  memory->grow(noise_vec, nlocal_max, 3, "fix_nve_spin_sib:noise_vec");
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

      // Initialize noise (will be filled by Langevin fix)
      noise_vec[i][0] = 0.0;
      noise_vec[i][1] = 0.0;
      noise_vec[i][2] = 0.0;
    }
  }

  // --- Step B: Compute omega(s_current) - NN call #1 ---
  comm->forward_comm();
  pair_spin_ml->recompute_forces();

  // Distribute magnetic forces from cached tensor to fm array
  distribute_magnetic_forces();

  // Predictor step
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double spi[3], fmi[3];
      spi[0] = sp[i][0];
      spi[1] = sp[i][1];
      spi[2] = sp[i][2];
      fmi[0] = fm[i][0];
      fmi[1] = fm[i][1];
      fmi[2] = fm[i][2];

      // Add precession contributions
      if (precession_spin_flag) {
        for (int k = 0; k < nprecspin; k++) {
          lockprecessionspin[k]->compute_single_precession(i, spi, fmi);
        }
      }

      // Add Langevin contributions and store noise
      if (maglangevin_sib_flag) {
        for (int k = 0; k < nlangspin_sib; k++) {
          locklangevinspin_sib[k]->compute_single_langevin_store_noise(i, spi, fmi, noise_vec[i]);
        }
      }

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
    }
  }

  // --- Step C: Compute omega(s_mid) - NN call #2 ---
  comm->forward_comm();
  pair_spin_ml->recompute_forces();

  // Distribute magnetic forces from cached tensor to fm array
  distribute_magnetic_forces();

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
      if (precession_spin_flag) {
        for (int k = 0; k < nprecspin; k++) {
          lockprecessionspin[k]->compute_single_precession(i, spi, fmi);
        }
      }

      // Add Langevin contributions using SAME noise as predictor step
      if (maglangevin_sib_flag) {
        for (int k = 0; k < nlangspin_sib; k++) {
          locklangevinspin_sib[k]->compute_single_langevin_reuse_noise(i, spi, fmi, noise_vec[i]);
        }
      }

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

      // Update spin to new value
      sp[i][0] = s_new[0];
      sp[i][1] = s_new[1];
      sp[i][2] = s_new[2];
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

   This exactly preserves |Y| = |X| = 1.
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

  // Renormalize for numerical stability
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

  // Distribute cached magnetic forces from pair_spin_ml
  pair_spin_ml->distribute_cached_mag_forces();
}
