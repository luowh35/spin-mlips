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
   Combined spin+lattice minimizer (Alternating Optimization):
   - FIRE algorithm for lattice optimization
   - Riemannian L-BFGS for spin optimization
   - Each outer iteration: 1 FIRE atom step, then N L-BFGS spin substeps
   - L-BFGS history is preserved within spin substeps (atoms fixed)
------------------------------------------------------------------------- */

#include "min_spin_lattice.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "output.h"
#include "pair.h"
#include "pair_spin_ml.h"
#include "timer.h"
#include "universe.h"
#include "update.h"

#include <cmath>
#include <cstring>

#include "fmt/format.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define EPS_ENERGY 1.0e-8
#define DELAYSTEP 5

/* ---------------------------------------------------------------------- */

MinSpinLattice::MinSpinLattice(LAMMPS *lmp) : Min(lmp),
  g_atom(nullptr), g_spin(nullptr), p_atom(nullptr), p_spin(nullptr),
  v_atom(nullptr), fm_full(nullptr), pair_spin_ml(nullptr),
  s_spin(nullptr), y_spin(nullptr), rho_spin(nullptr), alpha_lbfgs(nullptr), g_spin_old(nullptr)
{
  nlocal_max = 0;
  fm_full_allocated = 0;

  // FIRE parameters for lattice optimization
  dt_atom = 0.1;
  dt_max_atom = 1.0;
  alpha_fire = 0.1;
  f_inc = 1.1;
  f_dec = 0.5;
  n_min = 5;
  last_negative = 0;

  // L-BFGS parameters for spin optimization
  lbfgs_mem = 10;              // store last 10 history vectors
  lbfgs_iter = 0;
  spin_substeps = 10;          // L-BFGS spin steps per FIRE atom step

  // Spin steepest descent step size (same as optimize.py spin_step_size)
  alpha_init = 40.0;
  alpha_min = 1.0e-10;
  c1 = 1.0e-4;              // Armijo constant (unused in SD mode)
  backtrack_factor = 0.5;   // shrink factor (unused in SD mode)

  // Scale balancing (unused in alternating mode, kept for future)
  lambda = 1.0;
  lambda_max = 10.0;
  lambda_scale = 1.0;
  eps_lambda = 1.0e-10;
}

/* ---------------------------------------------------------------------- */

MinSpinLattice::~MinSpinLattice()
{
  memory->destroy(g_atom);
  memory->destroy(g_spin);
  memory->destroy(g_spin_old);
  memory->destroy(p_atom);
  memory->destroy(p_spin);
  memory->destroy(v_atom);
  memory->destroy(fm_full);

  // Free L-BFGS history
  if (s_spin) {
    for (int i = 0; i < lbfgs_mem; i++) {
      memory->destroy(s_spin[i]);
      memory->destroy(y_spin[i]);
    }
    delete[] s_spin;
    delete[] y_spin;
  }
  memory->destroy(rho_spin);
  memory->destroy(alpha_lbfgs);
}

/* ---------------------------------------------------------------------- */

void MinSpinLattice::init()
{
  Min::init();

  // Detect PairSpinML
  pair_spin_ml = dynamic_cast<PairSpinML*>(force->pair);

  // allocate arrays
  nlocal_max = atom->nlocal;
  memory->grow(g_atom, 3*nlocal_max, "min_spin_lattice:g_atom");
  memory->grow(g_spin, 3*nlocal_max, "min_spin_lattice:g_spin");
  memory->grow(g_spin_old, 3*nlocal_max, "min_spin_lattice:g_spin_old");
  memory->grow(p_atom, 3*nlocal_max, "min_spin_lattice:p_atom");
  memory->grow(p_spin, 3*nlocal_max, "min_spin_lattice:p_spin");
  memory->grow(v_atom, 3*nlocal_max, "min_spin_lattice:v_atom");

  // Initialize FIRE velocities to zero
  for (int i = 0; i < 3*nlocal_max; i++) v_atom[i] = 0.0;

  // Allocate L-BFGS history arrays
  s_spin = new double*[lbfgs_mem];
  y_spin = new double*[lbfgs_mem];
  for (int i = 0; i < lbfgs_mem; i++) {
    memory->create(s_spin[i], 3*nlocal_max, "min_spin_lattice:s_spin");
    memory->create(y_spin[i], 3*nlocal_max, "min_spin_lattice:y_spin");
  }
  memory->create(rho_spin, lbfgs_mem, "min_spin_lattice:rho_spin");
  memory->create(alpha_lbfgs, lbfgs_mem, "min_spin_lattice:alpha_lbfgs");

  lbfgs_iter = 0;

  // Allocate fm_full if using PairSpinML
  if (pair_spin_ml) {
    memory->create(fm_full, nlocal_max, 3, "min_spin_lattice:fm_full");
    fm_full_allocated = nlocal_max;
  }
}

/* ---------------------------------------------------------------------- */

void MinSpinLattice::setup_style()
{
  double **v = atom->v;
  int nlocal = atom->nlocal;

  if (!atom->sp_flag)
    error->all(FLERR,"min/spin/lattice requires atom/spin style");

  if (comm->me == 0) {
    if (pair_spin_ml) {
      utils::logmesg(lmp, fmt::format(
        "  min/spin/lattice: FIRE + spin steepest descent (callback)\n"
        "    Lattice: FIRE algorithm\n"
        "    Spin: steepest descent (alpha_S={:.1f}, using PairSpinML fm_full)\n",
        alpha_init));
    } else {
      utils::logmesg(lmp, fmt::format(
        "  min/spin/lattice: FIRE + spin steepest descent (callback)\n"
        "    Lattice: FIRE algorithm\n"
        "    Spin: steepest descent (alpha_S={:.1f}, using standard PairSpin)\n",
        alpha_init));
    }
  }

  // zero velocities (not used in CG, but for consistency)
  for (int i = 0; i < nlocal; i++)
    v[i][0] = v[i][1] = v[i][2] = 0.0;
}

/* ---------------------------------------------------------------------- */

int MinSpinLattice::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"alpha_init") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    alpha_init = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"c1") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    c1 = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"backtrack_factor") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    backtrack_factor = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"lambda_max") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    lambda_max = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"lambda_scale") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    lambda_scale = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  // FIRE parameters
  if (strcmp(arg[0],"dt_atom") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    dt_atom = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"dt_max_atom") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    dt_max_atom = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"alpha_fire") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    alpha_fire = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"n_min") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    n_min = utils::inumeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  // L-BFGS spin parameters
  if (strcmp(arg[0],"spin_substeps") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    spin_substeps = utils::inumeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  if (strcmp(arg[0],"lbfgs_mem") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal min_modify command");
    lbfgs_mem = utils::inumeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void MinSpinLattice::reset_vectors()
{
  nvec = 3 * atom->nlocal;
  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];
}

/* ---------------------------------------------------------------------- */

int MinSpinLattice::iterate(int maxiter)
{
  bigint ntimestep;
  int flag, flagall;

  alpha_final = 0.0;

  int nlocal = atom->nlocal;
  realloc_arrays(nlocal);

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++update->ntimestep;
    niter++;

    // Check nlocal and reallocate if needed
    nlocal = atom->nlocal;
    realloc_arrays(nlocal);

    // =============================================
    // Phase 1: Compute energy and forces (full)
    // =============================================

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;
    if (neval >= update->max_eval) return MAXEVAL;

    // Recheck nlocal after energy_force (atom migration)
    nlocal = atom->nlocal;
    realloc_arrays(nlocal);

    // =============================================
    // Phase 2: FIRE step for atoms (1 step)
    // =============================================

    advance_atoms_fire();

    // =============================================
    // Phase 3: Single steepest-descent spin update (no extra energy_force)
    // Uses magforces from Phase 1, projects to tangent space, large fixed step
    // Same approach as optimize.py spin_update_callback
    // =============================================

    {
      double **sp = atom->sp;
      nlocal = atom->nlocal;

      if (pair_spin_ml) {
        // fm_full = -dE/dm (raw energy gradient, unprojected)
        pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);

        for (int i = 0; i < nlocal; i++) {
          double mag = sp[i][3];

          // Project to tangent space: f_perp = fm_full - (fm_full . S) * S
          double dot = fm_full[i][0]*sp[i][0] + fm_full[i][1]*sp[i][1] + fm_full[i][2]*sp[i][2];
          double fpx = fm_full[i][0] - dot * sp[i][0];
          double fpy = fm_full[i][1] - dot * sp[i][1];
          double fpz = fm_full[i][2] - dot * sp[i][2];

          // Steepest descent: S_new = S + alpha_S * f_perp
          // fm_full = -dE/dm, so f_perp points downhill already
          sp[i][0] += alpha_init * fpx;
          sp[i][1] += alpha_init * fpy;
          sp[i][2] += alpha_init * fpz;

          // Renormalize to unit sphere, restore magnitude
          double norm = sqrt(sp[i][0]*sp[i][0] + sp[i][1]*sp[i][1] + sp[i][2]*sp[i][2]);
          if (norm > 1.0e-14) {
            sp[i][0] /= norm;
            sp[i][1] /= norm;
            sp[i][2] /= norm;
          }
          sp[i][3] = mag;
        }
      } else {
        // Standard PairSpin: fm = H_eff
        // Force = mag * fm, project to tangent space
        double **fm = atom->fm;

        for (int i = 0; i < nlocal; i++) {
          double mag = sp[i][3];
          double fx = mag * fm[i][0];
          double fy = mag * fm[i][1];
          double fz = mag * fm[i][2];

          double dot = fx*sp[i][0] + fy*sp[i][1] + fz*sp[i][2];
          double fpx = fx - dot * sp[i][0];
          double fpy = fy - dot * sp[i][1];
          double fpz = fz - dot * sp[i][2];

          sp[i][0] += alpha_init * fpx;
          sp[i][1] += alpha_init * fpy;
          sp[i][2] += alpha_init * fpz;

          double norm = sqrt(sp[i][0]*sp[i][0] + sp[i][1]*sp[i][1] + sp[i][2]*sp[i][2]);
          if (norm > 1.0e-14) {
            sp[i][0] /= norm;
            sp[i][1] /= norm;
            sp[i][2] /= norm;
          }
          sp[i][3] = mag;
        }
      }
    }

    // =============================================
    // Phase 4: Convergence checks
    // =============================================

    // Debug output for first few iterations
    if (iter < 10 && comm->me == 0) {
      nlocal = atom->nlocal;
      double gnorm_atom_local = 0.0, gnorm_spin_local = 0.0;
      double **f = atom->f;
      double **sp = atom->sp;
      double **fm = atom->fm;
      for (int i = 0; i < nlocal; i++) {
        gnorm_atom_local += f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2];
        // torque magnitude for display
        double tx = fm[i][1]*sp[i][2] - fm[i][2]*sp[i][1];
        double ty = fm[i][2]*sp[i][0] - fm[i][0]*sp[i][2];
        double tz = fm[i][0]*sp[i][1] - fm[i][1]*sp[i][0];
        gnorm_spin_local += tx*tx + ty*ty + tz*tz;
      }
      double gnorm_atom_global, gnorm_spin_global;
      MPI_Allreduce(&gnorm_atom_local, &gnorm_atom_global, 1, MPI_DOUBLE, MPI_SUM, world);
      MPI_Allreduce(&gnorm_spin_local, &gnorm_spin_global, 1, MPI_DOUBLE, MPI_SUM, world);

      utils::logmesg(lmp, fmt::format(
        "  Iter {}: E={:.6f}, ||f_atom||={:.3e}, ||torque||={:.3e}\n",
        iter, ecurrent, sqrt(gnorm_atom_global), sqrt(gnorm_spin_global)));
    }

    // Energy tolerance
    if (update->etol > 0.0) {
      if (update->multireplica == 0) {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          return ETOL;
      } else {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return ETOL;
      }
    }

    // Combined force tolerance: max(atomic_force, spin_torque)
    if (update->ftol > 0.0) {
      double fnorm, tnorm, combined;

      if (normstyle == MAX) fnorm = sqrt(fnorm_max());
      else if (normstyle == INF) fnorm = sqrt(fnorm_inf());
      else if (normstyle == TWO) fnorm = sqrt(fnorm_sqr());
      else error->all(FLERR,"Illegal min_modify command");

      if (normstyle == MAX) tnorm = max_torque();
      else if (normstyle == INF) tnorm = inf_torque();
      else tnorm = total_torque();

      combined = MAX(fnorm, tnorm);

      if (update->multireplica == 0) {
        if (combined < update->ftol) return FTOL;
      } else {
        if (combined < update->ftol) flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return FTOL;
      }
    }

    // Output for thermo, dump, restart
    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

/* ----------------------------------------------------------------------
   calculate atomic gradient (negative force)
------------------------------------------------------------------------- */

void MinSpinLattice::calc_atom_gradient()
{
  int nlocal = atom->nlocal;
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++) {
    g_atom[3*i + 0] = -f[i][0];
    g_atom[3*i + 1] = -f[i][1];
    g_atom[3*i + 2] = -f[i][2];
  }
}

/* ----------------------------------------------------------------------
   calculate spin gradient (Riemannian gradient on S^2)

   For PairSpinML: use fm_full = -dE/dm (raw energy gradient)
   For standard PairSpin: use -mag * fm (fm = H_eff)

   Then project to tangent space: g = grad - (grad . S) * S
   This gives the Riemannian gradient for minimization on the sphere.
------------------------------------------------------------------------- */

void MinSpinLattice::calc_spin_gradient()
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;

  if (pair_spin_ml) {
    // Get unprojected magnetic forces: fm_full = -dE/dm
    pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);

    for (int i = 0; i < nlocal; i++) {
      // fm_full is -dE/dm, so gradient for minimization is +dE/dm = -fm_full
      double gx = -fm_full[i][0];
      double gy = -fm_full[i][1];
      double gz = -fm_full[i][2];

      // Project to tangent space: g - (g . S) * S
      double dot = gx*sp[i][0] + gy*sp[i][1] + gz*sp[i][2];
      g_spin[3*i + 0] = gx - dot * sp[i][0];
      g_spin[3*i + 1] = gy - dot * sp[i][1];
      g_spin[3*i + 2] = gz - dot * sp[i][2];
    }
  } else {
    // Standard PairSpin: fm = H_eff (effective field)
    // Energy gradient: dE/dS = -mag * H_eff = -mag * fm
    double **fm = atom->fm;

    for (int i = 0; i < nlocal; i++) {
      double mag = sp[i][3];
      double gx = -mag * fm[i][0];
      double gy = -mag * fm[i][1];
      double gz = -mag * fm[i][2];

      // Project to tangent space
      double dot = gx*sp[i][0] + gy*sp[i][1] + gz*sp[i][2];
      g_spin[3*i + 0] = gx - dot * sp[i][0];
      g_spin[3*i + 1] = gy - dot * sp[i][1];
      g_spin[3*i + 2] = gz - dot * sp[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   realloc_arrays: grow all working arrays if nlocal increased
   Also grows L-BFGS history arrays and fm_full
------------------------------------------------------------------------- */

void MinSpinLattice::realloc_arrays(int nlocal)
{
  if (nlocal_max >= nlocal) return;

  int old_max = nlocal_max;
  nlocal_max = nlocal;

  memory->grow(g_atom, 3*nlocal_max, "min_spin_lattice:g_atom");
  memory->grow(g_spin, 3*nlocal_max, "min_spin_lattice:g_spin");
  memory->grow(g_spin_old, 3*nlocal_max, "min_spin_lattice:g_spin_old");
  memory->grow(p_atom, 3*nlocal_max, "min_spin_lattice:p_atom");
  memory->grow(p_spin, 3*nlocal_max, "min_spin_lattice:p_spin");
  memory->grow(v_atom, 3*nlocal_max, "min_spin_lattice:v_atom");

  // Zero new velocity entries
  for (int i = 3*old_max; i < 3*nlocal_max; i++) v_atom[i] = 0.0;

  if (pair_spin_ml) {
    memory->grow(fm_full, nlocal_max, 3, "min_spin_lattice:fm_full");
    fm_full_allocated = nlocal_max;
  }

  // Grow L-BFGS history arrays
  if (s_spin) {
    for (int i = 0; i < lbfgs_mem; i++) {
      memory->grow(s_spin[i], 3*nlocal_max, "min_spin_lattice:s_spin");
      memory->grow(y_spin[i], 3*nlocal_max, "min_spin_lattice:y_spin");
    }
  }

  // Reset L-BFGS when nlocal changes (history vectors are invalid)
  lbfgs_iter = 0;
}

/* ----------------------------------------------------------------------
   spin_lbfgs_substep: single L-BFGS spin optimization substep
   Atoms are fixed. Only spins are updated.
   Returns number of energy evaluations used.

   Steps:
   1. Recompute spin forces (atoms fixed, spins may have changed)
   2. Compute Riemannian gradient
   3. Update L-BFGS history from previous substep
   4. Compute L-BFGS search direction
   5. Armijo backtracking line search along search direction
   6. Advance spins
------------------------------------------------------------------------- */

int MinSpinLattice::spin_lbfgs_substep()
{
  int nlocal = atom->nlocal;
  int sub_neval = 0;

  // --- Step 1: Recompute forces (spins may have changed from previous substep) ---
  // For the first substep, forces are already computed by iterate()
  // For subsequent substeps, we need to recompute

  if (lbfgs_iter > 0) {
    // Need to recompute energy and forces since spins changed
    ecurrent = energy_force(0);
    sub_neval++;

    nlocal = atom->nlocal;
    realloc_arrays(nlocal);
  }

  // --- Step 2: Compute Riemannian gradient ---

  calc_spin_gradient();

  // --- Step 3: Update L-BFGS history ---

  if (lbfgs_iter > 0) {
    update_lbfgs_history(alpha_final);
  }

  // Save current gradient for next update
  for (int i = 0; i < 3*nlocal; i++) {
    g_spin_old[i] = g_spin[i];
  }

  // --- Step 4: Compute L-BFGS search direction ---

  calc_spin_lbfgs_direction();

  // --- Step 5: Armijo backtracking line search ---

  // Compute directional derivative: dE/dalpha = g . p
  double gp_local = 0.0, gp_global;
  for (int i = 0; i < 3*nlocal; i++) {
    gp_local += g_spin[i] * p_spin[i];
  }
  MPI_Allreduce(&gp_local, &gp_global, 1, MPI_DOUBLE, MPI_SUM, world);

  // If search direction is not a descent direction, fall back to steepest descent
  if (gp_global >= 0.0) {
    for (int i = 0; i < 3*nlocal; i++) {
      p_spin[i] = -g_spin[i];
    }
    gp_local = 0.0;
    for (int i = 0; i < 3*nlocal; i++) {
      gp_local += g_spin[i] * p_spin[i];
    }
    MPI_Allreduce(&gp_local, &gp_global, 1, MPI_DOUBLE, MPI_SUM, world);

    // Also reset L-BFGS history since direction was bad
    lbfgs_iter = 0;
  }

  // Backtracking line search
  double alpha = alpha_init;
  double e0 = ecurrent;

  // Save current spin positions
  double **sp = atom->sp;
  double *sp_save;
  memory->create(sp_save, 4*nlocal, "min_spin_lattice:sp_save");
  for (int i = 0; i < nlocal; i++) {
    sp_save[4*i+0] = sp[i][0];
    sp_save[4*i+1] = sp[i][1];
    sp_save[4*i+2] = sp[i][2];
    sp_save[4*i+3] = sp[i][3];
  }

  int ls_iter = 0;
  int max_ls = 10;

  while (ls_iter < max_ls) {
    // Restore spins
    for (int i = 0; i < nlocal; i++) {
      sp[i][0] = sp_save[4*i+0];
      sp[i][1] = sp_save[4*i+1];
      sp[i][2] = sp_save[4*i+2];
      sp[i][3] = sp_save[4*i+3];
    }

    // Trial step
    advance_spins(alpha);

    // Evaluate energy
    double e_trial = energy_force(0);
    sub_neval++;

    // Armijo condition: e_trial <= e0 + c1 * alpha * gp
    if (e_trial <= e0 + c1 * alpha * gp_global) {
      ecurrent = e_trial;
      break;
    }

    alpha *= backtrack_factor;

    if (alpha < alpha_min) {
      // Line search failed, accept current position with minimal step
      for (int i = 0; i < nlocal; i++) {
        sp[i][0] = sp_save[4*i+0];
        sp[i][1] = sp_save[4*i+1];
        sp[i][2] = sp_save[4*i+2];
        sp[i][3] = sp_save[4*i+3];
      }
      advance_spins(alpha_min);
      ecurrent = energy_force(0);
      sub_neval++;
      alpha = alpha_min;
      break;
    }

    ls_iter++;
  }

  memory->destroy(sp_save);

  alpha_final = alpha;
  lbfgs_iter++;

  return sub_neval;
}

/* ----------------------------------------------------------------------
   advance atoms using FIRE algorithm
   FIRE: Fast Inertial Relaxation Engine
   - Uses velocity-based acceleration
   - Adaptive time step
   - More efficient than simple gradient descent for lattice relaxation
------------------------------------------------------------------------- */

void MinSpinLattice::advance_atoms_fire()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;

  // Compute power P = F · v (globally)
  double P_local = 0.0;
  for (int i = 0; i < nlocal; i++) {
    P_local += f[i][0] * v_atom[3*i + 0];
    P_local += f[i][1] * v_atom[3*i + 1];
    P_local += f[i][2] * v_atom[3*i + 2];
  }

  double P_global;
  MPI_Allreduce(&P_local, &P_global, 1, MPI_DOUBLE, MPI_SUM, world);

  if (update->multireplica == 1) {
    double P_temp = P_global;
    MPI_Allreduce(&P_temp, &P_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  }

  if (P_global > 0.0) {
    // Positive power: acceleration phase
    last_negative++;

    if (last_negative > n_min) {
      dt_atom = MIN(dt_atom * f_inc, dt_max_atom);
      alpha_fire = alpha_fire * f_dec;
    }

    // Mix velocity: v = (1-alpha)*v + alpha*|v|*F/|F|
    double vnorm_local = 0.0, fnorm_local = 0.0;
    for (int i = 0; i < 3*nlocal; i++) {
      vnorm_local += v_atom[i] * v_atom[i];
      fnorm_local += f[i/3][i%3] * f[i/3][i%3];
    }

    double vnorm_global, fnorm_global;
    MPI_Allreduce(&vnorm_local, &vnorm_global, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&fnorm_local, &fnorm_global, 1, MPI_DOUBLE, MPI_SUM, world);

    if (update->multireplica == 1) {
      double temp;
      temp = vnorm_global;
      MPI_Allreduce(&temp, &vnorm_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
      temp = fnorm_global;
      MPI_Allreduce(&temp, &fnorm_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    }

    double vnorm = sqrt(vnorm_global);
    double fnorm = sqrt(fnorm_global);

    if (fnorm > 1e-10) {
      for (int i = 0; i < nlocal; i++) {
        v_atom[3*i + 0] = (1.0 - alpha_fire) * v_atom[3*i + 0]
                          + alpha_fire * vnorm * f[i][0] / fnorm;
        v_atom[3*i + 1] = (1.0 - alpha_fire) * v_atom[3*i + 1]
                          + alpha_fire * vnorm * f[i][1] / fnorm;
        v_atom[3*i + 2] = (1.0 - alpha_fire) * v_atom[3*i + 2]
                          + alpha_fire * vnorm * f[i][2] / fnorm;
      }
    }

  } else {
    // Negative power: reset
    fire_reset();
  }

  // Update velocity and position
  for (int i = 0; i < nlocal; i++) {
    v_atom[3*i + 0] += dt_atom * f[i][0];
    v_atom[3*i + 1] += dt_atom * f[i][1];
    v_atom[3*i + 2] += dt_atom * f[i][2];

    x[i][0] += dt_atom * v_atom[3*i + 0];
    x[i][1] += dt_atom * v_atom[3*i + 1];
    x[i][2] += dt_atom * v_atom[3*i + 2];
  }
}

/* ----------------------------------------------------------------------
   reset FIRE parameters when power becomes negative
------------------------------------------------------------------------- */

void MinSpinLattice::fire_reset()
{
  int nlocal = atom->nlocal;

  last_negative = 0;
  dt_atom *= f_dec;
  alpha_fire = 0.1;

  for (int i = 0; i < 3*nlocal; i++) {
    v_atom[i] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   advance spins using retraction (normalization)
   S_new = normalize(S + alpha * p_S)
------------------------------------------------------------------------- */

void MinSpinLattice::advance_spins(double alpha)
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;

  for (int i = 0; i < nlocal; i++) {
    double mag = sp[i][3];

    // Update: S_new = S + alpha * p_S
    sp[i][0] += alpha * p_spin[3*i + 0];
    sp[i][1] += alpha * p_spin[3*i + 1];
    sp[i][2] += alpha * p_spin[3*i + 2];

    // Normalize to maintain sphere constraint
    double norm = sqrt(sp[i][0]*sp[i][0] + sp[i][1]*sp[i][1] + sp[i][2]*sp[i][2]);
    if (norm > 1.0e-14) {
      sp[i][0] /= norm;
      sp[i][1] /= norm;
      sp[i][2] /= norm;
    }

    // Restore fixed magnitude
    sp[i][3] = mag;
  }
}

/* ----------------------------------------------------------------------
   Calculate L-BFGS search direction for spins (Riemannian L-BFGS)
   Uses two-loop recursion with vector transport
------------------------------------------------------------------------- */

void MinSpinLattice::calc_spin_lbfgs_direction()
{
  int nlocal = atom->nlocal;

  if (lbfgs_iter == 0) {
    // First iteration: use steepest descent
    for (int i = 0; i < 3*nlocal; i++) {
      p_spin[i] = -g_spin[i];
    }
    return;
  }

  // L-BFGS two-loop recursion
  // q = current gradient
  double *q;
  memory->create(q, 3*nlocal, "min_spin_lattice:q");
  for (int i = 0; i < 3*nlocal; i++) {
    q[i] = g_spin[i];
  }

  // Determine how many history vectors to use
  int m = MIN(lbfgs_iter, lbfgs_mem);

  // First loop: backward
  for (int i = m - 1; i >= 0; i--) {
    int idx = (lbfgs_iter - 1 - (m - 1 - i)) % lbfgs_mem;

    double sq = 0.0, sq_global;
    for (int j = 0; j < 3*nlocal; j++) {
      sq += s_spin[idx][j] * q[j];
    }
    MPI_Allreduce(&sq, &sq_global, 1, MPI_DOUBLE, MPI_SUM, world);
    if (update->multireplica == 1) {
      sq = sq_global;
      MPI_Allreduce(&sq, &sq_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    }

    alpha_lbfgs[idx] = rho_spin[idx] * sq_global;

    for (int j = 0; j < 3*nlocal; j++) {
      q[j] -= alpha_lbfgs[idx] * y_spin[idx][j];
    }
  }

  // Initial Hessian approximation: H0 = I (identity)
  // r = H0 * q = q
  double *r;
  memory->create(r, 3*nlocal, "min_spin_lattice:r");
  for (int i = 0; i < 3*nlocal; i++) {
    r[i] = q[i];
  }

  // Second loop: forward
  for (int i = 0; i < m; i++) {
    int idx = (lbfgs_iter - m + i) % lbfgs_mem;

    double yr = 0.0, yr_global;
    for (int j = 0; j < 3*nlocal; j++) {
      yr += y_spin[idx][j] * r[j];
    }
    MPI_Allreduce(&yr, &yr_global, 1, MPI_DOUBLE, MPI_SUM, world);
    if (update->multireplica == 1) {
      yr = yr_global;
      MPI_Allreduce(&yr, &yr_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    }

    double beta = rho_spin[idx] * yr_global;

    for (int j = 0; j < 3*nlocal; j++) {
      r[j] += s_spin[idx][j] * (alpha_lbfgs[idx] - beta);
    }
  }

  // p = -r (descent direction)
  for (int i = 0; i < 3*nlocal; i++) {
    p_spin[i] = -r[i];
  }

  memory->destroy(q);
  memory->destroy(r);
}

/* ----------------------------------------------------------------------
   Vector transport: project tangent vector to new tangent space
   Used to transport L-BFGS history vectors after spin update
------------------------------------------------------------------------- */

void MinSpinLattice::vector_transport(double *vec, int n)
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;

  for (int i = 0; i < nlocal; i++) {
    double dot = vec[3*i+0]*sp[i][0] + vec[3*i+1]*sp[i][1] + vec[3*i+2]*sp[i][2];
    vec[3*i+0] -= dot * sp[i][0];
    vec[3*i+1] -= dot * sp[i][1];
    vec[3*i+2] -= dot * sp[i][2];
  }
}

/* ----------------------------------------------------------------------
   Update L-BFGS history after taking a step
   Called at the beginning of the NEXT iteration, after new gradient is computed
   s_k = alpha * p_spin (step taken in previous iteration)
   y_k = g_new - transported(g_old)
   rho_k = 1 / <s_k, y_k>
------------------------------------------------------------------------- */

void MinSpinLattice::update_lbfgs_history(double alpha)
{
  int nlocal = atom->nlocal;
  int idx = (lbfgs_iter - 1) % lbfgs_mem;

  // s_k = alpha * p_spin (from previous iteration, still in p_spin)
  for (int i = 0; i < 3*nlocal; i++) {
    s_spin[idx][i] = alpha * p_spin[i];
  }

  // Transport s_k and old gradient to current tangent space
  vector_transport(s_spin[idx], 3*nlocal);
  vector_transport(g_spin_old, 3*nlocal);

  // y_k = g_new - transported(g_old)
  for (int i = 0; i < 3*nlocal; i++) {
    y_spin[idx][i] = g_spin[i] - g_spin_old[i];
  }

  // Transport all existing history vectors to current tangent space
  int m = MIN(lbfgs_iter - 1, lbfgs_mem);
  for (int i = 0; i < m; i++) {
    int hidx = (lbfgs_iter - 2 - i) % lbfgs_mem;
    if (hidx == idx) continue;  // skip the one we just computed
    vector_transport(s_spin[hidx], 3*nlocal);
    vector_transport(y_spin[hidx], 3*nlocal);
  }

  // Compute rho_k = 1 / <s_k, y_k>
  double sy = 0.0, sy_global;
  for (int i = 0; i < 3*nlocal; i++) {
    sy += s_spin[idx][i] * y_spin[idx][i];
  }
  MPI_Allreduce(&sy, &sy_global, 1, MPI_DOUBLE, MPI_SUM, world);
  if (update->multireplica == 1) {
    sy = sy_global;
    MPI_Allreduce(&sy, &sy_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  }

  if (fabs(sy_global) > 1.0e-14) {
    rho_spin[idx] = 1.0 / sy_global;
  } else {
    rho_spin[idx] = 0.0;
  }
}
