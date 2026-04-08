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
   Combined spin+lattice minimizer (Product Manifold Optimization):
   - Preconditioned gradient descent on M = R^{3N} x (S^2)^N
   - Scale balancing via lambda factor
   - Armijo backtracking line search
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
  fm_full(nullptr), pair_spin_ml(nullptr)
{
  nlocal_max = 0;
  fm_full_allocated = 0;

  // Line search parameters
  alpha_init = 1.0;
  alpha_min = 1.0e-10;
  c1 = 1.0e-4;              // Armijo constant
  backtrack_factor = 0.5;   // shrink factor

  // Scale balancing
  lambda = 1.0;
  lambda_max = 10.0;        // prevent huge atom steps
  lambda_scale = 1.0;       // manual scaling factor
  eps_lambda = 1.0e-10;
}

/* ---------------------------------------------------------------------- */

MinSpinLattice::~MinSpinLattice()
{
  memory->destroy(g_atom);
  memory->destroy(g_spin);
  memory->destroy(p_atom);
  memory->destroy(p_spin);
  memory->destroy(fm_full);
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
  memory->grow(p_atom, 3*nlocal_max, "min_spin_lattice:p_atom");
  memory->grow(p_spin, 3*nlocal_max, "min_spin_lattice:p_spin");

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
      utils::logmesg(lmp,
        "  min/spin/lattice: Product manifold optimization (using PairSpinML fm_full)\n");
    } else {
      utils::logmesg(lmp,
        "  min/spin/lattice: Product manifold optimization (using standard PairSpin)\n");
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

  // reallocate arrays if needed
  if (nlocal_max < nlocal) {
    nlocal_max = nlocal;
    memory->grow(g_atom, 3*nlocal_max, "min_spin_lattice:g_atom");
    memory->grow(g_spin, 3*nlocal_max, "min_spin_lattice:g_spin");
    memory->grow(p_atom, 3*nlocal_max, "min_spin_lattice:p_atom");
    memory->grow(p_spin, 3*nlocal_max, "min_spin_lattice:p_spin");
    if (pair_spin_ml) {
      memory->grow(fm_full, nlocal_max, 3, "min_spin_lattice:fm_full");
      fm_full_allocated = nlocal_max;
    }
  }

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++update->ntimestep;
    niter++;

    // Check nlocal and reallocate if needed (atom migration may occur)
    nlocal = atom->nlocal;
    if (nlocal_max < nlocal) {
      nlocal_max = nlocal;
      memory->grow(g_atom, 3*nlocal_max, "min_spin_lattice:g_atom");
      memory->grow(g_spin, 3*nlocal_max, "min_spin_lattice:g_spin");
      memory->grow(p_atom, 3*nlocal_max, "min_spin_lattice:p_atom");
      memory->grow(p_spin, 3*nlocal_max, "min_spin_lattice:p_spin");
    }

    // --- Compute energy and forces ---

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    // Check max_eval stopping condition
    if (neval >= update->max_eval) return MAXEVAL;

    // Check nlocal again after energy_force (may trigger atom migration)
    nlocal = atom->nlocal;
    if (nlocal_max < nlocal) {
      nlocal_max = nlocal;
      memory->grow(g_atom, 3*nlocal_max, "min_spin_lattice:g_atom");
      memory->grow(g_spin, 3*nlocal_max, "min_spin_lattice:g_spin");
      memory->grow(p_atom, 3*nlocal_max, "min_spin_lattice:p_atom");
      memory->grow(p_spin, 3*nlocal_max, "min_spin_lattice:p_spin");
      if (pair_spin_ml) {
        memory->grow(fm_full, nlocal_max, 3, "min_spin_lattice:fm_full");
        fm_full_allocated = nlocal_max;
      }
    }

    // --- Calculate gradients ---

    calc_atom_gradient();
    calc_spin_gradient();

    // --- Compute scale balancing factor ---

    lambda = compute_lambda();

    // --- Calculate preconditioned search direction ---

    calc_search_direction();

    // Debug output for first few iterations
    if (iter < 5 && comm->me == 0) {
      double gnorm_atom = 0.0, gnorm_spin = 0.0;
      for (int i = 0; i < 3*nlocal; i++) {
        gnorm_atom += g_atom[i] * g_atom[i];
        gnorm_spin += g_spin[i] * g_spin[i];
      }
      gnorm_atom = sqrt(gnorm_atom);
      gnorm_spin = sqrt(gnorm_spin);

      double pnorm_atom = 0.0, pnorm_spin = 0.0;
      for (int i = 0; i < 3*nlocal; i++) {
        pnorm_atom += p_atom[i] * p_atom[i];
        pnorm_spin += p_spin[i] * p_spin[i];
      }
      pnorm_atom = sqrt(pnorm_atom);
      pnorm_spin = sqrt(pnorm_spin);

      utils::logmesg(lmp, fmt::format(
        "  Iter {}: E={:.6f}, ||g_atom||={:.3e}, ||g_spin||={:.3e}, lambda={:.3e}\n"
        "         ||p_atom||={:.3e}, ||p_spin||={:.3e}, alpha={:.3e}\n"
        "         step_atom={:.3e}, step_spin={:.3e}\n",
        iter, ecurrent, gnorm_atom, gnorm_spin, lambda,
        pnorm_atom, pnorm_spin, alpha_final,
        alpha_final*pnorm_atom, alpha_final*pnorm_spin));
    }

    calc_search_direction();

    // --- Simple line search (fixed step size) ---

    double alpha = line_search();
    alpha_final = alpha;

    // NOTE: Now we need to actually take the step
    advance_atoms(alpha);
    advance_spins(alpha);

    // NOTE: Do NOT call advance_atoms/advance_spins here!
    // Line search already accepted the step at line 391-405

    // --- Recompute energy and forces ---

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    // Check max_eval again
    if (neval >= update->max_eval) return MAXEVAL;

    // --- Energy tolerance ---

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

    // --- Combined force tolerance: max(atomic_force, spin_torque) ---

    if (update->ftol > 0.0) {
      double fnorm, tnorm, combined;

      // atomic force norm (note: fnorm_max/fnorm_inf return squared values)
      if (normstyle == MAX) fnorm = sqrt(fnorm_max());
      else if (normstyle == INF) fnorm = sqrt(fnorm_inf());
      else if (normstyle == TWO) fnorm = sqrt(fnorm_sqr());
      else error->all(FLERR,"Illegal min_modify command");

      // spin torque norm (already returns sqrt values)
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

    // --- Output for thermo, dump, restart ---

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
   calculate spin gradient (Riemannian gradient on sphere)

   For PairSpinML (e.g., pair_spin_step):
     fm_full = -dE/dm (raw energy gradient in eV/μ_B, no conversions)
     Riemannian gradient = Project_tangent(fm_full)

   For standard PairSpin:
     fm = H_eff (effective field)
     Energy: E = -mag * S · H_eff
     Euclidean gradient: -dE/dS = -mag * H_eff = -mag * fm
     Riemannian gradient = Project_tangent(-mag * fm)

   Projection to tangent space: g = grad - (grad · S) * S
------------------------------------------------------------------------- */

void MinSpinLattice::calc_spin_gradient()
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double **fm = atom->fm;

  if (pair_spin_ml) {
    // Use fm_full from PairSpinML
    // fm_full = -dE/dm (already negative gradient)
    // We need gradient (not negative gradient) for our convention
    pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);

    for (int i = 0; i < nlocal; i++) {
      // Convert to gradient: dE/dm = -fm_full
      double grad_x = -fm_full[i][0];
      double grad_y = -fm_full[i][1];
      double grad_z = -fm_full[i][2];

      // Project to tangent space: g = grad - (grad · S) * S
      double dot = grad_x * sp[i][0] + grad_y * sp[i][1] + grad_z * sp[i][2];

      g_spin[3*i + 0] = grad_x - dot * sp[i][0];
      g_spin[3*i + 1] = grad_y - dot * sp[i][1];
      g_spin[3*i + 2] = grad_z - dot * sp[i][2];
    }
  } else {
    // Standard PairSpin: use -mag * fm
    for (int i = 0; i < nlocal; i++) {
      double mag = sp[i][3];

      // Euclidean gradient: -dE/dS = -mag * fm
      double grad_x = -mag * fm[i][0];
      double grad_y = -mag * fm[i][1];
      double grad_z = -mag * fm[i][2];

      // Project to tangent space: g = grad - (grad · S) * S
      double dot = grad_x * sp[i][0] + grad_y * sp[i][1] + grad_z * sp[i][2];

      g_spin[3*i + 0] = grad_x - dot * sp[i][0];
      g_spin[3*i + 1] = grad_y - dot * sp[i][1];
      g_spin[3*i + 2] = grad_z - dot * sp[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   compute scale balancing factor lambda
   lambda = ||g_spin|| / (||g_atom|| + eps)

   This makes p_atom = -lambda * g_atom have comparable magnitude to p_spin = -g_spin
   When spin gradient is small, lambda is small, reducing atom step size
------------------------------------------------------------------------- */

double MinSpinLattice::compute_lambda()
{
  int nlocal = atom->nlocal;
  double gnorm_atom = 0.0, gnorm_spin = 0.0;
  double gnorm_atom_global, gnorm_spin_global;

  // Compute local norms
  for (int i = 0; i < 3*nlocal; i++) {
    gnorm_atom += g_atom[i] * g_atom[i];
    gnorm_spin += g_spin[i] * g_spin[i];
  }

  // Global reduction
  MPI_Allreduce(&gnorm_atom, &gnorm_atom_global, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&gnorm_spin, &gnorm_spin_global, 1, MPI_DOUBLE, MPI_SUM, world);

  if (update->multireplica == 1) {
    gnorm_atom = gnorm_atom_global;
    gnorm_spin = gnorm_spin_global;
    MPI_Allreduce(&gnorm_atom, &gnorm_atom_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    MPI_Allreduce(&gnorm_spin, &gnorm_spin_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  }

  gnorm_atom = sqrt(gnorm_atom_global);
  gnorm_spin = sqrt(gnorm_spin_global);

  // Compute lambda with upper bound and manual scaling
  // lambda_scale < 1.0: reduce atom step, increase spin step
  // lambda_scale > 1.0: increase atom step, reduce spin step
  double lambda_raw = gnorm_spin / (gnorm_atom + eps_lambda);
  lambda_raw *= lambda_scale;
  return MIN(lambda_raw, lambda_max);
}

/* ----------------------------------------------------------------------
   calculate preconditioned search direction
   p_atom = -lambda * g_atom
   p_spin = -g_spin
------------------------------------------------------------------------- */

void MinSpinLattice::calc_search_direction()
{
  int nlocal = atom->nlocal;

  for (int i = 0; i < 3*nlocal; i++) {
    p_atom[i] = -lambda * g_atom[i];
    p_spin[i] = -g_spin[i];
  }
}

/* ----------------------------------------------------------------------
   Simple line search with fixed step size
   TODO: Implement proper Armijo backtracking if needed
------------------------------------------------------------------------- */

double MinSpinLattice::line_search()
{
  // For now, use fixed step size to avoid expensive backtracking
  // Each backtracking iteration requires a full energy_force() call
  // which is very expensive for ML potentials
  return alpha_init;
}

/* ----------------------------------------------------------------------
   advance atoms along search direction
------------------------------------------------------------------------- */

void MinSpinLattice::advance_atoms(double alpha)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;

  for (int i = 0; i < nlocal; i++) {
    x[i][0] += alpha * p_atom[3*i + 0];
    x[i][1] += alpha * p_atom[3*i + 1];
    x[i][2] += alpha * p_atom[3*i + 2];
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
