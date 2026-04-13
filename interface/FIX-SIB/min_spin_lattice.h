/* -*- c++ -*- ----------------------------------------------------------
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
   Combined spin+lattice minimizer:
   - FIRE algorithm for lattice optimization
   - Riemannian L-BFGS for spin optimization (alternating)
   - Each outer iteration: 1 FIRE step for atoms, then N L-BFGS steps for spins
------------------------------------------------------------------------- */

#ifdef MINIMIZE_CLASS
// clang-format off
MinimizeStyle(spin/lattice,MinSpinLattice)
// clang-format on
#else

#ifndef LMP_MIN_SPIN_LATTICE_H
#define LMP_MIN_SPIN_LATTICE_H

#include "min.h"

namespace LAMMPS_NS {

class MinSpinLattice : public Min {
 public:
  MinSpinLattice(class LAMMPS *);
  ~MinSpinLattice() override;

  void init() override;
  void setup_style() override;
  int modify_param(int, char **) override;
  void reset_vectors() override;
  int iterate(int) override;

 private:
  int nlocal_max;

  // Gradients
  double *g_atom;                // atomic gradient (negative force)
  double *g_spin;                // spin gradient (Riemannian gradient, tangent vector)

  // Search directions
  double *p_atom;                // atomic search direction
  double *p_spin;                // spin search direction (tangent vector)

  // FIRE algorithm variables (for lattice optimization)
  double *v_atom;                // atomic velocities
  double dt_atom;                // time step for lattice
  double dt_max_atom;            // maximum time step
  double alpha_fire;             // FIRE mixing parameter
  double f_inc;                  // time step increase factor
  double f_dec;                  // time step decrease factor
  int n_min;                     // minimum steps before increasing dt
  int last_negative;             // steps since last P < 0

  // Full magnetic forces (for PairSpinML)
  double **fm_full;              // unprojected magnetic forces (-dE/dm)
  int fm_full_allocated;

  // PairSpinML detection
  class PairSpinML *pair_spin_ml;

  // L-BFGS parameters for spin optimization
  int lbfgs_mem;                 // number of history vectors to store
  int lbfgs_iter;                // current L-BFGS iteration
  double **s_spin;               // history: position differences (tangent vectors)
  double **y_spin;               // history: gradient differences (tangent vectors)
  double *rho_spin;              // history: 1 / <s, y>
  double *alpha_lbfgs;           // temporary array for two-loop recursion
  double *g_spin_old;            // previous spin gradient for L-BFGS
  int spin_substeps;             // number of L-BFGS spin steps per FIRE atom step

  // Line search parameters
  double alpha_init;             // initial step size
  double alpha_min;              // minimum step size
  double c1;                     // Armijo constant
  double backtrack_factor;       // backtracking shrink factor

  // Scale balancing (unused in alternating mode, kept for future)
  double lambda;
  double lambda_max;
  double lambda_scale;
  double eps_lambda;

  // Methods
  void calc_atom_gradient();
  void calc_spin_gradient();
  void calc_spin_lbfgs_direction();
  void advance_atoms_fire();     // FIRE update for atoms
  void advance_spins(double alpha);
  void fire_reset();             // Reset FIRE parameters
  void vector_transport(double *vec, int n);
  void update_lbfgs_history(double alpha);
  void realloc_arrays(int nlocal);
  int spin_lbfgs_substep();      // single L-BFGS spin substep, returns neval
};

}    // namespace LAMMPS_NS

#endif
#endif
