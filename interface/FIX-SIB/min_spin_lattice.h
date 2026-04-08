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
   Combined spin+lattice minimizer (Product Manifold Optimization):
   - Preconditioned gradient descent on M = R^{3N} x (S^2)^N
   - Scale balancing via lambda factor
   - Armijo backtracking line search
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

  // Full magnetic forces (for PairSpinML)
  double **fm_full;              // unprojected magnetic forces (-dE/dm)
  int fm_full_allocated;

  // PairSpinML detection
  class PairSpinML *pair_spin_ml;

  // Line search parameters
  double alpha_init;             // initial step size
  double alpha_min;              // minimum step size
  double c1;                     // Armijo constant
  double backtrack_factor;       // backtracking shrink factor

  // Scale balancing
  double lambda;                 // balance factor between atom and spin
  double lambda_max;             // maximum lambda to prevent huge atom steps
  double lambda_scale;           // manual scaling factor for lambda
  double eps_lambda;             // small constant for lambda calculation

  // Methods
  void calc_atom_gradient();
  void calc_spin_gradient();
  void calc_search_direction();
  double compute_lambda();
  double line_search();
  void advance_atoms(double alpha);
  void advance_spins(double alpha);
};

}    // namespace LAMMPS_NS

#endif
#endif
