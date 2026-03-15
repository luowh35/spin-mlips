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
   On-site Landau potential for variable-length spin dynamics:
     E_L(m) = a2*m^2 + a4*m^4 + a6*m^6 + ...
   Provides longitudinal driving force -dE/d|m| for use with
   fix glangevin/spin/sib.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(landau/spin,FixLandauSpin)
// clang-format on
#else

#ifndef LMP_FIX_LANDAU_SPIN_H
#define LMP_FIX_LANDAU_SPIN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLandauSpin : public Fix {
 public:
  FixLandauSpin(class LAMMPS *, int, char **);
  ~FixLandauSpin() override;
  int setmask() override;
  void post_force(int) override;
  double compute_scalar() override;

  // Called by fix_nve_spin_sib to accumulate longitudinal force
  void compute_single_landau(int i, double mag, double *fm_full_i);

  // Return single-atom Landau energy
  double compute_single_landau_energy(double mag);

 protected:
  int nterms;        // number of expansion terms
  int *powers;       // power of each term (2, 4, 6, ...)
  double *coeffs;    // coefficient of each term
  double landau_energy;  // accumulated global energy
};

}    // namespace LAMMPS_NS

#endif
#endif
