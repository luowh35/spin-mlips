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

#ifndef LMP_FIX_NH_SPIN_SIB_H
#define LMP_FIX_NH_SPIN_SIB_H

#include "fix_nh.h"

namespace LAMMPS_NS {

class FixGLangevinSpinSIB;
class FixLandauSpin;
class FixLangevinSpinSIB;
class FixPrecessionSpin;
class PairSpin;
class PairSpinML;

class FixNHSpinSIB : public FixNH {
 public:
  FixNHSpinSIB(class LAMMPS *, int, char **);
  ~FixNHSpinSIB() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  int lattice_flag;    // 0 = frozen lattice, 1 = moving lattice

 protected:
  double dts;
  int nlocal_max;
  int npairspin;
  int nprecspin;
  int nlandauspin;

  PairSpinML *pair_spin_ml;
  PairSpin **spin_pairs;
  FixPrecessionSpin **lockprecessionspin;
  FixLangevinSpinSIB *locklangevinspin_sib;
  FixGLangevinSpinSIB *lockglangevinspin_sib;
  FixLandauSpin **locklandauspin;

  double **s_save;
  double *mag_save;
  double *H_par_save;
  double **noise_vec;
  double *noise_L_vec;
  double **fm_full;

  void grow_arrays();
  void sib_spin_half_step();
  void solve_implicit_sib(double *, double *, double *);
  void distribute_magnetic_forces();
};

}    // namespace LAMMPS_NS

#endif
