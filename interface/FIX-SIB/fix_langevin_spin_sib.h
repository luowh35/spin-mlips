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
   Langevin thermostat for SPIN-STEP SIB method

   This fix is designed to work with fix nve/spin/sib which uses the
   SIB predictor-corrector method.

   Key features for SIB:
   1. compute_single_langevin_store_noise(): generates and stores noise
   2. compute_single_langevin_reuse_noise(): reuses stored noise

   The same noise must be used in both predictor and corrector steps
   for correct Stratonovich interpretation.

   Usage:
     fix ID group langevin/spin/sib temp damp seed

   Reference:
   J.H. Mentink, M.V. Tretyakov, A. Fasolino, M.I. Katsnelson, T. Rasing,
   "Stable and fast semi-implicit integration of the stochastic
   Landau-Lifshitz equation", J. Phys.: Condens. Matter 22, 176001 (2010)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(langevin/spin/sib,FixLangevinSpinSIB)
// clang-format on
#else

#ifndef LMP_FIX_LANGEVIN_SPIN_SIB_H
#define LMP_FIX_LANGEVIN_SPIN_SIB_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLangevinSpinSIB : public Fix {
 public:
  int tdamp_flag, temp_flag;    // damping and temperature flags

  FixLangevinSpinSIB(class LAMMPS *, int, char **);
  ~FixLangevinSpinSIB() override;
  int setmask() override;
  void init() override;
  void setup(int) override;

  // Standard Langevin computation (for compatibility)
  void compute_single_langevin(int, double *, double *);

  // SIB-specific: generate noise and store it
  void compute_single_langevin_store_noise(int, double *, double *, double *);

  // SIB-specific: reuse stored noise (for corrector step)
  void compute_single_langevin_reuse_noise(int, double *, double *, double *);

 protected:
  double alpha_t;       // transverse mag. damping
  double dts;           // magnetic timestep (dt/2 for SIB method)
  double temp;          // spin bath temperature
  double D, sigma;      // bath intensity var.
  double gil_factor;    // gilbert's prefactor

  class RanMars *random;
  int seed;

  // Helper functions
  void add_tdamping(double *, double *);
  void add_noise(double *, double *);
  void apply_gil_factor(double *);
};

}    // namespace LAMMPS_NS

#endif
#endif
