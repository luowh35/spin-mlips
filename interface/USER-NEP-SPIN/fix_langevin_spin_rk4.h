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
   Langevin thermostat for NEP-SPIN RK4 method

   This fix is designed to work with fix nve/spin/rk4 which uses an RK4
   half-step for spins (dt/2) and applies the same noise to all k1-k4
   stages for the Stratonovich interpretation.

   Key features for RK4:
   1. compute_single_langevin_store_noise(): generates and stores noise
   2. compute_single_langevin_reuse_noise(): reuses stored noise

   Usage:
     fix ID group langevin/spin/rk4 temp damp seed
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(langevin/spin/rk4,FixLangevinSpinRK4);
// clang-format on
#else

#ifndef LMP_FIX_LANGEVIN_SPIN_RK4_H
#define LMP_FIX_LANGEVIN_SPIN_RK4_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLangevinSpinRK4 : public Fix {
 public:
  int tdamp_flag, temp_flag;    // damping and temperature flags

  FixLangevinSpinRK4(class LAMMPS *, int, char **);
  ~FixLangevinSpinRK4() override;
  int setmask() override;
  void init() override;
  void setup(int) override;

  // Standard Langevin computation (for compatibility)
  void compute_single_langevin(int, double *, double *);

  // RK4-specific: generate noise and store it
  void compute_single_langevin_store_noise(int, double *, double *, double *);

  // RK4-specific: reuse stored noise (for k2-k4 stages)
  void compute_single_langevin_reuse_noise(int, double *, double *, double *);

 protected:
  double alpha_t;       // transverse mag. damping
  double dts;           // magnetic timestep (dt/2 for RK4 half-step)
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
