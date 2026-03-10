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
   Generalized Langevin thermostat for variable-length spin dynamics

   This fix implements a generalized Langevin thermostat that allows
   spin magnitude to fluctuate (variable-length spin mode).

   Based on SPILADY (Spin-Lattice Dynamics) implementation:
   - Ma & Dudarev, PRB 83, 134418 (2011)

   For fixed-length spin dynamics, use fix langevin/spin/sib instead.

   Usage:
     fix ID group glangevin/spin/sib temp alpha_t tau_L seed

   Parameters:
     temp    : spin bath temperature (K)
     alpha_t : transverse (Gilbert) damping coefficient (dimensionless, typical: 0.01)
     tau_L   : longitudinal relaxation time (ps, typical: 0.01-0.1)
               Controls spin magnitude relaxation rate
               Internally converted to gamma_L = 1/tau_L
     seed    : random number seed

   References:
   [1] P.-W. Ma, S.L. Dudarev, PRB 83, 134418 (2011) - SPILADY
   [2] J.H. Mentink et al., J. Phys.: Condens. Matter 22, 176001 (2010) - SIB
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(glangevin/spin/sib,FixGLangevinSpinSIB);
// clang-format on
#else

#ifndef LMP_FIX_GLANGEVIN_SPIN_SIB_H
#define LMP_FIX_GLANGEVIN_SPIN_SIB_H

#include "fix.h"

namespace LAMMPS_NS {

class FixGLangevinSpinSIB : public Fix {
 public:
  int tdamp_flag, temp_flag;           // damping and temperature flags

  FixGLangevinSpinSIB(class LAMMPS *, int, char **);
  ~FixGLangevinSpinSIB() override;
  int setmask() override;
  void init() override;
  void setup(int) override;

  // SIB-specific: generate noise and store it (transverse only, like langevin/spin/sib)
  void compute_single_langevin_store_noise(int, double *, double *, double *);

  // SIB-specific: reuse stored noise (for corrector step)
  void compute_single_langevin_reuse_noise(int, double *, double *, double *);

  // Longitudinal predictor: Euler step in log-space, returns H_parallel for averaging
  void compute_longitudinal_predictor(int i, double *spi, double *fmi,
                                       double &H_par_out, double dt_step);

  // Longitudinal corrector: Heun/trapezoidal step from mag_save, adds noise
  void compute_longitudinal_corrector(int i, double *spi, double *fmi,
                                       double H_par_pred, double mag_save,
                                       double &noise_L_out, double dt_step);

 protected:
  // Transverse (Gilbert) damping parameters
  double alpha_t;       // transverse mag. damping coefficient
  double dts;           // magnetic timestep (dt/2 for SIB method)
  double temp;          // spin bath temperature
  double D_T, sigma_T;  // transverse bath intensity variables
  double gil_factor;    // Gilbert's prefactor = 1/(1+alpha_t^2)

  // Longitudinal damping parameters
  double tau_L;         // longitudinal relaxation time (ps)
  double gamma_L;       // longitudinal damping coefficient = 1/tau_L (ps^-1)
  double sigma_L;       // longitudinal noise strength per half-step

  class RanMars *random;
  int seed;

  // Helper functions for transverse components
  void add_tdamping(double *, double *);
  void add_noise(double *, double *);
  void apply_gil_factor(double *);
};

}    // namespace LAMMPS_NS

#endif
#endif
