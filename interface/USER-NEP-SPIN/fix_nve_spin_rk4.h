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
   NEP-SPIN 4th-order Runge-Kutta (RK4) Method with Spin-Lattice Coupling

   This fix implements the classical RK4 method for spin dynamics coupled
   with molecular dynamics, following the framework of Tranchida et al.
   (2018) for spin-lattice coupling.

   Algorithm:

   initial_integrate():
     1. v^{n+1/2} = v^n + (dt/2) * f^n              [velocity half-step]
     2. RK4 spin update (dt/2):                      [first half spin step]
        - k1 = -s × omega(s)
        - k2 = -s_1 × omega(s_1), where s_1 = s + 0.5*k1
        - k3 = -s_2 × omega(s_2), where s_2 = s + 0.5*k2
        - k4 = -s_3 × omega(s_3), where s_3 = s + k3
        - s^{n+1/2} = s + (k1 + 2*k2 + 2*k3 + k4)/6
     3. x^{n+1} = x^n + dt * v^{n+1/2}              [position full step]
     4. RK4 spin update (dt/2):                      [second half spin step]
        - Same as step 2

   [LAMMPS calls pair->compute() for new forces]

   final_integrate():
     5. v^{n+1} = v^{n+1/2} + (dt/2) * f^{n+1}      [velocity complete]

   Properties:
   - Fourth-order accuracy O(dt^4) for spin dynamics
   - 8 NN calls per timestep (4 per half-step)
   - Requires explicit renormalization
   - Proper spin-lattice coupling following Tranchida et al. (2018)

   For stochastic LLG (finite temperature):
   - Noise is generated once at the start of each half-step
   - Same noise is used for all k1-k4 stages (Stratonovich interpretation)
   - Following Spirit's approach in Method_LLG.cpp

   References:
   [1] J. Tranchida et al., J. Comput. Phys. 372, 406-425 (2018)
       DOI: 10.1016/j.jcp.2018.06.042
   [2] Spirit code: https://github.com/spirit-code/spirit
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nve/spin/rk4,FixNVESpinRK4);
// clang-format on
#else

#ifndef LMP_FIX_NVE_SPIN_RK4_H
#define LMP_FIX_NVE_SPIN_RK4_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNVESpinRK4 : public Fix {
 public:
  FixNVESpinRK4(class LAMMPS *, int, char **);
  ~FixNVESpinRK4() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  int lattice_flag;    // lattice_flag = 0 if spins only (frozen lattice)
                       // lattice_flag = 1 if spin-lattice coupling

 protected:
  double dtv, dtf, dts;    // velocity, force, and spin timesteps
                           // dts = dt/2 for half-step RK4 updates

  int nlocal_max;    // max value of nlocal (for size of arrays)

  // Pointer to NEP-SPIN pair style
  class PairNEPSpin *pair_nep_spin;

  // Pointers to fix langevin/spin/rk4 styles
  int nlangspin_rk4;
  int maglangevin_rk4_flag;
  class FixLangevinSpinRK4 **locklangevinspin_rk4;

  // Pointers to fix precession/spin styles
  int nprecspin;
  int precession_spin_flag;
  class FixPrecessionSpin **lockprecessionspin;

  // Storage for RK4 method
  double **s_save;      // saved spin at start of half-step [nlocal][3]
  double **k1;          // RK4 stage 1 [nlocal][3]
  double **k2;          // RK4 stage 2 [nlocal][3]
  double **k3;          // RK4 stage 3 [nlocal][3]
  double **k4;          // RK4 stage 4 [nlocal][3]
  double **xi;          // thermal noise vector [nlocal][3]
  bool reuse_noise;     // reuse stored noise for k2-k4 stages

  // Helper functions
  void rk4_spin_half_step();  // perform one RK4 half-step update
  void compute_omega(int i, double *spi, double *fmi);  // compute effective field
  void distribute_magnetic_forces();  // distribute cached mag forces to fm array
  void prepare_thermal_field();  // generate thermal noise (Spirit-style)
  void grow_arrays();
};

}    // namespace LAMMPS_NS

#endif
#endif
