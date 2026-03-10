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
   Semi-Implicit B (SIB) Method with Spin-Lattice Coupling

   This fix implements the SIB predictor-corrector method for spin dynamics
   coupled with molecular dynamics. It works with any pair style that
   inherits from PairSpinML (e.g., spin/step, spin/nep).

   Algorithm:

   initial_integrate():
     1. v^{n+1/2} = v^n + (dt/2) * f^n              [velocity half-step]
     2. SIB spin update (dt/2):                      [first half spin step]
        - NN call #1: compute omega(s^n)
        - Predictor: s_pred
        - s_mid = (s^n + s_pred) / 2
        - NN call #2: compute omega(s_mid)
        - Corrector: s^{n+1/2}
     3. x^{n+1} = x^n + dt * v^{n+1/2}              [position full step]
     4. SIB spin update (dt/2):                      [second half spin step]

   [LAMMPS calls pair->compute() for new forces]

   final_integrate():
     5. v^{n+1} = v^{n+1/2} + (dt/2) * f^{n+1}      [velocity complete]

   References:
   [1] J.H. Mentink et al., J. Phys.: Condens. Matter 22, 176001 (2010)
   [2] J. Tranchida et al., J. Comput. Phys. 372, 406-425 (2018)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nve/spin/sib,FixNVESpinSIB)
// clang-format on
#else

#ifndef LMP_FIX_NVE_SPIN_SIB_H
#define LMP_FIX_NVE_SPIN_SIB_H

#include "fix.h"

namespace LAMMPS_NS {

// Forward declarations
class FixLangevinSpinSIB;
class FixGLangevinSpinSIB;
class FixPrecessionSpin;
class FixLandauSpin;
class PairSpinML;
class PairSpin;

class FixNVESpinSIB : public Fix {
 public:
  FixNVESpinSIB(class LAMMPS *, int, char **);
  ~FixNVESpinSIB() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  int lattice_flag;    // lattice_flag = 0 if spins only (frozen lattice)
                       // lattice_flag = 1 if spin-lattice coupling

 protected:
  double dtv, dtf, dts;    // velocity, force, and spin timesteps
                           // dts = dt/2 for half-step SIB updates

  int nlocal_max;    // max value of nlocal (for size of arrays)

  // Pointer to ML spin pair style (base class)
  PairSpinML *pair_spin_ml;

  // Pointers to standard PairSpin styles (e.g. spin/exchange, spin/dmi)
  int npairspin;
  PairSpin **spin_pairs;

  // Pointer to fix langevin/spin/sib (at most one allowed)
  FixLangevinSpinSIB *locklangevinspin_sib;

  // Pointer to fix glangevin/spin/sib (at most one allowed)
  FixGLangevinSpinSIB *lockglangevinspin_sib;

  // Pointers to fix precession/spin styles (multiple allowed)
  int nprecspin;
  FixPrecessionSpin **lockprecessionspin;

  // Pointers to fix landau/spin styles (multiple allowed)
  int nlandauspin;
  FixLandauSpin **locklandauspin;

  // Storage for SIB predictor-corrector method
  double **s_save;         // saved spin at start of half-step
  double *mag_save;        // saved spin magnitude |m|^n for longitudinal corrector
  double *H_par_save;      // saved H_∥ from longitudinal predictor for Heun averaging
  double **noise_vec;      // stored transverse noise vector (same for predictor and corrector)
  double *noise_L_vec;     // stored longitudinal noise (for glangevin)
  double **fm_full;        // full (unprojected) magnetic forces for longitudinal dynamics

  // Helper functions
  void sib_spin_half_step();  // perform one SIB half-step update
  void solve_implicit_sib(double *s_in, double *F_vec, double *s_out);
  void distribute_magnetic_forces();  // distribute cached mag forces to fm array
  void grow_arrays();
};

}    // namespace LAMMPS_NS

#endif
#endif
