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
   PairSpinML: Abstract base class for Machine Learning Spin Potentials

   This class provides a common interface for ML-based spin potentials
   (e.g., STEP, NEP-SPIN) to work with unified spin dynamics integrators
   (fix nve/spin/sib, fix langevin/spin/sib).

   Required interface methods:
   - recompute_forces(): Recompute magnetic forces only (not atomic forces)
   - distribute_cached_mag_forces(): Distribute cached magnetic forces to fm array
------------------------------------------------------------------------- */

#ifndef LMP_PAIR_SPIN_ML_H
#define LMP_PAIR_SPIN_ML_H

#include "pair_spin.h"

namespace LAMMPS_NS {

class PairSpinML : public PairSpin {
 public:
  PairSpinML(class LAMMPS *lmp) : PairSpin(lmp) {}
  ~PairSpinML() override = default;

  // =========================================================================
  // Interface for ML spin potentials - must be implemented by derived classes
  // =========================================================================

  // Recompute magnetic forces only (without recomputing atomic forces)
  // This is called during spin dynamics integration when only spin
  // orientations change but atomic positions remain fixed.
  virtual void recompute_forces() = 0;

  // Distribute cached magnetic forces to the fm array
  // This is called by fix nve/spin/sib after recompute_forces()
  // to update fm without modifying f (atomic forces).
  // These are the PROJECTED (perpendicular to magmom) forces for transverse dynamics.
  virtual void distribute_cached_mag_forces() = 0;

  // Distribute the full (unprojected) magnetic forces to the fm_full array.
  // This includes the longitudinal component dE/d|m| needed for variable-length
  // spin dynamics (glangevin). Called by fix nve/spin/sib for the longitudinal step.
  // Default implementation zeros the array; override in derived classes that
  // cache the unprojected gradient.
  virtual bool has_longitudinal_force() const { return false; }
  virtual void distribute_full_mag_forces(double **fm_full, int nlocal) {
    for (int i = 0; i < nlocal; i++) {
      fm_full[i][0] = 0.0;
      fm_full[i][1] = 0.0;
      fm_full[i][2] = 0.0;
    }
  }
};

}    // namespace LAMMPS_NS

#endif
