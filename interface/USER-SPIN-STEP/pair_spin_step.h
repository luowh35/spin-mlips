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
   Contributing author: SPIN-STEP LAMMPS integration
   SPIN-STEP (Spin Tensor Equivariant Potential): ML spin potential
   Based on e3nn equivariant neural networks
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(spin/step, PairSpinSTEP)
// clang-format on
#else

#ifndef LMP_PAIR_SPIN_STEP_H
#define LMP_PAIR_SPIN_STEP_H

#include "pair_spin_ml.h"
#include <string>
#include <vector>
#include <memory>

namespace LAMMPS_NS {

// Forward declaration - implementation hidden in cpp file
struct PairSpinSTEPImpl;

class PairSpinSTEP : public PairSpinML {
 public:
  PairSpinSTEP(class LAMMPS *);
  ~PairSpinSTEP() override;

  // Required Pair interface methods
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;

  // PairSpin interface for spin dynamics
  void compute_single_pair(int, double *) override;

  // PairSpinML interface - recompute magnetic forces only
  void recompute_forces() override;

  // PairSpinML interface - distribute cached magnetic forces to fm array
  void distribute_cached_mag_forces() override;

  // PairSpinML interface - distribute full (unprojected) magnetic forces
  void distribute_full_mag_forces(double **fm_full, int nlocal) override;

 protected:
  // pimpl to hide torch dependencies
  std::unique_ptr<PairSpinSTEPImpl> impl_;

  // Model parameters (non-torch types only in header)
  double cutoff_;              // Position cutoff (r_max)
  std::string model_path_;
  bool model_loaded_;

  // Element mapping
  std::vector<std::string> elements_;
  std::vector<int> type_mapper_;  // LAMMPS type -> model element index

  // Per-type magnetic moment magnitude (muB)
  double *sp_magnitude_;

  // Force caching for compute_single_pair
  bool forces_cached_;

  // Internal methods
  void allocate() override;
  void load_model(const std::string &path);
};

}    // namespace LAMMPS_NS

#endif
#endif
