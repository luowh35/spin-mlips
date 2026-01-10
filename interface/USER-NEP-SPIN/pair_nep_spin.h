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
   Contributing author: NEP-SPIN LAMMPS integration
   NEP-SPIN: Machine learning potential for magnetic systems
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(spin/nep, PairNEPSpin)
// clang-format on
#else

#ifndef LMP_PAIR_NEP_SPIN_H
#define LMP_PAIR_NEP_SPIN_H

#include "pair_spin.h"
#include <string>
#include <vector>
#include <memory>

namespace LAMMPS_NS {

// Forward declaration - implementation hidden in cpp file
struct PairNEPSpinImpl;

class PairNEPSpin : public PairSpin {
 public:
  PairNEPSpin(class LAMMPS *);
  ~PairNEPSpin() override;

  // Required Pair interface methods
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;

  // PairSpin interface for spin dynamics
  void compute_single_pair(int, double *) override;

  // For SIB method: get cached magnetic forces and distribute to fm array
  void distribute_cached_mag_forces();

  // Recompute magnetic forces only (without atomic forces)
  void recompute_forces();

 protected:
  // pimpl to hide torch dependencies
  std::unique_ptr<PairNEPSpinImpl> impl_;

  // Model parameters (non-torch types only in header)
  double cutoff_;              // Position cutoff (rc)
  double m_cut_;               // Magnetic moment cutoff
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
