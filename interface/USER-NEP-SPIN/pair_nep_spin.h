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
#include "nep_types.h"
#include "descriptor.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

namespace LAMMPS_NS {

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

 protected:
  // TorchScript model
  torch::jit::Module model_;
  torch::Device device_;
  bool model_loaded_;

  // Model parameters
  double cutoff_;              // Position cutoff (rc)
  double m_cut_;               // Magnetic moment cutoff
  std::string model_path_;

  // Descriptor configuration
  nep::DescriptorConfig descriptor_config_;
  std::unique_ptr<nep::MagneticACEDescriptor> descriptor_;

  // Element mapping
  std::vector<std::string> elements_;
  std::vector<int> type_mapper_;  // LAMMPS type -> model element index

  // Per-type magnetic moment magnitude (muB)
  double *sp_magnitude_;

  // Cached tensors for compute
  torch::Tensor positions_tensor_;
  torch::Tensor numbers_tensor_;
  torch::Tensor magmoms_tensor_;
  torch::Tensor cell_tensor_;

  // Cached magnetic forces for compute_single_pair
  torch::Tensor cached_mag_forces_;
  bool forces_cached_;

  // Spin state tracking for detecting when recomputation is needed
  std::vector<double> cached_spins_;  // Cached spin state [nlocal*4]
  bool spins_changed();               // Check if spins have changed since last compute
  void cache_current_spins();         // Store current spin state
  void recompute_forces();            // Recompute forces with current spin configuration

  // Internal methods
  void allocate() override;
  void load_model(const std::string &path);
  void read_config_from_model();

  // Data conversion methods
  torch::Tensor convert_positions(int ntotal);
  torch::Tensor convert_types(int ntotal);
  torch::Tensor convert_spins_to_magmoms(int ntotal);
  torch::Tensor get_cell_tensor();

  // Neighbor list conversion
  nep::NeighborList build_neighbor_list_from_lammps(int ntotal);

  // Force distribution
  void distribute_forces(const torch::Tensor &forces, int nlocal, int nghost);
  void distribute_magnetic_forces(const torch::Tensor &mag_forces, int nlocal);
};

}    // namespace LAMMPS_NS

#endif
#endif
