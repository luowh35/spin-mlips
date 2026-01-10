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
   NEP-SPIN Data Management Class
   Optimized neighbor list and data structure for NEP-SPIN potential
   Based on MLIAP design pattern
------------------------------------------------------------------------- */

#ifndef LMP_NEP_SPIN_DATA_H
#define LMP_NEP_SPIN_DATA_H

#include "pointers.h"
#include <torch/torch.h>
#include <vector>
#include <string>

namespace LAMMPS_NS {

class NEPSpinData : protected Pointers {
 public:
  NEPSpinData(class LAMMPS *, double cutoff, const std::vector<std::string> &elements);
  ~NEPSpinData() override;

  // Generate neighbor data from LAMMPS neighbor list
  void generate_neighdata(class NeighList *list, int eflag, int vflag);

  // Convert data to PyTorch tensors (with optional GPU transfer)
  void to_tensors(torch::Device device);

  // Accessors for tensor data
  torch::Tensor get_positions() const { return positions_tensor_; }
  torch::Tensor get_numbers() const { return numbers_tensor_; }
  torch::Tensor get_magmoms() const { return magmoms_tensor_; }
  torch::Tensor get_cell() const { return cell_tensor_; }
  torch::Tensor get_center_indices() const { return center_indices_; }
  torch::Tensor get_neighbor_indices() const { return neighbor_indices_; }
  torch::Tensor get_shifts() const { return shifts_tensor_; }
  torch::Tensor get_batch_idx() const { return batch_idx_; }

  // Public data for external access
  int nlocal;              // number of local atoms
  int ntotal;              // total atoms (local + ghost)
  int npairs;              // number of neighbor pairs within cutoff
  int eflag;               // energy flag
  int vflag;               // virial flag

  // Cached forces for compute_single_pair (always on CPU for fast access)
  torch::Tensor cached_mag_forces;
  float *cached_mag_forces_ptr;  // Direct pointer for fast access
  int cached_mag_forces_size;
  bool forces_cached;

 private:
  // Configuration
  double cutoff_;
  double cutoff_sq_;
  std::vector<std::string> elements_;
  std::vector<int> type_map_;  // LAMMPS type -> element index

  // Maximum allocated sizes (for incremental growth)
  int nmax_;               // max atoms (for positions, types, etc.)
  int npairs_max_;         // max neighbor pairs

  // Raw data arrays (LAMMPS memory managed)
  // Atom data [ntotal]
  double *positions_;      // [ntotal * 3] flattened positions
  double *magmoms_;        // [ntotal * 3] flattened magnetic moments
  int *atom_types_;        // [ntotal] atomic numbers

  // Pair data [npairs]
  int *center_idx_;        // [npairs] center atom index
  int *neighbor_idx_;      // [npairs] neighbor atom index
  double *rij_;            // [npairs * 3] distance vectors

  // Atom neighbor counts [nlocal]
  int *numneighs_;         // [nlocal] neighbor count per atom

  // Cell data
  double cell_[9];         // 3x3 cell matrix flattened

  // PyTorch tensors (created in to_tensors)
  torch::Tensor positions_tensor_;
  torch::Tensor numbers_tensor_;
  torch::Tensor magmoms_tensor_;
  torch::Tensor cell_tensor_;
  torch::Tensor center_indices_;
  torch::Tensor neighbor_indices_;
  torch::Tensor shifts_tensor_;
  torch::Tensor batch_idx_;

  // Tensor valid flag
  bool tensors_valid_;

  // Helper methods
  void grow_arrays(int ntotal_new, int npairs_new);
  int element_to_number(const std::string &elem);
};

}  // namespace LAMMPS_NS

#endif
