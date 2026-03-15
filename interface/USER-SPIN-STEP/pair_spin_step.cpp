/* ----------------------------------------------------------------------
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
   SPIN-STEP: E3nn Magnetic Atomic SPIN potential
   Based on MagNequIP model with e3nn equivariant neural networks
------------------------------------------------------------------------- */

#include "pair_spin_step.h"
#include "step_utils.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "fix_nve_spin.h"

// Torch headers - only in cpp file
#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

// =============================================================================
// Implementation struct that hides torch dependencies
// =============================================================================

struct LAMMPS_NS::PairSpinSTEPImpl {
  // TorchScript model
  torch::jit::Module model;
  torch::Device device;

  // Model configuration (loaded from embedded config.json)
  double r_max;
  int num_types;
  int num_features;
  int lmax;
  int num_layers;
  double avg_num_neighbors;
  std::vector<int> magnetic_atom_types;
  std::unordered_map<int, int> atom_types_map;  // atomic number -> type index
  double scale;
  double shift;

  // Cached magnetic forces for compute_single_pair
  torch::Tensor cached_mag_forces;

  // Cached full (unprojected) magnetic forces for longitudinal dynamics
  torch::Tensor cached_full_mag_forces;

  // Cache for last valid gradients (used when NaN occurs)
  torch::Tensor last_valid_mag_grads;
  bool has_valid_grads = false;

  // Cache for last valid projected forces
  torch::Tensor last_valid_projected_forces;
  bool has_valid_projected_forces = false;

  PairSpinSTEPImpl() : device(torch::kCPU) {
    // Default configuration
    r_max = 5.0;
    num_types = 1;
    num_features = 64;
    lmax = 2;
    num_layers = 3;
    avg_num_neighbors = 25.0;
    scale = 1.0;
    shift = 0.0;
  }

  // Data conversion methods
  torch::Tensor convert_positions(double **x, int ntotal) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto positions = torch::zeros({ntotal, 3}, options);
    auto pos_accessor = positions.accessor<float, 2>();

    for (int i = 0; i < ntotal; i++) {
      pos_accessor[i][0] = static_cast<float>(x[i][0]);
      pos_accessor[i][1] = static_cast<float>(x[i][1]);
      pos_accessor[i][2] = static_cast<float>(x[i][2]);
    }

    return positions.to(device);
  }

  torch::Tensor convert_types(int *type, int ntotal, const std::vector<std::string>& elements) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto numbers = torch::zeros({ntotal}, options);
    auto num_accessor = numbers.accessor<int64_t, 1>();

    for (int i = 0; i < ntotal; i++) {
      int lmp_type = type[i];
      std::string elem = elements[lmp_type - 1];
      int z = step::element_to_number(elem);

      // Map atomic number to model type index
      auto it = atom_types_map.find(z);
      if (it != atom_types_map.end()) {
        num_accessor[i] = it->second;
      } else {
        num_accessor[i] = 0;  // Default to first type
      }
    }

    return numbers.to(device);
  }

  torch::Tensor convert_spins_to_magmoms(double **sp, int ntotal) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto magmoms = torch::zeros({ntotal, 3}, options);
    auto mag_accessor = magmoms.accessor<float, 2>();

    for (int i = 0; i < ntotal; i++) {
      double mag = sp[i][3];
      mag_accessor[i][0] = static_cast<float>(sp[i][0] * mag);
      mag_accessor[i][1] = static_cast<float>(sp[i][1] * mag);
      mag_accessor[i][2] = static_cast<float>(sp[i][2] * mag);
    }

    return magmoms.to(device);
  }

  step::NeighborListData build_neighbor_list(NeighList *list, int nlocal, int ntotal,
                                            double **x, double rc_sq, tagint *tag, Atom *atom) {
    step::NeighborListData result;
    result.n_pairs = 0;

    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    // Store pairs with shift vectors
    // For ghost atoms, we map back to local atom index and compute shift
    struct EdgeData {
      int i_local;      // local atom index (dst)
      int j_local;      // local atom index (src) - mapped from ghost if needed
      double shift[3];  // shift vector = x[ghost] - x[local]
    };
    std::vector<EdgeData> valid_edges;
    valid_edges.reserve(inum * 20);

    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      int *jlist = firstneigh[i];
      int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj] & NEIGHMASK;

        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];
        double rsq = dx*dx + dy*dy + dz*dz;

        if (rsq < rc_sq) {
          EdgeData edge;
          edge.i_local = i;  // i is always local

          if (j < nlocal) {
            // j is a local atom
            edge.j_local = j;
            edge.shift[0] = 0.0;
            edge.shift[1] = 0.0;
            edge.shift[2] = 0.0;
          } else {
            // j is a ghost atom, map back to local atom using tag
            int j_local = atom->map(tag[j]);
            // atom->map returns -1 if not found, or could return a ghost index
            // (>= nlocal) if the atom is not owned by this rank.
            // Only local indices [0, nlocal) are valid for the node tensor.
            if (j_local < 0 || j_local >= nlocal) {
              continue;
            }
            edge.j_local = j_local;
            // shift = x[ghost] - x[local] (the periodic displacement)
            edge.shift[0] = x[j][0] - x[j_local][0];
            edge.shift[1] = x[j][1] - x[j_local][1];
            edge.shift[2] = x[j][2] - x[j_local][2];
          }
          valid_edges.push_back(edge);
        }
      }
    }

    int total_pairs = valid_edges.size();

    if (total_pairs == 0) {
      result.edge_index = torch::zeros({2, 0}, torch::kInt64).to(device);
      result.shifts = torch::zeros({0, 3}, torch::kFloat32).to(device);
      return result;
    }

    auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    auto edge_index = torch::zeros({2, total_pairs}, options_int);
    result.shifts = torch::zeros({total_pairs, 3}, options_float);

    auto edge_accessor = edge_index.accessor<int64_t, 2>();
    auto shift_accessor = result.shifts.accessor<float, 2>();

    for (int pair_idx = 0; pair_idx < total_pairs; pair_idx++) {
      // edge_index[0] = dst (center), edge_index[1] = src (neighbor)
      // Both are now local atom indices (0 to nlocal-1)
      edge_accessor[0][pair_idx] = valid_edges[pair_idx].i_local;
      edge_accessor[1][pair_idx] = valid_edges[pair_idx].j_local;
      shift_accessor[pair_idx][0] = static_cast<float>(valid_edges[pair_idx].shift[0]);
      shift_accessor[pair_idx][1] = static_cast<float>(valid_edges[pair_idx].shift[1]);
      shift_accessor[pair_idx][2] = static_cast<float>(valid_edges[pair_idx].shift[2]);
    }

    result.edge_index = edge_index.to(device);
    result.shifts = result.shifts.to(device);
    result.n_pairs = total_pairs;
    return result;
  }
};

// =============================================================================
// PairSpinSTEP Constructor / Destructor
// =============================================================================

PairSpinSTEP::PairSpinSTEP(LAMMPS *lmp) : PairSpinML(lmp)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  // Create implementation object
  impl_ = std::make_unique<PairSpinSTEPImpl>();

  // Auto-detect GPU and use if available
  if (torch::cuda::is_available()) {
    int num_gpus = torch::cuda::device_count();
    int gpu_id = comm->me % num_gpus;
    impl_->device = torch::Device(torch::kCUDA, gpu_id);
    if (comm->me == 0) {
      utils::logmesg(lmp, "SPIN-STEP: {} CUDA GPU(s) detected, using GPU acceleration\n", num_gpus);
    }
  } else {
    impl_->device = torch::kCPU;
    if (comm->me == 0) {
      utils::logmesg(lmp, "SPIN-STEP: No CUDA GPU detected, using CPU\n");
    }
  }

  model_loaded_ = false;
  forces_cached_ = false;
  sp_magnitude_ = nullptr;
  cutoff_ = 0.0;
}

PairSpinSTEP::~PairSpinSTEP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  if (sp_magnitude_) {
    memory->destroy(sp_magnitude_);
  }
}

// =============================================================================
// Allocate Memory
// =============================================================================

void PairSpinSTEP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(sp_magnitude_, n + 1, "pair:sp_magnitude");

  for (int i = 1; i <= n; i++) {
    sp_magnitude_[i] = 1.0;
    for (int j = i; j <= n; j++) {
      setflag[i][j] = 0;
    }
  }
}

// =============================================================================
// Settings
// =============================================================================

void PairSpinSTEP::settings(int narg, char ** /*arg*/)
{
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command: spin/step takes no arguments");
}

// =============================================================================
// Coefficients
// =============================================================================

void PairSpinSTEP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg < 4)
    error->all(FLERR, "Incorrect args for pair coefficients: "
               "pair_coeff * * <model.pt> <elem1> [elem2 ...]");

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for spin/step must use * * wildcard");

  model_path_ = arg[2];

  elements_.clear();
  for (int iarg = 3; iarg < narg; iarg++) {
    elements_.push_back(arg[iarg]);
  }

  if (elements_.empty())
    error->all(FLERR, "No elements specified in pair_coeff");

  int ntypes = atom->ntypes;
  if ((int)elements_.size() != ntypes)
    error->all(FLERR, "Number of elements ({}) does not match atom types ({})",
               elements_.size(), ntypes);

  type_mapper_.resize(ntypes);
  for (int i = 0; i < ntypes; i++) {
    type_mapper_[i] = i;
    sp_magnitude_[i + 1] = 1.0;
  }

  load_model(model_path_);

  cutoff_ = impl_->r_max;

  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoff_ * cutoff_;
    }
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "SPIN-STEP: Model loaded from {}\n", model_path_);
    utils::logmesg(lmp, "SPIN-STEP: Cutoff = {} Angstrom\n", cutoff_);
    utils::logmesg(lmp, "SPIN-STEP: Elements:");
    for (const auto& elem : elements_) {
      utils::logmesg(lmp, " {}", elem);
    }
    utils::logmesg(lmp, "\n");
  }
}

// =============================================================================
// Load Model
// =============================================================================

void PairSpinSTEP::load_model(const std::string &path)
{
  try {
    std::unordered_map<std::string, std::string> extra_files;
    extra_files["config.json"] = "";

    impl_->model = torch::jit::load(path, impl_->device, extra_files);
    impl_->model.eval();

    std::string config_json = extra_files["config.json"];
    if (!config_json.empty()) {
      if (comm->me == 0) {
        utils::logmesg(lmp, "SPIN-STEP: Found embedded config in model file\n");
      }

      // Parse configuration
      impl_->r_max = step::extract_float(config_json, "r_max", impl_->r_max);
      impl_->num_features = step::extract_int(config_json, "num_features", impl_->num_features);
      impl_->lmax = step::extract_int(config_json, "lmax", impl_->lmax);
      impl_->num_layers = step::extract_int(config_json, "num_layers", impl_->num_layers);
      impl_->avg_num_neighbors = step::extract_float(config_json, "avg_num_neighbors", impl_->avg_num_neighbors);
      impl_->scale = step::extract_float(config_json, "scale", impl_->scale);
      impl_->shift = step::extract_float(config_json, "shift", impl_->shift);

      // Parse magnetic atom types
      impl_->magnetic_atom_types = step::extract_int_array(config_json, "magnetic_atom_types");

      // Parse atom_types_map
      impl_->atom_types_map = step::extract_atom_types_map(config_json);

      // If atom_types_map is empty, create default mapping from elements
      if (impl_->atom_types_map.empty()) {
        for (size_t i = 0; i < elements_.size(); i++) {
          int z = step::element_to_number(elements_[i]);
          impl_->atom_types_map[z] = static_cast<int>(i);
        }
      }

      impl_->num_types = impl_->atom_types_map.size();

      if (comm->me == 0) {
        utils::logmesg(lmp, "SPIN-STEP: Loaded config: r_max={}, num_features={}, lmax={}, num_layers={}\n",
                      impl_->r_max, impl_->num_features, impl_->lmax, impl_->num_layers);
        utils::logmesg(lmp, "SPIN-STEP: avg_num_neighbors={}, num_types={}\n",
                      impl_->avg_num_neighbors, impl_->num_types);
      }
    } else {
      if (comm->me == 0) {
        utils::logmesg(lmp, "SPIN-STEP: No embedded config, using default values\n");
      }
      // Create default mapping from elements
      for (size_t i = 0; i < elements_.size(); i++) {
        int z = step::element_to_number(elements_[i]);
        impl_->atom_types_map[z] = static_cast<int>(i);
      }
      impl_->num_types = elements_.size();
    }

    // Try to freeze model for faster inference
    if (impl_->model.hasattr("training")) {
      impl_->model = torch::jit::freeze(impl_->model);
    }

    model_loaded_ = true;

    if (comm->me == 0) {
      utils::logmesg(lmp, "SPIN-STEP: TorchScript model loaded successfully\n");
    }
  } catch (const c10::Error &e) {
    error->all(FLERR, "Failed to load TorchScript model: {}", e.what());
  }
}

// =============================================================================
// Init Style
// =============================================================================

void PairSpinSTEP::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style spin/step requires atom IDs");

  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "Pair style spin/step requires metal units");

  if (!atom->sp_flag)
    error->all(FLERR, "Pair style spin/step requires atom_style spin");

  // Require atom map for ghost atom mapping
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Pair style spin/step requires atom_modify map array or hash");

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style spin/step requires newton pair on");

  // Check for compatible spin integration fix
  auto nve_spin_fixes = modify->get_fix_by_style("^nve/spin$");
  if ((comm->me == 0) && (nve_spin_fixes.size() == 0))
    error->warning(FLERR, "Using spin pair style without nve/spin");

  if (nve_spin_fixes.size() == 1) {
    lattice_flag = (dynamic_cast<FixNVESpin *>(nve_spin_fixes.front()))->lattice_flag;
  }

  nlocal_max = atom->nlocal;
}

// =============================================================================
// Init One
// =============================================================================

double PairSpinSTEP::init_one(int /*i*/, int /*j*/)
{
  return cutoff_;
}

// =============================================================================
// Extract
// =============================================================================

void *PairSpinSTEP::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "cut") == 0) return (void *) &cutoff_;
  if (strcmp(str, "sp_magnitude") == 0) {
    dim = 1;
    return (void *) sp_magnitude_;
  }
  return nullptr;
}

// =============================================================================
// Compute - Main force calculation
// =============================================================================

void PairSpinSTEP::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (!model_loaded_) {
    error->all(FLERR, "SPIN-STEP model not loaded");
  }

  double **x = atom->x;
  double **f = atom->f;
  double **sp = atom->sp;
  double **fm = atom->fm;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;

  forces_cached_ = false;

  try {
    // 1. Convert LAMMPS data to PyTorch tensors (only local atoms needed now)
    auto positions = impl_->convert_positions(x, nlocal);
    auto numbers = impl_->convert_types(type, nlocal, elements_);
    auto magmoms = impl_->convert_spins_to_magmoms(sp, nlocal);

    // 2. Enable gradient tracking
    positions.set_requires_grad(true);
    magmoms.set_requires_grad(true);

    // 3. Build neighbor list with proper shifts for periodic boundaries
    double rc_sq = cutoff_ * cutoff_;
    auto neighbors = impl_->build_neighbor_list(list, nlocal, ntotal, x, rc_sq, tag, atom);

    // Debug output for neighbor list
    if (comm->me == 0 && update->ntimestep == 0) {
      utils::logmesg(lmp, "SPIN-STEP DEBUG: nlocal={}, nghost={}, ntotal={}\n", nlocal, nghost, ntotal);
      utils::logmesg(lmp, "SPIN-STEP DEBUG: cutoff={}, rc_sq={}\n", cutoff_, rc_sq);
      utils::logmesg(lmp, "SPIN-STEP DEBUG: n_pairs={}\n", neighbors.n_pairs);
    }

    if (neighbors.n_pairs == 0) {
      eng_vdwl = 0.0;
      forces_cached_ = true;
      impl_->cached_mag_forces = torch::zeros({nlocal, 3}, torch::kFloat32);
      impl_->cached_full_mag_forces = torch::zeros({nlocal, 3}, torch::kFloat32);
      return;
    }

    // 4. Build input dictionary for model
    c10::Dict<std::string, torch::Tensor> data_dict;
    data_dict.insert("pos", positions);
    data_dict.insert("numbers", numbers);
    data_dict.insert("magmoms", magmoms);
    data_dict.insert("edge_index", neighbors.edge_index);
    data_dict.insert("shifts", neighbors.shifts);

    // 5. Model forward pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(data_dict);
    auto atomic_energies = impl_->model.forward(inputs).toTensor();

    // Check for NaN in energies
    if (step::has_nan(atomic_energies)) {
      error->warning(FLERR, "SPIN-STEP: NaN detected in atomic energies");
    }

    // atomic_energies now has shape [nlocal, 1] since we only pass local atoms
    auto total_energy = atomic_energies.sum();

    // 6. Compute gradients via autograd
    auto grads = torch::autograd::grad(
        {total_energy},
        {positions, magmoms},
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    auto pos_grads = grads[0];
    auto mag_grads = grads[1];

    // 7. Handle NaN in gradients
    if (step::has_nan(pos_grads)) {
      error->warning(FLERR, "SPIN-STEP: NaN detected in position gradients");
      pos_grads = step::replace_nan(pos_grads);
    }

    bool has_nan_mag = step::has_nan(mag_grads);
    if (has_nan_mag) {
      mag_grads = step::replace_nan(mag_grads, impl_->last_valid_mag_grads);
    }

    // Cache valid gradients
    if (!has_nan_mag) {
      impl_->last_valid_mag_grads = mag_grads.detach().clone();
      impl_->has_valid_grads = true;
    }

    // 8. Compute forces (negative gradient of energy)
    auto forces_tensor = -pos_grads;

    // 9. Magnetic forces: project to perpendicular direction
    auto full_mag_forces = -mag_grads;  // full (unprojected) for longitudinal dynamics
    auto mag_forces_tensor = step::project_forces_perpendicular(full_mag_forces, magmoms);

    // Handle NaN after projection
    bool has_nan_projected = step::has_nan(mag_forces_tensor);
    if (has_nan_projected) {
      mag_forces_tensor = step::replace_nan(mag_forces_tensor, impl_->last_valid_projected_forces);
    }

    // Cache valid projected forces
    if (!has_nan_projected) {
      impl_->last_valid_projected_forces = mag_forces_tensor.detach().clone();
      impl_->has_valid_projected_forces = true;
    }

    // 10. Distribute atomic forces to LAMMPS arrays (only local atoms now)
    auto forces_cpu = forces_tensor.cpu();
    auto force_accessor = forces_cpu.accessor<float, 2>();

    for (int i = 0; i < nlocal; i++) {
      f[i][0] += static_cast<double>(force_accessor[i][0]);
      f[i][1] += static_cast<double>(force_accessor[i][1]);
      f[i][2] += static_cast<double>(force_accessor[i][2]);
    }

    // 11. Distribute magnetic forces (only local atoms now)
    auto mag_forces_cpu = mag_forces_tensor.cpu();
    auto mag_accessor = mag_forces_cpu.accessor<float, 2>();

    // Apply to fm array: fm += mag * projected_force / hbar
    for (int i = 0; i < nlocal; i++) {
      double mag = sp[i][3];
      if (mag > 1e-10) {
        fm[i][0] += mag * static_cast<double>(mag_accessor[i][0]) / hbar;
        fm[i][1] += mag * static_cast<double>(mag_accessor[i][1]) / hbar;
        fm[i][2] += mag * static_cast<double>(mag_accessor[i][2]) / hbar;
      }
    }

    // 12. Cache magnetic forces for compute_single_pair
    impl_->cached_mag_forces = mag_forces_cpu.detach().clone();
    impl_->cached_full_mag_forces = full_mag_forces.cpu().detach().clone();
    forces_cached_ = true;

    // 13. Energy bookkeeping
    if (eflag_global) {
      eng_vdwl = total_energy.item<double>();
    }

    if (eflag_atom) {
      auto atomic_e = atomic_energies.cpu();
      auto e_accessor = atomic_e.accessor<float, 2>();
      for (int i = 0; i < nlocal; i++) {
        eatom[i] = e_accessor[i][0];
      }
    }

  } catch (const c10::Error &e) {
    error->all(FLERR, "PyTorch error in SPIN-STEP compute: {}", e.what());
  } catch (const std::exception &e) {
    error->all(FLERR, "Error in SPIN-STEP compute: {}", e.what());
  }
}

// =============================================================================
// Compute Single Pair - Called by fix_nve_spin for each atom
// =============================================================================

void PairSpinSTEP::compute_single_pair(int ii, double fmi[3])
{
  static int single_pair_calls = 0;
  static int cached_nmagnetic = 0;
  static bigint last_step = -1;

  bigint current_step = update->ntimestep;
  double **sp = atom->sp;
  int nlocal = atom->nlocal;

  // Reset counter at new step
  if (current_step != last_step) {
    single_pair_calls = 0;
    last_step = current_step;
    // Count magnetic atoms
    cached_nmagnetic = 0;
    for (int i = 0; i < nlocal; i++) {
      if (sp[i][3] > 1e-10) cached_nmagnetic++;
    }
  }

  single_pair_calls++;
  int nmagnetic = cached_nmagnetic;

  // Recompute at start of sweeps 2, 3, 4 (not sweep 1)
  bool start_of_new_sweep = (nmagnetic > 0) &&
                            (single_pair_calls > nmagnetic) &&
                            ((single_pair_calls - 1) % nmagnetic == 0);

  if (!forces_cached_ || start_of_new_sweep) {
    recompute_forces();
  }

  if (!forces_cached_) {
    return;
  }

  double mag = sp[ii][3];
  if (mag < 1e-10) {
    return;
  }

  auto mag_forces_cpu = impl_->cached_mag_forces.cpu();
  auto mag_accessor = mag_forces_cpu.accessor<float, 2>();

  fmi[0] += mag * static_cast<double>(mag_accessor[ii][0]) / hbar;
  fmi[1] += mag * static_cast<double>(mag_accessor[ii][1]) / hbar;
  fmi[2] += mag * static_cast<double>(mag_accessor[ii][2]) / hbar;
}

// =============================================================================
// Recompute Forces - Recompute magnetic forces only
// =============================================================================

void PairSpinSTEP::recompute_forces()
{
  if (!model_loaded_) {
    return;
  }

  double **x = atom->x;
  double **sp = atom->sp;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;

  try {
    // Only convert local atoms (shifts handle periodic boundaries)
    auto positions = impl_->convert_positions(x, nlocal);
    auto numbers = impl_->convert_types(type, nlocal, elements_);
    auto magmoms = impl_->convert_spins_to_magmoms(sp, nlocal);

    // Only need gradients for magnetic moments
    positions.set_requires_grad(false);
    magmoms.set_requires_grad(true);

    double rc_sq = cutoff_ * cutoff_;
    auto neighbors = impl_->build_neighbor_list(list, nlocal, ntotal, x, rc_sq, tag, atom);

    if (neighbors.n_pairs == 0) {
      impl_->cached_mag_forces = torch::zeros({nlocal, 3}, torch::kFloat32);
      impl_->cached_full_mag_forces = torch::zeros({nlocal, 3}, torch::kFloat32);
      forces_cached_ = true;
      return;
    }

    // Build input dictionary
    c10::Dict<std::string, torch::Tensor> data_dict;
    data_dict.insert("pos", positions);
    data_dict.insert("numbers", numbers);
    data_dict.insert("magmoms", magmoms);
    data_dict.insert("edge_index", neighbors.edge_index);
    data_dict.insert("shifts", neighbors.shifts);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(data_dict);
    auto atomic_energies = impl_->model.forward(inputs).toTensor();

    // Check for NaN in energies
    if (step::has_nan(atomic_energies)) {
      error->warning(FLERR, "SPIN-STEP recompute_forces: NaN detected in atomic energies");
    }

    auto total_energy = atomic_energies.sum();

    auto grads = torch::autograd::grad(
        {total_energy},
        {magmoms},
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    auto mag_grads = grads[0];

    // Handle NaN
    bool has_nan_mag = step::has_nan(mag_grads);
    if (has_nan_mag) {
      mag_grads = step::replace_nan(mag_grads, impl_->last_valid_mag_grads);
    }
    if (!has_nan_mag) {
      impl_->last_valid_mag_grads = mag_grads.detach().clone();
      impl_->has_valid_grads = true;
    }

    // Project to perpendicular direction
    auto full_mag_forces = -mag_grads;  // full (unprojected) for longitudinal dynamics
    auto mag_forces_tensor = step::project_forces_perpendicular(full_mag_forces, magmoms);

    // Handle NaN after projection
    bool has_nan_projected = step::has_nan(mag_forces_tensor);
    if (has_nan_projected) {
      mag_forces_tensor = step::replace_nan(mag_forces_tensor, impl_->last_valid_projected_forces);
    }
    if (!has_nan_projected) {
      impl_->last_valid_projected_forces = mag_forces_tensor.detach().clone();
      impl_->has_valid_projected_forces = true;
    }

    // Cache magnetic forces (only local atoms now, no ghost accumulation needed)
    impl_->cached_mag_forces = mag_forces_tensor.cpu().detach().clone();
    impl_->cached_full_mag_forces = full_mag_forces.cpu().detach().clone();
    forces_cached_ = true;

  } catch (const c10::Error &e) {
    error->warning(FLERR, "PyTorch error in SPIN-STEP recompute_forces: {}", e.what());
    forces_cached_ = false;
  } catch (const std::exception &e) {
    error->warning(FLERR, "Error in SPIN-STEP recompute_forces: {}", e.what());
    forces_cached_ = false;
  }
}

// =============================================================================
// Distribute Cached Magnetic Forces - For RK4/SIB integrators
// =============================================================================

void PairSpinSTEP::distribute_cached_mag_forces()
{
  if (!forces_cached_) {
    return;
  }

  double **sp = atom->sp;
  double **fm = atom->fm;
  int nlocal = atom->nlocal;

  auto mag_forces_cpu = impl_->cached_mag_forces.cpu();
  auto mag_accessor = mag_forces_cpu.accessor<float, 2>();

  // Apply cached magnetic forces to fm array
  for (int i = 0; i < nlocal; i++) {
    double mag = sp[i][3];
    if (mag > 1e-10) {
      // fm += mag * cached_mag_forces / hbar
      fm[i][0] += mag * static_cast<double>(mag_accessor[i][0]) / hbar;
      fm[i][1] += mag * static_cast<double>(mag_accessor[i][1]) / hbar;
      fm[i][2] += mag * static_cast<double>(mag_accessor[i][2]) / hbar;
    }
  }
}

// =============================================================================
// Distribute Full (Unprojected) Magnetic Forces - For longitudinal dynamics
// =============================================================================

void PairSpinSTEP::distribute_full_mag_forces(double **fm_full, int nlocal_in)
{
  if (!forces_cached_) {
    for (int i = 0; i < nlocal_in; i++) {
      fm_full[i][0] = 0.0;
      fm_full[i][1] = 0.0;
      fm_full[i][2] = 0.0;
    }
    return;
  }

  auto full_forces_cpu = impl_->cached_full_mag_forces.cpu();
  auto full_accessor = full_forces_cpu.accessor<float, 2>();

  // fm_full = cached_full_mag_forces = -dE/dm (raw gradient in eV/μ_B)
  // NO mag/hbar conversion here! The longitudinal step needs the raw energy
  // gradient, not the transverse field (rad/ps). The transverse conversion
  // fm = mag * force / hbar is only for precession dynamics (dŝ/dt = ŝ × fm).
  // For longitudinal dynamics: d|m|/dt = γ_L * (-dE/d|m|), where γ_L is in
  // μ_B²/(eV·ps) and (-dE/d|m|) = (-dE/dm)·m̂ is in eV/μ_B.
  // This matches SPILADY's additive update: ds += gamma_S_HL * dt * Heff.
  for (int i = 0; i < nlocal_in; i++) {
    fm_full[i][0] = static_cast<double>(full_accessor[i][0]);
    fm_full[i][1] = static_cast<double>(full_accessor[i][1]);
    fm_full[i][2] = static_cast<double>(full_accessor[i][2]);
  }
}
