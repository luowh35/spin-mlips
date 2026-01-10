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
   Contributing author: NEP-SPIN LAMMPS integration
   NEP-SPIN: Machine learning potential for magnetic systems
------------------------------------------------------------------------- */

#include "pair_nep_spin.h"
#include "nep_spin_data.h"
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
#include "fix_nve_spin_sib.h"

// Torch headers - only in cpp file
#include <torch/script.h>
#include <torch/torch.h>
#include "nep_types.h"
#include "descriptor.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <regex>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

// Implementation struct that hides torch dependencies
struct LAMMPS_NS::PairNEPSpinImpl {
  // TorchScript model
  torch::jit::Module model;
  torch::Device device;

  // Descriptor configuration and instance
  nep::DescriptorConfig descriptor_config;
  std::unique_ptr<nep::MagneticACEDescriptor> descriptor;

  // Cached magnetic forces for compute_single_pair
  torch::Tensor cached_mag_forces;

  PairNEPSpinImpl() : device(torch::kCPU) {
    // Default descriptor config
    descriptor_config.rc = 4.7f;
    descriptor_config.n_max = 5;
    descriptor_config.l_max = 3;
    descriptor_config.nu_max = 2;
    descriptor_config.m_cut = 3.5f;
    descriptor_config.use_spin_invariants = true;
    descriptor_config.pos_scale = 100.0f;
    descriptor_config.spin_scale = 1.0f;
    descriptor_config.epsilon = 1e-7f;
    descriptor_config.mag_noise_threshold = 0.35f;
    descriptor_config.pos_noise_threshold = 1e-8f;
    // Increased pole_threshold to avoid numerical instability in gradient computation
    // when spins are nearly aligned to z-axis
    descriptor_config.pole_threshold = 0.05f;
  }

  // Cache for last valid magnetic gradients (used when NaN occurs)
  torch::Tensor last_valid_mag_grads;
  bool has_valid_grads = false;

  // Cache for last valid projected forces (used when projection produces NaN)
  torch::Tensor last_valid_projected_forces;
  bool has_valid_projected_forces = false;

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
      num_accessor[i] = nep::element_to_number(elem);
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

  torch::Tensor get_cell_tensor(Domain *domain) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto cell = torch::zeros({3, 3}, options);
    auto cell_accessor = cell.accessor<float, 2>();

    double *boxlo = domain->boxlo;
    double *boxhi = domain->boxhi;

    cell_accessor[0][0] = static_cast<float>(boxhi[0] - boxlo[0]);
    cell_accessor[0][1] = 0.0f;
    cell_accessor[0][2] = 0.0f;

    if (domain->triclinic) {
      cell_accessor[1][0] = static_cast<float>(domain->xy);
      cell_accessor[1][1] = static_cast<float>(boxhi[1] - boxlo[1]);
      cell_accessor[1][2] = 0.0f;
      cell_accessor[2][0] = static_cast<float>(domain->xz);
      cell_accessor[2][1] = static_cast<float>(domain->yz);
      cell_accessor[2][2] = static_cast<float>(boxhi[2] - boxlo[2]);
    } else {
      cell_accessor[1][0] = 0.0f;
      cell_accessor[1][1] = static_cast<float>(boxhi[1] - boxlo[1]);
      cell_accessor[1][2] = 0.0f;
      cell_accessor[2][0] = 0.0f;
      cell_accessor[2][1] = 0.0f;
      cell_accessor[2][2] = static_cast<float>(boxhi[2] - boxlo[2]);
    }

    return cell.to(device);
  }

  nep::NeighborList build_neighbor_list(NeighList *list, int ntotal, double **x, double rc_sq) {
    nep::NeighborList result;
    result.n_pairs = 0;

    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    // First pass: count pairs within cutoff
    std::vector<std::pair<int, int>> valid_pairs;
    valid_pairs.reserve(inum * 20);  // Estimate

    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      int *jlist = firstneigh[i];
      int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj] & NEIGHMASK;

        // Calculate distance squared
        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];
        double rsq = dx*dx + dy*dy + dz*dz;

        // Only include pairs within cutoff
        if (rsq < rc_sq) {
          valid_pairs.emplace_back(i, j);
        }
      }
    }

    int total_pairs = valid_pairs.size();

    if (total_pairs == 0) {
      result.center_indices = torch::zeros({0}, torch::kInt64).to(device);
      result.neighbor_indices = torch::zeros({0}, torch::kInt64).to(device);
      result.shifts = torch::zeros({0, 3}, torch::kFloat32).to(device);
      result.batch_idx = torch::zeros({ntotal}, torch::kInt64).to(device);
      return result;
    }

    auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    result.center_indices = torch::zeros({total_pairs}, options_int);
    result.neighbor_indices = torch::zeros({total_pairs}, options_int);
    result.shifts = torch::zeros({total_pairs, 3}, options_float);
    result.batch_idx = torch::zeros({ntotal}, options_int);

    auto center_accessor = result.center_indices.accessor<int64_t, 1>();
    auto neighbor_accessor = result.neighbor_indices.accessor<int64_t, 1>();

    for (int pair_idx = 0; pair_idx < total_pairs; pair_idx++) {
      center_accessor[pair_idx] = valid_pairs[pair_idx].first;
      neighbor_accessor[pair_idx] = valid_pairs[pair_idx].second;
    }

    result.center_indices = result.center_indices.to(device);
    result.neighbor_indices = result.neighbor_indices.to(device);
    result.shifts = result.shifts.to(device);
    result.batch_idx = result.batch_idx.to(device);

    result.n_pairs = total_pairs;
    return result;
  }
};

// Helper functions for parsing JSON config from model file
namespace {
  float extract_float(const std::string& json, const std::string& key, float default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([\\d.eE+-]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
      return std::stof(match[1].str());
    }
    return default_val;
  }

  int extract_int(const std::string& json, const std::string& key, int default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
      return std::stoi(match[1].str());
    }
    return default_val;
  }

  bool extract_bool(const std::string& json, const std::string& key, bool default_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
      return match[1].str() == "true";
    }
    return default_val;
  }

  std::vector<std::string> extract_elements(const std::string& json) {
    std::vector<std::string> elements;
    std::regex pattern("\"elements\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
      std::string arr = match[1].str();
      std::regex elem_pattern("\"([^\"]+)\"");
      std::sregex_iterator iter(arr.begin(), arr.end(), elem_pattern);
      std::sregex_iterator end;
      while (iter != end) {
        elements.push_back((*iter)[1].str());
        ++iter;
      }
    }
    return elements;
  }
}

/* ---------------------------------------------------------------------- */

PairNEPSpin::PairNEPSpin(LAMMPS *lmp) : PairSpin(lmp)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  // Create implementation object
  impl_ = std::make_unique<PairNEPSpinImpl>();

  // Auto-detect GPU and use if available
  if (torch::cuda::is_available()) {
    int num_gpus = torch::cuda::device_count();
    int gpu_id = comm->me % num_gpus;
    impl_->device = torch::Device(torch::kCUDA, gpu_id);
    if (comm->me == 0) {
      utils::logmesg(lmp, "NEP-SPIN: {} CUDA GPU(s) detected, using GPU acceleration\n", num_gpus);
    }
  } else {
    impl_->device = torch::kCPU;
    if (comm->me == 0) {
      utils::logmesg(lmp, "NEP-SPIN: No CUDA GPU detected, using CPU\n");
    }
  }

  model_loaded_ = false;
  forces_cached_ = false;
  sp_magnitude_ = nullptr;
  cutoff_ = 0.0;
  m_cut_ = 3.5;
}

/* ---------------------------------------------------------------------- */

PairNEPSpin::~PairNEPSpin()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  if (sp_magnitude_) {
    memory->destroy(sp_magnitude_);
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::allocate()
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

/* ---------------------------------------------------------------------- */

void PairNEPSpin::settings(int narg, char ** /*arg*/)
{
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command: spin/nep takes no arguments");
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg < 4)
    error->all(FLERR, "Incorrect args for pair coefficients: "
               "pair_coeff * * <model.pt> <elem1> [elem2 ...]");

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for spin/nep must use * * wildcard");

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

  impl_->descriptor_config.elements = elements_;

  load_model(model_path_);

  cutoff_ = impl_->descriptor_config.rc;

  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoff_ * cutoff_;
    }
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "NEP-SPIN: Model loaded from {}\n", model_path_);
    utils::logmesg(lmp, "NEP-SPIN: Cutoff = {} Angstrom\n", cutoff_);
    utils::logmesg(lmp, "NEP-SPIN: Elements:");
    for (const auto& elem : elements_) {
      utils::logmesg(lmp, " {}", elem);
    }
    utils::logmesg(lmp, "\n");
    utils::logmesg(lmp, "NEP-SPIN: Magnetic moments read from data file (sp[i][3])\n");
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::load_model(const std::string &path)
{
  try {
    std::unordered_map<std::string, std::string> extra_files;
    extra_files["config.json"] = "";

    impl_->model = torch::jit::load(path, impl_->device, extra_files);
    impl_->model.eval();

    std::string config_json = extra_files["config.json"];
    if (!config_json.empty()) {
      if (comm->me == 0) {
        utils::logmesg(lmp, "NEP-SPIN: Found embedded config in model file\n");
      }

      std::regex desc_pattern("\"descriptor_config\"\\s*:\\s*\\{([^}]+)\\}");
      std::smatch desc_match;
      if (std::regex_search(config_json, desc_match, desc_pattern)) {
        std::string desc_json = desc_match[1].str();

        impl_->descriptor_config.rc = extract_float(desc_json, "rc", impl_->descriptor_config.rc);
        impl_->descriptor_config.n_max = extract_int(desc_json, "n_max", impl_->descriptor_config.n_max);
        impl_->descriptor_config.l_max = extract_int(desc_json, "l_max", impl_->descriptor_config.l_max);
        impl_->descriptor_config.nu_max = extract_int(desc_json, "nu_max", impl_->descriptor_config.nu_max);
        impl_->descriptor_config.m_cut = extract_float(desc_json, "m_cut", impl_->descriptor_config.m_cut);
        impl_->descriptor_config.use_spin_invariants = extract_bool(desc_json, "use_spin_invariants", impl_->descriptor_config.use_spin_invariants);
        impl_->descriptor_config.pos_scale = extract_float(desc_json, "pos_scale", impl_->descriptor_config.pos_scale);
        impl_->descriptor_config.spin_scale = extract_float(desc_json, "spin_scale", impl_->descriptor_config.spin_scale);
        impl_->descriptor_config.epsilon = extract_float(desc_json, "epsilon", impl_->descriptor_config.epsilon);
        impl_->descriptor_config.mag_noise_threshold = extract_float(desc_json, "mag_noise_threshold", impl_->descriptor_config.mag_noise_threshold);
        impl_->descriptor_config.pos_noise_threshold = extract_float(desc_json, "pos_noise_threshold", impl_->descriptor_config.pos_noise_threshold);
        impl_->descriptor_config.pole_threshold = extract_float(desc_json, "pole_threshold", impl_->descriptor_config.pole_threshold);

        if (comm->me == 0) {
          utils::logmesg(lmp, "NEP-SPIN: Loaded config: rc={}, n_max={}, l_max={}, pos_scale={}\n",
                        impl_->descriptor_config.rc, impl_->descriptor_config.n_max,
                        impl_->descriptor_config.l_max, impl_->descriptor_config.pos_scale);
        }
      }
    } else {
      if (comm->me == 0) {
        utils::logmesg(lmp, "NEP-SPIN: No embedded config, using default values\n");
      }
    }

    if (impl_->model.hasattr("training")) {
      impl_->model = torch::jit::freeze(impl_->model);
    }

    model_loaded_ = true;

    if (comm->me == 0) {
      utils::logmesg(lmp, "NEP-SPIN: TorchScript model loaded successfully\n");
    }
  } catch (const c10::Error &e) {
    error->all(FLERR, "Failed to load TorchScript model: {}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style spin/nep requires atom IDs");

  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "Pair style spin/nep requires metal units");

  if (!atom->sp_flag)
    error->all(FLERR, "Pair style spin/nep requires atom_style spin");

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style spin/nep requires newton pair on");

  try {
    impl_->descriptor = std::make_unique<nep::MagneticACEDescriptor>(impl_->descriptor_config);
  } catch (const std::exception &e) {
    error->all(FLERR, "Failed to initialize NEP descriptor: {}", e.what());
  }

  // Check for compatible spin integration fix
  // Support standard nve/spin and SIB nve/spin/sib
  auto nve_spin_fixes = modify->get_fix_by_style("^nve/spin$");
  auto nve_spin_sib_fixes = modify->get_fix_by_style("^nve/spin/sib$");
  auto neb_spin_fixes = modify->get_fix_by_style("^neb/spin$");

  int total_spin_fixes =
      nve_spin_fixes.size() + nve_spin_sib_fixes.size() + neb_spin_fixes.size();

  if ((comm->me == 0) && (total_spin_fixes == 0))
    error->warning(FLERR, "Using spin pair style without nve/spin, nve/spin/sib, or neb/spin");

  // Get lattice_flag from the appropriate fix
  if (nve_spin_sib_fixes.size() == 1) {
    // Using SIB method
    lattice_flag = (dynamic_cast<FixNVESpinSIB *>(nve_spin_sib_fixes.front()))->lattice_flag;
  } else if (nve_spin_fixes.size() == 1) {
    // Using standard Suzuki-Trotter method
    lattice_flag = (dynamic_cast<FixNVESpin *>(nve_spin_fixes.front()))->lattice_flag;
  } else if (total_spin_fixes > 1) {
    error->warning(FLERR, "Using multiple instances of spin integration fixes");
  }

  // Initialize size of energy stacking lists (from PairSpin)
  nlocal_max = atom->nlocal;
}

/* ---------------------------------------------------------------------- */

double PairNEPSpin::init_one(int /*i*/, int /*j*/)
{
  return cutoff_;
}

/* ---------------------------------------------------------------------- */

void *PairNEPSpin::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "cut") == 0) return (void *) &cutoff_;
  if (strcmp(str, "sp_magnitude") == 0) {
    dim = 1;
    return (void *) sp_magnitude_;
  }
  return nullptr;
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (!model_loaded_) {
    error->all(FLERR, "NEP-SPIN model not loaded");
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
    auto positions = impl_->convert_positions(x, ntotal);
    auto numbers = impl_->convert_types(type, ntotal, elements_);
    auto magmoms = impl_->convert_spins_to_magmoms(sp, ntotal);
    auto cell = impl_->get_cell_tensor(domain);

    positions.set_requires_grad(true);
    magmoms.set_requires_grad(true);

    double rc_sq = cutoff_ * cutoff_;
    auto neighbors = impl_->build_neighbor_list(list, ntotal, x, rc_sq);

    if (neighbors.n_pairs == 0) {
      eng_vdwl = 0.0;
      forces_cached_ = true;
      impl_->cached_mag_forces = torch::zeros({ntotal, 3}, torch::kFloat32);
      return;
    }

    auto descriptors_all = impl_->descriptor->compute_from_precomputed_neighbors(
        positions, numbers, magmoms, neighbors, cell);

    // Check for NaN in descriptors
    if (torch::any(torch::isnan(descriptors_all)).item<bool>()) {
      error->warning(FLERR, "NEP-SPIN: NaN detected in descriptors");
    }

    auto descriptors = descriptors_all.slice(0, 0, nlocal);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(descriptors);
    auto atomic_energies = impl_->model.forward(inputs).toTensor();

    // Check for NaN in energies
    if (torch::any(torch::isnan(atomic_energies)).item<bool>()) {
      error->warning(FLERR, "NEP-SPIN: NaN detected in atomic energies");
    }

    auto total_energy = atomic_energies.sum();

    auto grads = torch::autograd::grad(
        {total_energy},
        {positions, magmoms},
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    // Handle NaN in gradients using cached valid gradients
    auto pos_grads = grads[0];
    auto mag_grads = grads[1];

    // Check and handle NaN in position gradients
    if (torch::any(torch::isnan(pos_grads)).item<bool>()) {
      error->warning(FLERR, "NEP-SPIN: NaN detected in position gradients, replacing with zeros");
      pos_grads = torch::where(torch::isnan(pos_grads), torch::zeros_like(pos_grads), pos_grads);
    }

    // Check and handle NaN in magnetic gradients
    bool has_nan_mag = torch::any(torch::isnan(mag_grads)).item<bool>();
    if (has_nan_mag) {
      if (impl_->has_valid_grads && impl_->last_valid_mag_grads.defined() &&
          impl_->last_valid_mag_grads.size(0) == mag_grads.size(0)) {
        // Use cached valid gradients for NaN values
        mag_grads = torch::where(torch::isnan(mag_grads), impl_->last_valid_mag_grads, mag_grads);
      } else {
        // No valid cache, replace with zeros
        mag_grads = torch::where(torch::isnan(mag_grads), torch::zeros_like(mag_grads), mag_grads);
      }
    }

    // Cache valid gradients for future use
    if (!has_nan_mag) {
      impl_->last_valid_mag_grads = mag_grads.detach().clone();
      impl_->has_valid_grads = true;
    }

    auto forces_tensor = -pos_grads;
    // Magnetic forces: -grads[1] = -dE/dM (same as Python)
    // MUST project to perpendicular direction for spin dynamics stability
    // Without projection, parallel components change spin magnitude -> NaN
    auto mag_forces_tensor = nep::MathUtils::project_forces_perpendicular(-mag_grads, magmoms);

    // Check and handle NaN after projection using cached projected forces
    bool has_nan_projected = torch::any(torch::isnan(mag_forces_tensor)).item<bool>();
    if (has_nan_projected) {
      if (impl_->has_valid_projected_forces && impl_->last_valid_projected_forces.defined() &&
          impl_->last_valid_projected_forces.size(0) == mag_forces_tensor.size(0)) {
        // Use cached valid projected forces for NaN values
        mag_forces_tensor = torch::where(torch::isnan(mag_forces_tensor),
                                         impl_->last_valid_projected_forces, mag_forces_tensor);
      } else {
        // No valid cache, replace with zeros
        mag_forces_tensor = torch::where(torch::isnan(mag_forces_tensor),
                                         torch::zeros_like(mag_forces_tensor), mag_forces_tensor);
      }
    }

    // Cache valid projected forces for future use
    if (!has_nan_projected) {
      impl_->last_valid_projected_forces = mag_forces_tensor.detach().clone();
      impl_->has_valid_projected_forces = true;
    }

    // Distribute forces
    auto forces_cpu = forces_tensor.cpu();
    auto force_accessor = forces_cpu.accessor<float, 2>();

    for (int i = 0; i < nlocal; i++) {
      f[i][0] += static_cast<double>(force_accessor[i][0]);
      f[i][1] += static_cast<double>(force_accessor[i][1]);
      f[i][2] += static_cast<double>(force_accessor[i][2]);
    }

    for (int j = nlocal; j < ntotal; j++) {
      int local_i = atom->map(tag[j]);
      if (local_i >= 0 && local_i < nlocal) {
        f[local_i][0] += static_cast<double>(force_accessor[j][0]);
        f[local_i][1] += static_cast<double>(force_accessor[j][1]);
        f[local_i][2] += static_cast<double>(force_accessor[j][2]);
      }
    }

    // Distribute magnetic forces: fm -= mag * dE/dM / hbar
    auto mag_forces_cpu = mag_forces_tensor.cpu();
    auto mag_accessor = mag_forces_cpu.accessor<float, 2>();

    // First accumulate magnetic force gradients from all atoms (local + ghost)
    // to properly handle periodic boundary contributions
    std::vector<double> accumulated_mag_forces(nlocal * 3, 0.0);

    // Local atom contributions
    for (int i = 0; i < nlocal; i++) {
      accumulated_mag_forces[i*3 + 0] = static_cast<double>(mag_accessor[i][0]);
      accumulated_mag_forces[i*3 + 1] = static_cast<double>(mag_accessor[i][1]);
      accumulated_mag_forces[i*3 + 2] = static_cast<double>(mag_accessor[i][2]);
    }

    // Ghost atom contributions (map back to local atoms)
    for (int j = nlocal; j < ntotal; j++) {
      int local_i = atom->map(tag[j]);
      if (local_i >= 0 && local_i < nlocal) {
        accumulated_mag_forces[local_i*3 + 0] += static_cast<double>(mag_accessor[j][0]);
        accumulated_mag_forces[local_i*3 + 1] += static_cast<double>(mag_accessor[j][1]);
        accumulated_mag_forces[local_i*3 + 2] += static_cast<double>(mag_accessor[j][2]);
      }
    }

    // Apply to fm
    for (int i = 0; i < nlocal; i++) {
      double mag = sp[i][3];
      if (mag > 1e-10) {
        // mag_forces = project(-dE/dM) = -project(dE/dM)
        // fm += mag * project(-dE/dM) / hbar = fm -= mag * project(dE/dM) / hbar
        fm[i][0] += mag * accumulated_mag_forces[i*3 + 0] / hbar;
        fm[i][1] += mag * accumulated_mag_forces[i*3 + 1] / hbar;
        fm[i][2] += mag * accumulated_mag_forces[i*3 + 2] / hbar;
      }
    }

    // Store accumulated values in cache for compute_single_pair
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto accumulated_tensor = torch::zeros({nlocal, 3}, options);
    auto acc_tensor_accessor = accumulated_tensor.accessor<float, 2>();
    for (int i = 0; i < nlocal; i++) {
      acc_tensor_accessor[i][0] = static_cast<float>(accumulated_mag_forces[i*3 + 0]);
      acc_tensor_accessor[i][1] = static_cast<float>(accumulated_mag_forces[i*3 + 1]);
      acc_tensor_accessor[i][2] = static_cast<float>(accumulated_mag_forces[i*3 + 2]);
    }
    impl_->cached_mag_forces = accumulated_tensor.detach().clone();
    forces_cached_ = true;

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
    error->all(FLERR, "PyTorch error in NEP-SPIN compute: {}", e.what());
  } catch (const std::exception &e) {
    error->all(FLERR, "Error in NEP-SPIN compute: {}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::compute_single_pair(int ii, double fmi[3])
{
  // fix_nve_spin uses Suzuki-Trotter decomposition with 4 sweeps per step
  // Each sweep goes through all magnetic atoms, calling compute_single_pair
  //
  // Force evaluation schedule (4 times per step total):
  //   - 1st: done in compute() at step start (forces_cached_ = true)
  //   - 2nd, 3rd, 4th: done here at start of sweeps 2, 3, 4
  //
  // Sweep detection: each sweep has nmagnetic calls
  //   - Sweep 1: calls 1 to nmagnetic (use cached from compute())
  //   - Sweep 2: calls nmagnetic+1 to 2*nmagnetic (recompute at start)
  //   - Sweep 3: calls 2*nmagnetic+1 to 3*nmagnetic (recompute at start)
  //   - Sweep 4: calls 3*nmagnetic+1 to 4*nmagnetic (recompute at start)

  static int single_pair_calls = 0;
  static int cached_nmagnetic = 0;
  static bigint last_step = -1;

  bigint current_step = update->ntimestep;
  double **sp = atom->sp;
  int nlocal = atom->nlocal;

  // Reset counter and recount magnetic atoms at new step
  if (current_step != last_step) {
    single_pair_calls = 0;
    last_step = current_step;
    // Count magnetic atoms once per step
    cached_nmagnetic = 0;
    for (int i = 0; i < nlocal; i++) {
      if (sp[i][3] > 1e-10) cached_nmagnetic++;
    }
  }

  single_pair_calls++;
  int nmagnetic = cached_nmagnetic;

  // Recompute at start of sweeps 2, 3, 4 (not sweep 1, which uses compute() result)
  // Sweep 2 starts at call nmagnetic+1, sweep 3 at 2*nmagnetic+1, sweep 4 at 3*nmagnetic+1
  bool start_of_new_sweep = (nmagnetic > 0) &&
                            (single_pair_calls > nmagnetic) &&  // Not first sweep
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
  auto mag_force_accessor = mag_forces_cpu.accessor<float, 2>();

  // cached_mag_forces contains project(-dE/dM)
  // fm += mag * project(-dE/dM) / hbar = fm -= mag * project(dE/dM) / hbar
  fmi[0] += mag * static_cast<double>(mag_force_accessor[ii][0]) / hbar;
  fmi[1] += mag * static_cast<double>(mag_force_accessor[ii][1]) / hbar;
  fmi[2] += mag * static_cast<double>(mag_force_accessor[ii][2]) / hbar;
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::recompute_forces()
{
  if (!model_loaded_) {
    return;
  }

  double **x = atom->x;
  double **sp = atom->sp;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;

  try {
    auto positions = impl_->convert_positions(x, ntotal);
    auto numbers = impl_->convert_types(type, ntotal, elements_);
    auto magmoms = impl_->convert_spins_to_magmoms(sp, ntotal);
    auto cell = impl_->get_cell_tensor(domain);

    positions.set_requires_grad(false);
    magmoms.set_requires_grad(true);

    double rc_sq = cutoff_ * cutoff_;
    auto neighbors = impl_->build_neighbor_list(list, ntotal, x, rc_sq);

    if (neighbors.n_pairs == 0) {
      impl_->cached_mag_forces = torch::zeros({ntotal, 3}, torch::kFloat32);
      forces_cached_ = true;
      return;
    }

    auto descriptors_all = impl_->descriptor->compute_from_precomputed_neighbors(
        positions, numbers, magmoms, neighbors, cell);

    auto descriptors = descriptors_all.slice(0, 0, nlocal);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(descriptors);
    auto atomic_energies = impl_->model.forward(inputs).toTensor();

    auto total_energy = atomic_energies.sum();

    auto grads = torch::autograd::grad(
        {total_energy},
        {magmoms},
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    // -grads[0] = -dE/dM (same as Python), project to perpendicular direction
    auto mag_grads = grads[0];

    // Handle NaN in magnetic gradients using cached valid gradients
    bool has_nan_mag = torch::any(torch::isnan(mag_grads)).item<bool>();
    if (has_nan_mag) {
      if (impl_->has_valid_grads && impl_->last_valid_mag_grads.defined() &&
          impl_->last_valid_mag_grads.size(0) == mag_grads.size(0)) {
        mag_grads = torch::where(torch::isnan(mag_grads), impl_->last_valid_mag_grads, mag_grads);
      } else {
        mag_grads = torch::where(torch::isnan(mag_grads), torch::zeros_like(mag_grads), mag_grads);
      }
    }
    if (!has_nan_mag) {
      impl_->last_valid_mag_grads = mag_grads.detach().clone();
      impl_->has_valid_grads = true;
    }

    auto mag_forces_tensor = nep::MathUtils::project_forces_perpendicular(-mag_grads, magmoms);

    // Handle NaN after projection using cached projected forces
    bool has_nan_projected = torch::any(torch::isnan(mag_forces_tensor)).item<bool>();
    if (has_nan_projected) {
      if (impl_->has_valid_projected_forces && impl_->last_valid_projected_forces.defined() &&
          impl_->last_valid_projected_forces.size(0) == mag_forces_tensor.size(0)) {
        mag_forces_tensor = torch::where(torch::isnan(mag_forces_tensor),
                                         impl_->last_valid_projected_forces, mag_forces_tensor);
      } else {
        mag_forces_tensor = torch::where(torch::isnan(mag_forces_tensor),
                                         torch::zeros_like(mag_forces_tensor), mag_forces_tensor);
      }
    }
    if (!has_nan_projected) {
      impl_->last_valid_projected_forces = mag_forces_tensor.detach().clone();
      impl_->has_valid_projected_forces = true;
    }

    // Accumulate ghost atom contributions to local atoms
    auto mag_forces_cpu = mag_forces_tensor.cpu();
    auto mag_accessor = mag_forces_cpu.accessor<float, 2>();

    // Create accumulated tensor for local atoms
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto accumulated = torch::zeros({nlocal, 3}, options);
    auto acc_accessor = accumulated.accessor<float, 2>();

    // Local atom contributions
    for (int i = 0; i < nlocal; i++) {
      acc_accessor[i][0] = mag_accessor[i][0];
      acc_accessor[i][1] = mag_accessor[i][1];
      acc_accessor[i][2] = mag_accessor[i][2];
    }

    // Ghost atom contributions (map back to local atoms)
    tagint *tag = atom->tag;
    for (int j = nlocal; j < ntotal; j++) {
      int local_i = atom->map(tag[j]);
      if (local_i >= 0 && local_i < nlocal) {
        acc_accessor[local_i][0] += mag_accessor[j][0];
        acc_accessor[local_i][1] += mag_accessor[j][1];
        acc_accessor[local_i][2] += mag_accessor[j][2];
      }
    }

    impl_->cached_mag_forces = accumulated.detach().clone();
    forces_cached_ = true;

  } catch (const c10::Error &e) {
    error->warning(FLERR, "PyTorch error in NEP-SPIN recompute_forces: {}", e.what());
    forces_cached_ = false;
  } catch (const std::exception &e) {
    error->warning(FLERR, "Error in NEP-SPIN recompute_forces: {}", e.what());
    forces_cached_ = false;
  }
}

/* ----------------------------------------------------------------------
   Distribute cached magnetic forces to fm array
   Called by fix_nve_spin_sib after recompute_forces()
------------------------------------------------------------------------- */

void PairNEPSpin::distribute_cached_mag_forces()
{
  if (!forces_cached_) {
    return;
  }

  double **sp = atom->sp;
  double **fm = atom->fm;
  int nlocal = atom->nlocal;

  auto mag_accessor = impl_->cached_mag_forces.accessor<float, 2>();

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
