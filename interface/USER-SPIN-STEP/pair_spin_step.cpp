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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
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

  // Model architecture flags
  bool project_target_mag_force;
  double S_ref;
  bool supports_stress;
  std::string stress_mode;
  std::string model_definition;

  // Cached magnetic forces for compute_single_pair
  torch::Tensor cached_mag_forces;

  // Cached full (unprojected) magnetic forces for longitudinal dynamics
  torch::Tensor cached_full_mag_forces;

  // Private CPU accumulation buffer: [atomic force, full magnetic force].
  // Reverse communication sums these before writing owned LAMMPS forces.
  std::vector<std::array<double, 6>> contributions;

  PairSpinSTEPImpl() : device(torch::kCPU) {
    // Default configuration
    r_max = -1.0;  // never guess a cutoff: an undersized halo silently loses forces
    num_types = 1;
    num_features = 64;
    lmax = 2;
    num_layers = -1;
    avg_num_neighbors = 25.0;
    scale = 1.0;
    shift = 0.0;
    project_target_mag_force = false;
    S_ref = 1.0;
    supports_stress = true;
    stress_mode = "external_autograd_strain";
    model_definition = "unknown";
  }

  struct Graph {
    std::vector<int> atoms;  // tensor node -> LAMMPS index; energy centers first
    std::vector<std::array<int64_t, 2>> edges;
    std::vector<std::array<float, 3>> shifts;
  };

  // Only expand centers that can affect the requested owned energies in L layers.
  // The outermost nodes need embeddings, but not their own neighbor lists.
  Graph build_graph(NeighList *list, Atom *atom, int first, int count,
                    int depth, double cutoff, bool serial) {
    Graph graph;
    const int nall = atom->nlocal + atom->nghost;
    std::vector<int> node(nall, -1);
    for (int i = first; i < first + count; ++i) {
      node[i] = graph.atoms.size();
      graph.atoms.push_back(i);
    }
    size_t begin = 0;
    for (int layer = 0; layer < depth; ++layer) {
      const size_t end = graph.atoms.size();
      for (size_t ni = begin; ni < end; ++ni) {
        const int i = graph.atoms[ni];
        for (int k = 0; k < list->numneigh[i]; ++k) {
          const int image_j = list->firstneigh[i][k] & NEIGHMASK;
          double rsq = 0.0;
          for (int d = 0; d < 3; ++d) {
            const double dx = atom->x[image_j][d] - atom->x[i][d];
            rsq += dx * dx;
          }
          if (rsq >= cutoff * cutoff) continue;
          int j = image_j;
          // On one rank all periodic copies can share the owned node. This
          // avoids replicating the whole cell in GPU memory for small boxes.
          if (serial && j >= atom->nlocal) {
            j = atom->map(atom->tag[j]);
            if (j < 0 || j >= atom->nlocal)
              throw std::runtime_error("Cannot map periodic ghost to owned atom");
          }
          if (node[j] < 0) {
            node[j] = graph.atoms.size();
            graph.atoms.push_back(j);
          }
          graph.edges.push_back({static_cast<int64_t>(ni), node[j]});
          graph.shifts.push_back({
              static_cast<float>(atom->x[image_j][0] - atom->x[j][0]),
              static_cast<float>(atom->x[image_j][1] - atom->x[j][1]),
              static_cast<float>(atom->x[image_j][2] - atom->x[j][2])});
        }
      }
      begin = end;
    }
    return graph;
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
  no_virial_fdotr_compute = 1;

  // Create implementation object
  impl_ = std::make_unique<PairSpinSTEPImpl>();

  comm_forward = 4;  // spin direction and magnitude at SIB substeps
  comm_reverse = 6;  // atomic force and full magnetic force
  batch_size_ = 0;
  halo_layers_ = 0;

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

void PairSpinSTEP::settings(int narg, char **arg)
{
  std::string device = "auto";
  batch_size_ = 0;
  halo_layers_ = 0;
  for (int i = 0; i < narg; i += 2) {
    if (i + 1 == narg) error->all(FLERR, "Missing spin/step option value");
    if (strcmp(arg[i], "batch_size") == 0) {
      batch_size_ = utils::inumeric(FLERR, arg[i+1], false, lmp);
      if (batch_size_ < 0) error->all(FLERR, "spin/step batch_size must be nonnegative");
    } else if (strcmp(arg[i], "halo_layers") == 0) {
      halo_layers_ = utils::inumeric(FLERR, arg[i+1], false, lmp);
      if (halo_layers_ < 1) error->all(FLERR, "spin/step halo_layers must be positive");
    } else if (strcmp(arg[i], "device") == 0) {
      device = arg[i+1];
    } else error->all(FLERR, "Unknown spin/step option: {}", arg[i]);
  }
  // Rank within a shared-memory node, not the global MPI rank. Respect
  // CUDA_VISIBLE_DEVICES (including schedulers exposing one GPU per rank).
  int local_rank = comm->me;
#if MPI_VERSION >= 3
  MPI_Comm local_world;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_world);
  MPI_Comm_rank(local_world, &local_rank);
  MPI_Comm_free(&local_world);
#endif
  try {
    if (device == "auto") {
      if (torch::cuda::is_available())
        impl_->device = torch::Device(torch::kCUDA, local_rank % torch::cuda::device_count());
      else impl_->device = torch::Device(torch::kCPU);
    } else {
      impl_->device = torch::Device(device);
      if (!impl_->device.is_cpu() && !impl_->device.is_cuda())
        error->one(FLERR, "spin/step device must be auto, cpu, or cuda:N");
      if (impl_->device.is_cuda() && (!torch::cuda::is_available() ||
          !impl_->device.has_index() || impl_->device.index() >= torch::cuda::device_count()))
        error->one(FLERR, "spin/step CUDA device is unavailable; use a visible cuda:N index");
    }
  } catch (const c10::Error &e) {
    error->one(FLERR, "Invalid spin/step device: {}", e.what());
  }
  if (comm->me == 0)
    utils::logmesg(lmp, "SPIN-STEP: {} MPI rank(s), rank 0 device {}, batch_size={}\n",
                  comm->nprocs, impl_->device.str(), batch_size_);
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
    // map_location moves weights, but traced .to(device) calls also contain
    // Device constants. Retarget them so cuda:0 exports work on every rank.
    std::function<void(torch::jit::Block *)> retarget = [&](torch::jit::Block *block) {
      for (auto *node : block->nodes()) {
        if (node->kind() == torch::jit::prim::Constant && node->outputs().size() == 1 &&
            node->output()->type()->kind() == c10::TypeKind::DeviceObjType &&
            node->hasAttribute(torch::jit::attr::value))
          node->s_(torch::jit::attr::value, impl_->device.str());
        for (auto *sub : node->blocks()) retarget(sub);
      }
    };
    for (auto module : impl_->model.modules())
      for (auto method : module.get_methods()) retarget(method.graph()->block());

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

      // Parse model architecture flags
      impl_->project_target_mag_force = step::extract_bool(config_json, "project_target_mag_force", false);
      impl_->S_ref = step::extract_float(config_json, "S_ref", 1.0f);
      impl_->supports_stress = step::extract_bool(config_json, "supports_stress", true);
      impl_->stress_mode = step::extract_string(config_json, "stress_mode", "external_autograd_strain");
      impl_->model_definition = step::extract_string(config_json, "model_definition", "unknown");

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
        utils::logmesg(lmp, "SPIN-STEP: project_target_mag_force={}\n",
                      impl_->project_target_mag_force ? "true" : "false");
        utils::logmesg(lmp, "SPIN-STEP: S_ref={}\n", impl_->S_ref);
        utils::logmesg(lmp, "SPIN-STEP: model_definition={}\n", impl_->model_definition);
        utils::logmesg(lmp, "SPIN-STEP: supports_stress={}, stress_mode={}\n",
                      impl_->supports_stress ? "true" : "false", impl_->stress_mode);
      }

      if (!impl_->supports_stress && comm->me == 0) {
        error->warning(FLERR,
                       "SPIN-STEP: model metadata says supports_stress=false; "
                       "pair_style will still compute virial via dE/deps, but this model export may be inconsistent");
      }
      if (impl_->stress_mode != "external_autograd_strain" && comm->me == 0) {
        error->warning(FLERR,
                       "SPIN-STEP: unexpected stress_mode in model metadata; "
                       "current pair_style assumes stress is computed from homogeneous strain autodiff");
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

    if (halo_layers_ && impl_->num_layers > halo_layers_)
      error->all(FLERR, "spin/step halo_layers is smaller than model num_layers");
    if (!halo_layers_) halo_layers_ = impl_->num_layers;
    if (halo_layers_ < 1)
      error->all(FLERR, "spin/step needs num_layers in config.json or explicit halo_layers");
    if (!std::isfinite(impl_->r_max) || impl_->r_max <= 0)
      error->all(FLERR, "spin/step model r_max must be positive and finite");

    // Try to freeze model for faster inference
    if (impl_->model.hasattr("training")) {
      impl_->model = torch::jit::freeze(impl_->model);
    }

    model_loaded_ = true;

    if (comm->me == 0) {
      utils::logmesg(lmp, "SPIN-STEP: TorchScript model loaded successfully\n");
    }
  } catch (const std::exception &e) {
    error->one(FLERR, "Failed to load TorchScript model: {}", e.what());
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

  // Ghost centers need neighbors for multi-hop inference. The physical
  // neighbor cutoff stays r_max; only the communication halo is enlarged.
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  const double halo = (comm->nprocs == 1 ? cutoff_ : halo_layers_ * cutoff_) + neighbor->skin;
  comm->cutghostuser = std::max(comm->cutghostuser, halo);
  if (comm->mode != Comm::SINGLE)
    error->all(FLERR, "spin/step requires comm_modify mode single");
  if (neighbor->includegroup)
    error->all(FLERR, "spin/step does not support neigh_modify include");
  if (comm->me == 0)
    utils::logmesg(lmp, "SPIN-STEP: message depth={}, communication halo={} Angstrom\n",
                  halo_layers_, halo);

  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style spin/step requires newton pair on");

  // Check for compatible spin integration fix
  auto nve_spin_fixes = modify->get_fix_by_style("^nve/spin$");
  if (comm->nprocs > 1 && !nve_spin_fixes.empty())
    error->all(FLERR, "MPI spin/step requires a collective SIB integrator (e.g. nve/spin/sib); "
               "nve/spin uses asynchronous per-atom force evaluations");

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
  evaluate(true, eflag, vflag);
  distribute_cached_mag_forces();
}

// All ranks must call this, including those with zero owned atoms. Batches
// are independent energy sums; gradients are accumulated before communication.
void PairSpinSTEP::evaluate(bool mechanical, int /*eflag*/, int /*vflag*/)
{
  if (!model_loaded_) error->all(FLERR, "SPIN-STEP model not loaded");
  forces_cached_ = false;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  impl_->contributions.assign(nall, {});
  comm->forward_comm(this);
  try {
    const int batch = batch_size_ ? batch_size_ : std::max(nlocal, 1);
    const bool strain = mechanical && vflag_global;
    for (int first = 0; first < nlocal; first += batch) {
      const int count = std::min(batch, nlocal - first);
      auto graph = impl_->build_graph(list, atom, first, count, halo_layers_,
                                      cutoff_, comm->nprocs == 1);
      const int nnodes = graph.atoms.size();
      auto positions = torch::empty({nnodes, 3}, torch::kFloat32);
      auto magmoms = torch::empty_like(positions);
      auto numbers = torch::empty({nnodes}, torch::kInt64);
      auto pa = positions.accessor<float, 2>();
      auto ma = magmoms.accessor<float, 2>();
      auto na = numbers.accessor<int64_t, 1>();
      // Translate each graph near zero before float32 conversion, preserving
      // resolution for large simulation boxes. Translation does not change E.
      const int origin = graph.atoms[0];
      for (int ni = 0; ni < nnodes; ++ni) {
        const int i = graph.atoms[ni];
        na[ni] = impl_->atom_types_map.at(step::element_to_number(elements_[atom->type[i]-1]));
        for (int d = 0; d < 3; ++d) {
          pa[ni][d] = atom->x[i][d] - atom->x[origin][d];
          ma[ni][d] = atom->sp[i][d] * atom->sp[i][3];
        }
      }
      const int64_t nedges = graph.edges.size();
      auto edges = torch::empty({2, nedges}, torch::kInt64);
      auto shifts = torch::empty({nedges, 3}, torch::kFloat32);
      auto ea = edges.accessor<int64_t, 2>();
      auto sa = shifts.accessor<float, 2>();
      for (int64_t e = 0; e < nedges; ++e) {
        ea[0][e] = graph.edges[e][0];
        ea[1][e] = graph.edges[e][1];
        for (int d = 0; d < 3; ++d) sa[e][d] = graph.shifts[e][d];
      }
      positions = positions.to(impl_->device).set_requires_grad(mechanical);
      magmoms = magmoms.to(impl_->device).set_requires_grad(true);
      shifts = shifts.to(impl_->device);
      torch::Tensor eps;
      auto pos_def = positions;
      if (strain) {
        eps = torch::zeros({3, 3}, positions.options()).set_requires_grad(true);
        auto def = torch::eye(3, positions.options()) + eps;
        pos_def = torch::matmul(positions, def);
        shifts = torch::matmul(shifts, def);
      }
      c10::Dict<std::string, torch::Tensor> data;
      data.insert("pos", pos_def);
      data.insert("magmoms", magmoms);
      data.insert("numbers", numbers.to(impl_->device));
      data.insert("edge_index", edges.to(impl_->device));
      data.insert("shifts", shifts);
      auto atomic = impl_->model.forward({data}).toTensor();
      if (atomic.dim() < 1 || atomic.size(0) != nnodes || atomic.numel() != nnodes)
        throw std::runtime_error("STEP model must return one energy per input node");
      auto owned = atomic.narrow(0, 0, count).reshape({count});
      if (!torch::isfinite(owned).all().item<bool>())
        throw std::runtime_error("Non-finite STEP energy");
      // Never sum ghost energies. Keep isolated-atom/on-site energies too.
      auto energy = owned.sum();
      std::vector<torch::Tensor> inputs = {magmoms};
      if (mechanical) inputs.push_back(positions);
      if (strain) inputs.push_back(eps);
      std::vector<torch::Tensor> grads(inputs.size());
      if (energy.requires_grad())
        grads = torch::autograd::grad({energy}, inputs, {}, false, false, true);
      for (size_t k = 0; k < grads.size(); ++k) {
        if (!grads[k].defined()) grads[k] = torch::zeros_like(inputs[k]);
        if (!torch::isfinite(grads[k]).all().item<bool>())
          throw std::runtime_error("Non-finite STEP gradient");
      }
      auto mag_cpu = (-grads[0]).to(torch::kCPU).contiguous();
      auto ga = mag_cpu.accessor<float, 2>();
      for (int ni = 0; ni < nnodes; ++ni)
        for (int d = 0; d < 3; ++d)
          impl_->contributions[graph.atoms[ni]][3+d] += ga[ni][d];
      if (mechanical) {
        auto f_cpu = (-grads[1]).to(torch::kCPU).contiguous();
        auto fa = f_cpu.accessor<float, 2>();
        for (int ni = 0; ni < nnodes; ++ni)
          for (int d = 0; d < 3; ++d)
            impl_->contributions[graph.atoms[ni]][d] += fa[ni][d];
        if (eflag_global) eng_vdwl += energy.item<double>();
        if (eflag_atom) {
          auto e_cpu = owned.to(torch::kCPU).contiguous();
          auto values = e_cpu.accessor<float, 1>();
          for (int k = 0; k < count; ++k) eatom[first+k] += values[k];
        }
        if (strain) {
          auto vir = (-grads[2]).to(torch::kCPU).contiguous();
          auto va = vir.accessor<float, 2>();
          virial[0] += va[0][0]; virial[1] += va[1][1]; virial[2] += va[2][2];
          virial[3] += va[0][1]; virial[4] += va[0][2]; virial[5] += va[1][2];
        }
      }
    }  // Autograd graph released after every batch.
  } catch (const std::exception &e) {
    // A rank-local failure must abort MPI, not strand peers in reverse_comm.
    error->one(FLERR, "SPIN-STEP evaluation failed: {}", e.what());
  }
  comm->reverse_comm(this);
  impl_->cached_full_mag_forces = torch::empty({nlocal, 3}, torch::kFloat32);
  auto full = impl_->cached_full_mag_forces.accessor<float, 2>();
  auto moments = torch::empty({nlocal, 3}, torch::kFloat32);
  auto ma = moments.accessor<float, 2>();
  for (int i = 0; i < nlocal; ++i) {
    for (int d = 0; d < 3; ++d) {
      if (mechanical) atom->f[i][d] += impl_->contributions[i][d];
      full[i][d] = impl_->contributions[i][3+d];
      ma[i][d] = atom->sp[i][d] * atom->sp[i][3];
    }
  }
  // Project only after summing the full gradient, and keep the full field for
  // longitudinal SIB dynamics. No STEP contribution is left in ghost f/fm.
  impl_->cached_mag_forces = impl_->project_target_mag_force ?
      step::project_forces_perpendicular(impl_->cached_full_mag_forces, moments) :
      impl_->cached_full_mag_forces;
  forces_cached_ = true;
}

int PairSpinSTEP::pack_forward_comm(int n, int *indices, double *buf, int, int *)
{
  int k = 0;
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 4; ++d) buf[k++] = atom->sp[indices[i]][d];
  return k;
}

void PairSpinSTEP::unpack_forward_comm(int n, int first, double *buf)
{
  int k = 0;
  for (int i = first; i < first+n; ++i)
    for (int d = 0; d < 4; ++d) atom->sp[i][d] = buf[k++];
}

int PairSpinSTEP::pack_reverse_comm(int n, int first, double *buf)
{
  int k = 0;
  for (int i = first; i < first+n; ++i)
    for (int d = 0; d < 6; ++d) buf[k++] = impl_->contributions[i][d];
  return k;
}

void PairSpinSTEP::unpack_reverse_comm(int n, int *indices, double *buf)
{
  int k = 0;
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 6; ++d) impl_->contributions[indices[i]][d] += buf[k++];
}

// =============================================================================
// Compute Single Pair - Called by fix_nve_spin for each atom
// =============================================================================

void PairSpinSTEP::compute_single_pair(int ii, double fmi[3])
{
  if (comm->nprocs > 1)
    error->one(FLERR, "MPI spin/step does not support per-atom spin sweeps; use a SIB integrator");
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
  evaluate(false, 0, 0);
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
