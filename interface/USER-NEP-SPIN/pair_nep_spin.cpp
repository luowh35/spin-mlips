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
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNEPSpin::PairNEPSpin(LAMMPS *lmp) : PairSpin(lmp), device_(torch::kCPU)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  // Auto-detect GPU and use if available
  if (torch::cuda::is_available()) {
    int num_gpus = torch::cuda::device_count();
    int gpu_id = comm->me % num_gpus;  // Distribute MPI ranks across GPUs
    device_ = torch::Device(torch::kCUDA, gpu_id);
    if (comm->me == 0) {
      utils::logmesg(lmp, "NEP-SPIN: {} CUDA GPU(s) detected, using GPU acceleration\n", num_gpus);
    }
  } else {
    device_ = torch::kCPU;
    if (comm->me == 0) {
      utils::logmesg(lmp, "NEP-SPIN: No CUDA GPU detected, using CPU\n");
    }
  }

  model_loaded_ = false;
  forces_cached_ = false;
  sp_magnitude_ = nullptr;
  cutoff_ = 0.0;
  m_cut_ = 3.5;

  // Default descriptor config
  descriptor_config_.rc = 4.7f;
  descriptor_config_.n_max = 5;
  descriptor_config_.l_max = 3;
  descriptor_config_.nu_max = 2;
  descriptor_config_.m_cut = 3.5f;
  descriptor_config_.use_spin_invariants = true;
  descriptor_config_.pos_scale = 200.0f;
  descriptor_config_.spin_scale = 1.0f;
  descriptor_config_.epsilon = 1e-7f;
  descriptor_config_.mag_noise_threshold = 0.35f;
  descriptor_config_.pos_noise_threshold = 1e-8f;
  descriptor_config_.pole_threshold = 1e-6f;
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
    sp_magnitude_[i] = 1.0;  // Default magnetic moment magnitude
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

  // pair_coeff * * model.pt elem1 elem2 ...
  // Minimum: * * model.pt elem1
  if (narg < 4)
    error->all(FLERR, "Incorrect args for pair coefficients: "
               "pair_coeff * * <model.pt> <elem1> [elem2 ...]");

  // Check * *
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for spin/nep must use * * wildcard");

  // Model path
  model_path_ = arg[2];

  // Parse elements (magnetic moments are read from data file sp[i][3])
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

  // Set up type mapper
  type_mapper_.resize(ntypes);
  for (int i = 0; i < ntypes; i++) {
    type_mapper_[i] = i;  // Direct mapping
    sp_magnitude_[i + 1] = 1.0;  // Placeholder (actual values from sp[i][3])
  }

  // Update descriptor config
  descriptor_config_.elements = elements_;

  // Load TorchScript model
  load_model(model_path_);

  // Set cutoff
  cutoff_ = descriptor_config_.rc;

  // Mark all pairs as set
  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoff_ * cutoff_;
    }
  }

  // Print configuration
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
    model_ = torch::jit::load(path, device_);
    model_.eval();

    // Try to freeze the model for better performance
    if (model_.hasattr("training")) {
      model_ = torch::jit::freeze(model_);
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
  // Check for atom IDs
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style spin/nep requires atom IDs");

  // Check unit style
  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "Pair style spin/nep requires metal units");

  // Check atom style for spin
  if (!atom->sp_flag)
    error->all(FLERR, "Pair style spin/nep requires atom_style spin");

  // Request full neighbor list with ghost atoms (like Allegro)
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  // Newton pair must be on for proper force communication
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style spin/nep requires newton pair on");

  // Initialize descriptor
  try {
    descriptor_ = std::make_unique<nep::MagneticACEDescriptor>(descriptor_config_);
  } catch (const std::exception &e) {
    error->all(FLERR, "Failed to initialize NEP descriptor: {}", e.what());
  }

  // Call parent init_style
  PairSpin::init_style();
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
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;

  forces_cached_ = false;

  // Convert LAMMPS data to tensors
  // Note: We need gradient tracking for force computation via autograd

  try {
    // 1. Convert positions (local + ghost for Allegro-like mode)
    auto positions = convert_positions(ntotal);

    // 2. Convert atom types to model element indices
    auto numbers = convert_types(ntotal);

    // 3. Convert spins to magnetic moments
    auto magmoms = convert_spins_to_magmoms(ntotal);

    // 4. Get simulation cell
    auto cell = get_cell_tensor();

    // Enable gradients for force computation
    positions.set_requires_grad(true);
    magmoms.set_requires_grad(true);

    // 5. Build neighbor list from LAMMPS
    auto neighbors = build_neighbor_list_from_lammps(ntotal);

    if (neighbors.n_pairs == 0) {
      // No neighbors, zero energy
      eng_vdwl = 0.0;
      forces_cached_ = true;
      cached_mag_forces_ = torch::zeros({ntotal, 3}, torch::kFloat32);
      return;
    }

    // 6. Compute descriptors (for all atoms including ghosts)
    auto descriptors_all = descriptor_->compute_from_precomputed_neighbors(
        positions, numbers, magmoms, neighbors, cell);

    // 7. Extract only local atoms' descriptors for model inference
    // Ghost atoms are only used for neighbor information, not for energy computation
    auto descriptors = descriptors_all.slice(0, 0, nlocal);

    // 8. Run TorchScript model inference on local atoms only
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(descriptors);
    auto atomic_energies = model_.forward(inputs).toTensor();

    // Sum to get total energy (only local atoms)
    auto total_energy = atomic_energies.sum();

    // 9. Compute gradients via autograd
    // Note: positions and magmoms include ghost atoms, but total_energy only depends
    // on local atoms' descriptors. autograd will correctly propagate gradients through
    // the neighbor contributions from ghost atoms.
    auto grads = torch::autograd::grad(
        {total_energy},
        {positions, magmoms},
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);  // Allow unused for ghost atoms with no contribution

    auto forces_tensor = -grads[0];  // F = -dE/dR
    auto mag_forces_tensor = grads[1];  // dE/dM

    // 10. Distribute forces to LAMMPS
    distribute_forces(forces_tensor, nlocal, nghost);

    // 11. Distribute magnetic forces to atom->fm
    distribute_magnetic_forces(mag_forces_tensor, nlocal);

    // 12. Cache magnetic forces for compute_single_pair (for future use)
    cached_mag_forces_ = mag_forces_tensor.detach().clone();
    forces_cached_ = true;

    // 13. Cache current spin state for detecting changes in compute_single_pair
    cache_current_spins();

    // 14. Energy accounting
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
  // Check if spins have changed since last compute - if so, recompute forces
  if (!forces_cached_ || spins_changed()) {
    recompute_forces();
  }

  if (!forces_cached_) {
    return;
  }

  double **sp = atom->sp;

  // Get spin magnitude for this atom (per-atom, not per-type)
  double mag = sp[ii][3];

  // Skip non-magnetic atoms
  if (mag < 1e-10) {
    return;
  }

  // Get cached magnetic force for atom ii (ensure on CPU for accessor)
  auto mag_forces_cpu = cached_mag_forces_.cpu();
  auto mag_force_accessor = mag_forces_cpu.accessor<float, 2>();

  // Convert dE/dM to LAMMPS fm format
  // NEP-SPIN computes dE/dM where M = |M| * s (s = unit spin vector)
  // The effective field for unit spin dynamics is: H_s = -dE/ds = -|M| * dE/dM
  // Therefore: fm = -|M| * dE/dM / hbar
  // Note: cached_mag_forces_ contains dE/dM (positive gradient from autograd)
  fmi[0] -= mag * static_cast<double>(mag_force_accessor[ii][0]) / hbar;
  fmi[1] -= mag * static_cast<double>(mag_force_accessor[ii][1]) / hbar;
  fmi[2] -= mag * static_cast<double>(mag_force_accessor[ii][2]) / hbar;
}

/* ---------------------------------------------------------------------- */

torch::Tensor PairNEPSpin::convert_positions(int ntotal)
{
  double **x = atom->x;
  // Create on CPU first, then move to target device
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto positions = torch::zeros({ntotal, 3}, options);
  auto pos_accessor = positions.accessor<float, 2>();

  for (int i = 0; i < ntotal; i++) {
    pos_accessor[i][0] = static_cast<float>(x[i][0]);
    pos_accessor[i][1] = static_cast<float>(x[i][1]);
    pos_accessor[i][2] = static_cast<float>(x[i][2]);
  }

  return positions.to(device_);
}

/* ---------------------------------------------------------------------- */

torch::Tensor PairNEPSpin::convert_types(int ntotal)
{
  int *type = atom->type;
  // Create on CPU first, then move to target device
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto numbers = torch::zeros({ntotal}, options);
  auto num_accessor = numbers.accessor<int64_t, 1>();

  for (int i = 0; i < ntotal; i++) {
    // Convert to atomic number based on element
    int lmp_type = type[i];
    std::string elem = elements_[lmp_type - 1];
    num_accessor[i] = nep::element_to_number(elem);
  }

  return numbers.to(device_);
}

/* ---------------------------------------------------------------------- */

torch::Tensor PairNEPSpin::convert_spins_to_magmoms(int ntotal)
{
  double **sp = atom->sp;
  // Create on CPU first, then move to target device
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto magmoms = torch::zeros({ntotal, 3}, options);
  auto mag_accessor = magmoms.accessor<float, 2>();

  for (int i = 0; i < ntotal; i++) {
    // In LAMMPS atom_style spin:
    // sp[i][0], sp[i][1], sp[i][2] = unit spin vector components
    // sp[i][3] = spin magnitude
    // Magnetic moment M = |sp| * (spx, spy, spz)
    double mag = sp[i][3];
    mag_accessor[i][0] = static_cast<float>(sp[i][0] * mag);
    mag_accessor[i][1] = static_cast<float>(sp[i][1] * mag);
    mag_accessor[i][2] = static_cast<float>(sp[i][2] * mag);
  }

  return magmoms.to(device_);
}

/* ---------------------------------------------------------------------- */

torch::Tensor PairNEPSpin::get_cell_tensor()
{
  // Create on CPU first, then move to target device
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto cell = torch::zeros({3, 3}, options);
  auto cell_accessor = cell.accessor<float, 2>();

  // Get cell vectors from domain
  // LAMMPS uses: boxlo, boxhi, xy, xz, yz for triclinic
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

  return cell.to(device_);
}

/* ---------------------------------------------------------------------- */

nep::NeighborList PairNEPSpin::build_neighbor_list_from_lammps(int ntotal)
{
  nep::NeighborList result;
  result.n_pairs = 0;

  double **x = atom->x;
  int *type = atom->type;
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // First pass: count valid pairs
  int total_pairs = 0;
  double cutoff_sq = cutoff_ * cutoff_;

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      double rsq = dx * dx + dy * dy + dz * dz;

      if (rsq <= cutoff_sq) {
        total_pairs++;
      }
    }
  }

  if (total_pairs == 0) {
    result.center_indices = torch::zeros({0}, torch::kInt64).to(device_);
    result.neighbor_indices = torch::zeros({0}, torch::kInt64).to(device_);
    result.shifts = torch::zeros({0, 3}, torch::kFloat32).to(device_);
    result.batch_idx = torch::zeros({ntotal}, torch::kInt64).to(device_);
    return result;
  }

  // Allocate tensors on CPU first for filling with accessor
  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  result.center_indices = torch::zeros({total_pairs}, options_int);
  result.neighbor_indices = torch::zeros({total_pairs}, options_int);
  result.shifts = torch::zeros({total_pairs, 3}, options_float);
  result.batch_idx = torch::zeros({ntotal}, options_int);

  auto center_accessor = result.center_indices.accessor<int64_t, 1>();
  auto neighbor_accessor = result.neighbor_indices.accessor<int64_t, 1>();
  auto shifts_accessor = result.shifts.accessor<float, 2>();

  // Second pass: fill tensors
  int pair_idx = 0;
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      double rsq = dx * dx + dy * dy + dz * dz;

      if (rsq <= cutoff_ * cutoff_) {
        center_accessor[pair_idx] = i;
        neighbor_accessor[pair_idx] = j;
        // For LAMMPS ghost atoms, positions are already unwrapped
        // So cell shifts are zero
        shifts_accessor[pair_idx][0] = 0.0f;
        shifts_accessor[pair_idx][1] = 0.0f;
        shifts_accessor[pair_idx][2] = 0.0f;
        pair_idx++;
      }
    }
  }

  // Move tensors to target device
  result.center_indices = result.center_indices.to(device_);
  result.neighbor_indices = result.neighbor_indices.to(device_);
  result.shifts = result.shifts.to(device_);
  result.batch_idx = result.batch_idx.to(device_);

  result.n_pairs = total_pairs;
  return result;
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::distribute_forces(const torch::Tensor &forces, int nlocal, int nghost)
{
  double **f = atom->f;
  tagint *tag = atom->tag;
  int ntotal = nlocal + nghost;

  auto forces_cpu = forces.cpu();
  auto force_accessor = forces_cpu.accessor<float, 2>();

  // Add forces to local atoms directly
  for (int i = 0; i < nlocal; i++) {
    f[i][0] += static_cast<double>(force_accessor[i][0]);
    f[i][1] += static_cast<double>(force_accessor[i][1]);
    f[i][2] += static_cast<double>(force_accessor[i][2]);
  }

  // For ghost atoms, map their forces to corresponding local atoms
  // Ghost atoms are periodic images, so their gradients should be accumulated
  // to the original local atom
  for (int j = nlocal; j < ntotal; j++) {
    // Get the local atom that this ghost corresponds to
    int local_i = atom->map(tag[j]);
    if (local_i >= 0 && local_i < nlocal) {
      f[local_i][0] += static_cast<double>(force_accessor[j][0]);
      f[local_i][1] += static_cast<double>(force_accessor[j][1]);
      f[local_i][2] += static_cast<double>(force_accessor[j][2]);
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::distribute_magnetic_forces(const torch::Tensor &mag_forces, int nlocal)
{
  double **fm = atom->fm;
  double **sp = atom->sp;

  auto mag_forces_cpu = mag_forces.cpu();
  auto mag_accessor = mag_forces_cpu.accessor<float, 2>();

  // Convert dE/dM to LAMMPS fm format
  // In LAMMPS spin: fm = effective_field / hbar
  // NEP-SPIN computes dE/dM where M = |M| * s (s = unit spin vector)
  // The effective field for unit spin dynamics is: H_s = -dE/ds = -|M| * dE/dM
  // Therefore: fm = -|M| * dE/dM / hbar
  // Note: mag_accessor contains dE/dM (positive gradient from autograd)

  for (int i = 0; i < nlocal; i++) {
    double mag = sp[i][3];  // spin magnitude |M|
    if (mag > 1e-10) {
      // fm = -|M| * dE/dM / hbar
      fm[i][0] -= mag * static_cast<double>(mag_accessor[i][0]) / hbar;
      fm[i][1] -= mag * static_cast<double>(mag_accessor[i][1]) / hbar;
      fm[i][2] -= mag * static_cast<double>(mag_accessor[i][2]) / hbar;
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::cache_current_spins()
{
  double **sp = atom->sp;
  int nlocal = atom->nlocal;

  cached_spins_.resize(nlocal * 4);
  for (int i = 0; i < nlocal; i++) {
    cached_spins_[i*4 + 0] = sp[i][0];
    cached_spins_[i*4 + 1] = sp[i][1];
    cached_spins_[i*4 + 2] = sp[i][2];
    cached_spins_[i*4 + 3] = sp[i][3];
  }
}

/* ---------------------------------------------------------------------- */

bool PairNEPSpin::spins_changed()
{
  double **sp = atom->sp;
  int nlocal = atom->nlocal;

  // If cache size doesn't match, spins definitely changed
  if (cached_spins_.size() != static_cast<size_t>(nlocal * 4)) {
    return true;
  }

  // Check if any local spin component has changed significantly
  const double tol = 0.001;
  for (int i = 0; i < nlocal; i++) {
    if (std::abs(sp[i][0] - cached_spins_[i*4 + 0]) > tol ||
        std::abs(sp[i][1] - cached_spins_[i*4 + 1]) > tol ||
        std::abs(sp[i][2] - cached_spins_[i*4 + 2]) > tol ||
        std::abs(sp[i][3] - cached_spins_[i*4 + 3]) > tol) {
      return true;
    }
  }
  return false;
}

/* ---------------------------------------------------------------------- */

void PairNEPSpin::recompute_forces()
{
  // Recompute magnetic forces with current spin configuration
  // This is called when compute_single_pair detects spin changes

  if (!model_loaded_) {
    return;
  }

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;

  try {
    // 1. Convert current positions
    auto positions = convert_positions(ntotal);

    // 2. Convert atom types
    auto numbers = convert_types(ntotal);

    // 3. Convert current spins to magnetic moments
    auto magmoms = convert_spins_to_magmoms(ntotal);

    // 4. Get simulation cell
    auto cell = get_cell_tensor();

    // Enable gradients for magnetic force computation
    positions.set_requires_grad(false);  // Don't need position gradients
    magmoms.set_requires_grad(true);

    // 5. Build neighbor list
    auto neighbors = build_neighbor_list_from_lammps(ntotal);

    if (neighbors.n_pairs == 0) {
      cached_mag_forces_ = torch::zeros({ntotal, 3}, torch::kFloat32);
      forces_cached_ = true;
      cache_current_spins();
      return;
    }

    // 6. Compute descriptors
    auto descriptors_all = descriptor_->compute_from_precomputed_neighbors(
        positions, numbers, magmoms, neighbors, cell);

    // 7. Extract local atoms' descriptors
    auto descriptors = descriptors_all.slice(0, 0, nlocal);

    // 8. Run model inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(descriptors);
    auto atomic_energies = model_.forward(inputs).toTensor();

    // Sum to get total energy
    auto total_energy = atomic_energies.sum();

    // 9. Compute magnetic force gradients only
    auto grads = torch::autograd::grad(
        {total_energy},
        {magmoms},
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    auto mag_forces_tensor = grads[0];  // dE/dM

    // 10. Cache the new magnetic forces
    cached_mag_forces_ = mag_forces_tensor.detach().clone();
    forces_cached_ = true;

    // 11. Update cached spin state
    cache_current_spins();

  } catch (const c10::Error &e) {
    error->warning(FLERR, "PyTorch error in NEP-SPIN recompute_forces: {}", e.what());
    forces_cached_ = false;
  } catch (const std::exception &e) {
    error->warning(FLERR, "Error in NEP-SPIN recompute_forces: {}", e.what());
    forces_cached_ = false;
  }
}
