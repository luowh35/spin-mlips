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
   Contributing author: NEP-SPIN Kokkos GPU acceleration
   Based on pair_nequip_allegro_kokkos by Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include "pair_nep_spin_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair_kokkos.h"

#include <torch/script.h>
#include <torch/torch.h>

#ifdef KOKKOS_ENABLE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace LAMMPS_NS;
using namespace MathConst;

// Kokkos reduction identity for FEV_FLOAT
namespace Kokkos {
template <>
struct reduction_identity<s_FEV_FLOAT> {
  KOKKOS_FORCEINLINE_FUNCTION static s_FEV_FLOAT sum() { return s_FEV_FLOAT(); }
};
}  // namespace Kokkos

/* ---------------------------------------------------------------------- */

PairNEPSpinKokkos::PairNEPSpinKokkos(LAMMPS *lmp) : PairNEPSpin(lmp)
{
  respa_enable = 0;

  // Get Kokkos atom pointers
  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  // Data masks for sync operations
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  device_forces_valid_ = false;
}

/* ---------------------------------------------------------------------- */

PairNEPSpinKokkos::~PairNEPSpinKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->destroy_kokkos(k_vatom, vatom);
    eatom = nullptr;
    vatom = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpinKokkos::compute(int eflag_in, int vflag_in)
{
  eflag_kk = eflag_in;
  vflag_kk = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag_kk, vflag_kk, 0);

  if (!model_loaded_) {
    error->all(FLERR, "NEP-SPIN model not loaded");
  }

  // Reallocate per-atom arrays if necessary
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  // Sync atom data to device
  atomKK->sync(execution_space, datamask_read);
  if (eflag_kk || vflag_kk)
    atomKK->modified(execution_space, datamask_modify);
  else
    atomKK->modified(execution_space, F_MASK);

  // Get device views
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  tag = atomKK->k_tag.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();

  // Sync and get spin views
  atomKK->sync(execution_space, SP_MASK | FM_MASK);
  atomKK->modified(execution_space, FM_MASK);

  sp = atomKK->k_sp.template view<DeviceType>();
  fm = atomKK->k_fm.template view<DeviceType>();

  nlocal_kk = atom->nlocal;
  newton_pair = force->newton_pair;
  nall_kk = atom->nlocal + atom->nghost;

  const int inum = list->inum;
  const int ignum = inum + atom->nghost;

  // Get Kokkos neighbor list
  NeighListKokkos<DeviceType> *k_list =
      static_cast<NeighListKokkos<DeviceType> *>(list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  if (inum == 0) return;  // Empty domain

  copymode = 1;

  // Step 1: Count total edges from LAMMPS neighbor list (no filtering needed)
  int nedges = count_edges_kokkos(inum);

  if (nedges == 0) {
    eng_vdwl = 0.0;
    copymode = 0;
    return;
  }

  // Step 2: Convert data to tensors using Kokkos parallel operations
  convert_data_to_tensors_kokkos(inum, ignum, nedges);

  // Step 3: Create PyTorch tensors from Kokkos views (zero-copy)
  int max_atoms = d_ij2type.extent(0);
  int max_edges = d_edges.extent(1);

  torch::Tensor ij2type_tensor = torch::from_blob(
      d_ij2type.data(), {max_atoms},
      torch::TensorOptions().dtype(torch::kInt64).device(device_));

  torch::Tensor pos_tensor = torch::from_blob(
      d_xfloat.data(), {max_atoms, 3}, {3, 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  torch::Tensor magmoms_tensor = torch::from_blob(
      d_magmoms.data(), {max_atoms, 3}, {3, 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  torch::Tensor edges_tensor = torch::from_blob(
      d_edges.data(), {2, max_edges}, {(long)d_edges.extent(1), 1},
      torch::TensorOptions().dtype(torch::kInt64).device(device_));

  torch::Tensor shifts_tensor = torch::from_blob(
      d_shifts.data(), {max_edges, 3}, {3, 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  // Enable gradients for force computation
  pos_tensor.set_requires_grad(true);
  magmoms_tensor.set_requires_grad(true);

  try {
    // Step 4: Build neighbor list structure for descriptor
    nep::NeighborList neighbors;
    neighbors.center_indices = edges_tensor.select(0, 0).slice(0, 0, nedges);
    neighbors.neighbor_indices = edges_tensor.select(0, 1).slice(0, 0, nedges);
    neighbors.shifts = shifts_tensor.slice(0, 0, nedges);
    neighbors.batch_idx = torch::zeros({max_atoms}, torch::kInt64).to(device_);
    neighbors.n_pairs = nedges;

    // Get cell tensor
    auto cell = get_cell_tensor();

    // Step 5: Compute descriptors (for all atoms)
    auto descriptors_all = descriptor_->compute_from_precomputed_neighbors(
        pos_tensor.slice(0, 0, ignum), ij2type_tensor.slice(0, 0, ignum),
        magmoms_tensor.slice(0, 0, ignum), neighbors, cell);

    // Extract only local atoms' descriptors
    auto descriptors = descriptors_all.slice(0, 0, inum);

    // Step 6: Run TorchScript model inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(descriptors);
    auto atomic_energies = model_.forward(inputs).toTensor();

    // Sum to get total energy
    auto total_energy = atomic_energies.sum();

    // Step 7: Compute gradients via autograd
    auto grads = torch::autograd::grad(
        {total_energy},
        {pos_tensor, magmoms_tensor},
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    auto forces_tensor = -grads[0];      // F = -dE/dR
    auto mag_forces_tensor = grads[1];   // dE/dM

    // Step 8: Distribute forces using Kokkos parallel
    distribute_forces_kokkos(forces_tensor, mag_forces_tensor, inum, ignum);

    // Step 9: Energy accounting
    eng_vdwl = total_energy.item<double>();

    if (eflag_atom) {
      auto atomic_e = atomic_energies.cpu();
      auto e_accessor = atomic_e.accessor<float, 2>();

      // Copy atomic energies to Kokkos view
      auto h_eatom = Kokkos::create_mirror_view(d_eatom);
      for (int i = 0; i < inum; i++) {
        h_eatom(i) = e_accessor[i][0];
      }
      Kokkos::deep_copy(d_eatom, h_eatom);

      k_eatom.template modify<DeviceType>();
      k_eatom.template sync<LMPHostType>();
    }

  } catch (const c10::Error &e) {
    error->all(FLERR, "PyTorch error in NEP-SPIN/kk compute: {}", e.what());
  } catch (const std::exception &e) {
    error->all(FLERR, "Error in NEP-SPIN/kk compute: {}", e.what());
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

/* ---------------------------------------------------------------------- */

int PairNEPSpinKokkos::count_edges_kokkos(int inum)
{
  // Count total number of edges directly from LAMMPS neighbor list
  // No distance filtering - LAMMPS already uses our cutoff from init_one()

  auto d_ilist_local = d_ilist;
  auto d_numneigh_local = d_numneigh;

  int total_edges = 0;
  Kokkos::parallel_reduce(
      "NEP-SPIN: count edges", Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii, int &sum) {
        const int i = d_ilist_local[ii];
        sum += d_numneigh_local[i];
      },
      total_edges);

  return total_edges;
}

/* ---------------------------------------------------------------------- */

void PairNEPSpinKokkos::build_cumsum_kokkos(int inum)
{
  // Build cumulative sum of neighbor counts for edge indexing

  if (d_cumsum_numneigh.extent(0) < (size_t)inum) {
    d_cumsum_numneigh = Kokkos::View<int *, DeviceType>(
        Kokkos::ViewAllocateWithoutInitializing("NEP-SPIN::cumsum_numneigh"), inum);
  }

  auto d_ilist_local = d_ilist;
  auto d_numneigh_local = d_numneigh;
  auto d_cumsum_local = d_cumsum_numneigh;

  // First copy neighbor counts
  Kokkos::parallel_for(
      "NEP-SPIN: copy numneigh", Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist_local[ii];
        d_cumsum_local(ii) = d_numneigh_local[i];
      });

  // Parallel scan for cumulative sum
  Kokkos::parallel_scan(
      "NEP-SPIN: cumsum", Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii, int &update, const bool is_final) {
        const int curr_val = d_cumsum_local(ii);
        update += curr_val;
        if (is_final) d_cumsum_local(ii) = update;
      });
}

/* ---------------------------------------------------------------------- */

void PairNEPSpinKokkos::convert_data_to_tensors_kokkos(int inum, int ignum, int nedges)
{
  double padding_factor = 1.05;

  // Build cumulative sum for edge indexing
  build_cumsum_kokkos(inum);

  // Allocate/reallocate edge arrays with padding
  if (d_edges.extent(1) < (size_t)nedges ||
      nedges * padding_factor * padding_factor < d_edges.extent(1)) {
    d_edges = decltype(d_edges)();
    d_edges = LongView2D("NEP-SPIN: edges", 2, padding_factor * nedges);
    d_shifts = decltype(d_shifts)();
    d_shifts = FloatView2D("NEP-SPIN: shifts", padding_factor * nedges, 3);
  }

  // Allocate/reallocate atom arrays with padding
  if (d_ij2type.extent(0) < (size_t)(ignum + 2) ||
      (ignum + 2) * padding_factor * padding_factor < d_ij2type.extent(0)) {
    d_ij2type = decltype(d_ij2type)();
    d_ij2type = LongView1D("NEP-SPIN: ij2type", padding_factor * ignum + 2);
    d_xfloat = decltype(d_xfloat)();
    d_xfloat = FloatView2D("NEP-SPIN: xfloat", padding_factor * ignum + 2, 3);
    d_magmoms = decltype(d_magmoms)();
    d_magmoms = FloatView2D("NEP-SPIN: magmoms", padding_factor * ignum + 2, 3);
  }

  // Local copies for lambda capture
  auto d_edges_local = d_edges;
  auto d_shifts_local = d_shifts;
  auto d_ij2type_local = d_ij2type;
  auto d_xfloat_local = d_xfloat;
  auto d_magmoms_local = d_magmoms;
  auto d_type_mapper_local = d_type_mapper;
  auto d_ilist_local = d_ilist;
  auto d_numneigh_local = d_numneigh;
  auto d_neighbors_local = d_neighbors;
  auto d_cumsum_local = d_cumsum_numneigh;
  auto x_local = x;
  auto sp_local = sp;
  auto type_local = type;

  int max_atoms = d_ij2type.extent(0);
  int max_edges = d_edges.extent(1);

  // Parallel for: store atom types, positions, and magnetic moments
  Kokkos::parallel_for(
      "NEP-SPIN: store atoms", Kokkos::RangePolicy<DeviceType>(0, ignum),
      KOKKOS_LAMBDA(const int i) {
        // Map LAMMPS type to model element index
        d_ij2type_local(i) = d_type_mapper_local(type_local(i) - 1);

        // Store positions as float32
        d_xfloat_local(i, 0) = x_local(i, 0);
        d_xfloat_local(i, 1) = x_local(i, 1);
        d_xfloat_local(i, 2) = x_local(i, 2);

        // Store magnetic moments: M = |sp| * (spx, spy, spz)
        const KK_FLOAT mag = sp_local(i, 3);
        d_magmoms_local(i, 0) = sp_local(i, 0) * mag;
        d_magmoms_local(i, 1) = sp_local(i, 1) * mag;
        d_magmoms_local(i, 2) = sp_local(i, 2) * mag;
      });

  // Parallel for: add fake/padding atoms
  Kokkos::parallel_for(
      "NEP-SPIN: store fake atoms",
      Kokkos::RangePolicy<DeviceType>(ignum, max_atoms),
      KOKKOS_LAMBDA(const int i) {
        d_ij2type_local(i) = d_type_mapper_local(0);
        d_xfloat_local(i, 0) = (i == max_atoms - 1) ? 100.0f : 0.0f;
        d_xfloat_local(i, 1) = 0.0f;
        d_xfloat_local(i, 2) = 0.0f;
        d_magmoms_local(i, 0) = 0.0f;
        d_magmoms_local(i, 1) = 0.0f;
        d_magmoms_local(i, 2) = 0.0f;
      });

  // Parallel for with TeamPolicy: create edges directly from LAMMPS neighbor list
  Kokkos::parallel_for(
      "NEP-SPIN: create edges",
      Kokkos::TeamPolicy<DeviceType>(inum, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const MemberType team_member) {
        const int ii = team_member.league_rank();
        const int i = d_ilist_local(ii);
        const int jnum = d_numneigh_local[i];
        const int startedge = (ii == 0) ? 0 : d_cumsum_local(ii - 1);

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, jnum),
            [&](const int jj) {
              const int j = d_neighbors_local(i, jj) & NEIGHMASK;
              d_edges_local(0, startedge + jj) = i;
              d_edges_local(1, startedge + jj) = j;
              // Shifts are zero for LAMMPS (ghost atoms have unwrapped positions)
              d_shifts_local(startedge + jj, 0) = 0.0f;
              d_shifts_local(startedge + jj, 1) = 0.0f;
              d_shifts_local(startedge + jj, 2) = 0.0f;
            });
      });

  // Parallel for: add fake edges for padding
  Kokkos::parallel_for(
      "NEP-SPIN: store fake edges",
      Kokkos::RangePolicy<DeviceType>(nedges, max_edges),
      KOKKOS_LAMBDA(const int i) {
        d_edges_local(0, i) = max_atoms - 2;
        d_edges_local(1, i) = max_atoms - 1;
        d_shifts_local(i, 0) = 0.0f;
        d_shifts_local(i, 1) = 0.0f;
        d_shifts_local(i, 2) = 0.0f;
      });
}

/* ---------------------------------------------------------------------- */

void PairNEPSpinKokkos::distribute_forces_kokkos(const torch::Tensor &forces,
                                                   const torch::Tensor &mag_forces,
                                                   int inum, int ignum)
{
  // Get raw pointers from PyTorch tensors (zero-copy)
  UnmanagedFloatView2D d_forces(forces.data_ptr<float>(), ignum, 3);
  UnmanagedFloatView2D d_mag_forces(mag_forces.data_ptr<float>(), ignum, 3);

  // Local copies for lambda capture
  auto f_local = f;
  auto fm_local = fm;
  auto sp_local = sp;
  double hbar_val = hbar;

  // Parallel reduce: distribute forces and accumulate energy
  Kokkos::parallel_reduce(
      "NEP-SPIN: store forces", Kokkos::RangePolicy<DeviceType>(0, ignum),
      KOKKOS_LAMBDA(const int i, double &eng_sum) {
        // Add position forces
        f_local(i, 0) += d_forces(i, 0);
        f_local(i, 1) += d_forces(i, 1);
        f_local(i, 2) += d_forces(i, 2);

        // Add magnetic forces for local atoms only
        if (i < inum) {
          const KK_FLOAT mag = sp_local(i, 3);
          if (mag > 1e-10) {
            // fm = -|M| * dE/dM / hbar
            fm_local(i, 0) -= mag * d_mag_forces(i, 0) / hbar_val;
            fm_local(i, 1) -= mag * d_mag_forces(i, 1) / hbar_val;
            fm_local(i, 2) -= mag * d_mag_forces(i, 2) / hbar_val;
          }
        }
      },
      eng_vdwl);
}

/* ---------------------------------------------------------------------- */

void PairNEPSpinKokkos::coeff(int narg, char **arg)
{
  // Call parent coeff to load model and set up elements
  super::coeff(narg, arg);

  int ntypes = atom->ntypes;

  // Allocate and copy type mapper to device
  d_type_mapper = IntView1D("NEP-SPIN: type_mapper", type_mapper_.size());
  auto h_type_mapper = Kokkos::create_mirror_view(d_type_mapper);
  for (size_t i = 0; i < type_mapper_.size(); i++) {
    h_type_mapper(i) = type_mapper_[i];
  }
  Kokkos::deep_copy(d_type_mapper, h_type_mapper);

  if (comm->me == 0) {
    utils::logmesg(lmp, "NEP-SPIN/kk: Type mapper copied to device\n");
  }
}

/* ---------------------------------------------------------------------- */

void PairNEPSpinKokkos::init_style()
{
  // Call parent init_style
  super::init_style();

  // Configure Kokkos neighbor list
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType, LMPHostType>::value &&
                           !std::is_same<DeviceType, LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);

  neighflag = lmp->kokkos->neighflag;

  // NEP-SPIN requires newton pair on for proper force communication
  if (!force->newton_pair) {
    error->all(FLERR, "Pair style spin/nep/kk requires newton pair on");
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "NEP-SPIN/kk: Kokkos neighbor list initialized\n");
  }
}

/* ---------------------------------------------------------------------- */

// Explicit instantiation is not needed since PairNEPSpinKokkos is not a template class
