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
   NEP-SPIN Data Management Implementation
   Optimized neighbor list and data structure for NEP-SPIN potential
------------------------------------------------------------------------- */

#include "nep_spin_data.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "neigh_list.h"

#include <cstring>
#include <map>

using namespace LAMMPS_NS;

// Static element map (up to element 86, Rn)
static const std::map<std::string, int> ELEMENT_MAP = {
    {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},
    {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
    {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},
    {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20},
    {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25},
    {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
    {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
    {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40},
    {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45},
    {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50},
    {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55},
    {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60},
    {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65},
    {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70},
    {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75},
    {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80},
    {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85},
    {"Rn", 86}
};

/* ---------------------------------------------------------------------- */

NEPSpinData::NEPSpinData(LAMMPS *lmp, double cutoff,
                         const std::vector<std::string> &elements) :
    Pointers(lmp),
    nlocal(0), ntotal(0), npairs(0), eflag(0), vflag(0),
    cached_mag_forces_ptr(nullptr), cached_mag_forces_size(0),
    forces_cached(false),
    cutoff_(cutoff), cutoff_sq_(cutoff * cutoff),
    elements_(elements),
    nmax_(0), npairs_max_(0),
    positions_(nullptr), magmoms_(nullptr), atom_types_(nullptr),
    center_idx_(nullptr), neighbor_idx_(nullptr), rij_(nullptr),
    numneighs_(nullptr),
    tensors_valid_(false)
{
  // Build type map: element string -> atomic number
  type_map_.resize(elements_.size());
  for (size_t i = 0; i < elements_.size(); ++i) {
    type_map_[i] = element_to_number(elements_[i]);
  }

  // Initialize cell to zeros
  std::memset(cell_, 0, 9 * sizeof(double));
}

/* ---------------------------------------------------------------------- */

NEPSpinData::~NEPSpinData()
{
  memory->destroy(positions_);
  memory->destroy(magmoms_);
  memory->destroy(atom_types_);
  memory->destroy(center_idx_);
  memory->destroy(neighbor_idx_);
  memory->destroy(rij_);
  memory->destroy(numneighs_);
}

/* ---------------------------------------------------------------------- */

int NEPSpinData::element_to_number(const std::string &elem)
{
  auto it = ELEMENT_MAP.find(elem);
  if (it != ELEMENT_MAP.end()) {
    return it->second;
  }
  error->all(FLERR, "NEP-SPIN: Unknown element {}", elem);
  return 0;
}

/* ---------------------------------------------------------------------- */

void NEPSpinData::grow_arrays(int ntotal_new, int npairs_new)
{
  // Grow atom arrays if needed
  if (ntotal_new > nmax_) {
    nmax_ = ntotal_new + 100;  // Add buffer to avoid frequent reallocation
    memory->grow(positions_, nmax_ * 3, "NEPSpinData:positions");
    memory->grow(magmoms_, nmax_ * 3, "NEPSpinData:magmoms");
    memory->grow(atom_types_, nmax_, "NEPSpinData:atom_types");
    memory->grow(numneighs_, nmax_, "NEPSpinData:numneighs");
  }

  // Grow pair arrays if needed
  if (npairs_new > npairs_max_) {
    npairs_max_ = npairs_new + 500;  // Add buffer
    memory->grow(center_idx_, npairs_max_, "NEPSpinData:center_idx");
    memory->grow(neighbor_idx_, npairs_max_, "NEPSpinData:neighbor_idx");
    memory->grow(rij_, npairs_max_ * 3, "NEPSpinData:rij");
  }
}

/* ---------------------------------------------------------------------- */

void NEPSpinData::generate_neighdata(NeighList *list, int eflag_in, int vflag_in)
{
  double **x = atom->x;
  double **sp = atom->sp;
  int *type = atom->type;

  int *ilist = list->ilist;
  int *numneigh_list = list->numneigh;
  int **firstneigh = list->firstneigh;

  nlocal = atom->nlocal;
  ntotal = nlocal + atom->nghost;
  eflag = eflag_in;
  vflag = vflag_in;

  // First pass: count pairs within cutoff
  int npairs_estimate = 0;
  for (int ii = 0; ii < list->inum; ii++) {
    int i = ilist[ii];
    int *jlist = firstneigh[i];
    int jnum = numneigh_list[i];

    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double delx = x[j][0] - xtmp;
      double dely = x[j][1] - ytmp;
      double delz = x[j][2] - ztmp;
      double rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutoff_sq_) {
        npairs_estimate++;
      }
    }
  }

  // Grow arrays if needed
  grow_arrays(ntotal, npairs_estimate);

  // Second pass: fill data arrays
  npairs = 0;
  for (int ii = 0; ii < list->inum; ii++) {
    int i = ilist[ii];
    int *jlist = firstneigh[i];
    int jnum = numneigh_list[i];

    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];

    int ninside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double delx = x[j][0] - xtmp;
      double dely = x[j][1] - ytmp;
      double delz = x[j][2] - ztmp;
      double rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutoff_sq_) {
        center_idx_[npairs] = i;
        neighbor_idx_[npairs] = j;
        rij_[npairs * 3 + 0] = delx;
        rij_[npairs * 3 + 1] = dely;
        rij_[npairs * 3 + 2] = delz;
        npairs++;
        ninside++;
      }
    }
    numneighs_[i] = ninside;
  }

  // Copy atom data (positions, types, magmoms)
  for (int i = 0; i < ntotal; i++) {
    positions_[i * 3 + 0] = x[i][0];
    positions_[i * 3 + 1] = x[i][1];
    positions_[i * 3 + 2] = x[i][2];

    // Convert LAMMPS type (1-based) to atomic number
    int lmp_type = type[i];
    atom_types_[i] = type_map_[lmp_type - 1];

    // Convert spin direction to magnetic moment
    double mag = sp[i][3];
    magmoms_[i * 3 + 0] = sp[i][0] * mag;
    magmoms_[i * 3 + 1] = sp[i][1] * mag;
    magmoms_[i * 3 + 2] = sp[i][2] * mag;
  }

  // Copy cell data
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;

  cell_[0] = boxhi[0] - boxlo[0];  // a_x
  cell_[1] = 0.0;                   // a_y
  cell_[2] = 0.0;                   // a_z

  if (domain->triclinic) {
    cell_[3] = domain->xy;          // b_x
    cell_[4] = boxhi[1] - boxlo[1]; // b_y
    cell_[5] = 0.0;                  // b_z
    cell_[6] = domain->xz;          // c_x
    cell_[7] = domain->yz;          // c_y
    cell_[8] = boxhi[2] - boxlo[2]; // c_z
  } else {
    cell_[3] = 0.0;
    cell_[4] = boxhi[1] - boxlo[1];
    cell_[5] = 0.0;
    cell_[6] = 0.0;
    cell_[7] = 0.0;
    cell_[8] = boxhi[2] - boxlo[2];
  }

  // Invalidate tensors (need to be regenerated)
  tensors_valid_ = false;
  forces_cached = false;
}

/* ---------------------------------------------------------------------- */

void NEPSpinData::to_tensors(torch::Device device)
{
  if (tensors_valid_) return;

  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto cpu_int64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

  // Create tensors from raw arrays using from_blob (zero-copy when possible)
  // Note: from_blob does not take ownership, data must remain valid

  // Positions: [ntotal, 3]
  positions_tensor_ = torch::from_blob(
      positions_, {ntotal, 3}, torch::kFloat64
  ).to(torch::kFloat32).to(device);

  // Magnetic moments: [ntotal, 3]
  magmoms_tensor_ = torch::from_blob(
      magmoms_, {ntotal, 3}, torch::kFloat64
  ).to(torch::kFloat32).to(device);

  // Atomic numbers: [ntotal]
  // Need to convert int* to int64_t tensor
  auto numbers_cpu = torch::zeros({ntotal}, cpu_int64);
  auto num_accessor = numbers_cpu.accessor<int64_t, 1>();
  for (int i = 0; i < ntotal; i++) {
    num_accessor[i] = atom_types_[i];
  }
  numbers_tensor_ = numbers_cpu.to(device);

  // Cell: [3, 3]
  cell_tensor_ = torch::from_blob(
      cell_, {3, 3}, torch::kFloat64
  ).to(torch::kFloat32).to(device);

  // Neighbor indices
  if (npairs > 0) {
    // Center indices: [npairs]
    auto center_cpu = torch::zeros({npairs}, cpu_int64);
    auto center_accessor = center_cpu.accessor<int64_t, 1>();
    for (int i = 0; i < npairs; i++) {
      center_accessor[i] = center_idx_[i];
    }
    center_indices_ = center_cpu.to(device);

    // Neighbor indices: [npairs]
    auto neighbor_cpu = torch::zeros({npairs}, cpu_int64);
    auto neighbor_accessor = neighbor_cpu.accessor<int64_t, 1>();
    for (int i = 0; i < npairs; i++) {
      neighbor_accessor[i] = neighbor_idx_[i];
    }
    neighbor_indices_ = neighbor_cpu.to(device);

    // Shifts: [npairs, 3] - zeros for now (no PBC shifts stored)
    shifts_tensor_ = torch::zeros({npairs, 3}, cpu_float).to(device);
  } else {
    center_indices_ = torch::zeros({0}, cpu_int64).to(device);
    neighbor_indices_ = torch::zeros({0}, cpu_int64).to(device);
    shifts_tensor_ = torch::zeros({0, 3}, cpu_float).to(device);
  }

  // Batch index: [ntotal] - all zeros for single frame
  batch_idx_ = torch::zeros({ntotal}, cpu_int64).to(device);

  tensors_valid_ = true;
}
