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
   Contributing author: NEP-SPIN Kokkos GPU acceleration
   Based on pair_nequip_allegro_kokkos by Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(spin/nep/kk, PairNEPSpinKokkos)
// clang-format on
#else

#ifndef LMP_PAIR_NEP_SPIN_KOKKOS_H
#define LMP_PAIR_NEP_SPIN_KOKKOS_H

#include "pair_nep_spin.h"
#include <pair_kokkos.h>
#include <kokkos_type.h>

namespace LAMMPS_NS {

class PairNEPSpinKokkos : public PairNEPSpin {
 public:
  typedef PairNEPSpin super;
  typedef LMPDeviceType DeviceType;
  using MemberType = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  enum {EnabledNeighFlags = FULL | HALFTHREAD | HALF};
  enum {COUL_FLAG = 0};
  typedef LMPDeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairNEPSpinKokkos(class LAMMPS *);
  ~PairNEPSpinKokkos() override;

  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;

  // Per-atom energy/virial arrays for Kokkos
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

 protected:
  // Kokkos DualView types for neighbor list data
  typedef Kokkos::DualView<int***, DeviceType> tdual_int_3d;
  typedef typename tdual_int_3d::t_dev_const_randomread t_int_3d_randomread;
  typedef typename tdual_int_3d::t_host t_host_int_3d;

  // Atom arrays - device views
  typename AT::t_kkfloat_1d_3_lr_randomread x;  // positions
  typename AT::t_kkacc_1d_3 f;                   // forces
  typename AT::t_tagint_1d tag;                  // atom tags
  typename AT::t_int_1d_randomread type;         // atom types

  // Spin arrays - device views (added for NEP-SPIN)
  // Note: sp is [sx, sy, sz, mag] stored as kkfloat_1d_4
  typename AT::tdual_kkfloat_1d_4::t_dev_const_randomread sp;
  typename AT::t_kkacc_1d_3 fm;                   // magnetic forces

  // DualView wrappers for per-atom arrays
  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;

  // Type definitions for Kokkos views
  using inputtype = float;   // Input type for neural network (float32)
  using outputtype = float;  // Output type from neural network (float32)

  using IntView1D = Kokkos::View<int*, Kokkos::LayoutRight, DeviceType>;
  using IntView2D = Kokkos::View<int**, Kokkos::LayoutRight, DeviceType>;
  using LongView1D = Kokkos::View<long*, Kokkos::LayoutRight, DeviceType>;
  using LongView2D = Kokkos::View<long**, Kokkos::LayoutRight, DeviceType>;
  using FloatView1D = Kokkos::View<float*, Kokkos::LayoutRight, DeviceType>;
  using FloatView2D = Kokkos::View<float**, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView1D = Kokkos::View<outputtype*, Kokkos::LayoutRight, DeviceType,
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UnmanagedFloatView2D = Kokkos::View<outputtype**, Kokkos::LayoutRight, DeviceType,
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using View1D = Kokkos::View<KK_ACC_FLOAT*, Kokkos::LayoutRight, DeviceType>;
  using View2D = Kokkos::View<KK_ACC_FLOAT**, Kokkos::LayoutRight, DeviceType>;

  // Device views for type mapping and cutoff
  IntView1D d_type_mapper;     // LAMMPS type -> model element index
  LongView1D d_ij2type;        // Per-atom type for PyTorch (int64)
  FloatView2D d_xfloat;        // Positions as float32 for PyTorch
  FloatView2D d_magmoms;       // Magnetic moments as float32 for PyTorch [N, 3]
  LongView2D d_edges;          // Edge list [2, M]
  FloatView2D d_shifts;        // Cell shifts for edges [M, 3]

  // Neighbor list handling
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  // Cumulative sum of neighbor counts for edge indexing
  Kokkos::View<int*, DeviceType> d_cumsum_numneigh;

  // State flags
  int neighflag, newton_pair;
  int nlocal_kk, nall_kk, eflag_kk, vflag_kk;

  // Cached forces on device
  FloatView2D d_forces_cached;
  FloatView2D d_mag_forces_cached;
  bool device_forces_valid_;

  // Helper methods for Kokkos parallel operations
  int count_edges_kokkos(int inum);
  void build_cumsum_kokkos(int inum);
  void convert_data_to_tensors_kokkos(int inum, int ignum, int nedges);
  void distribute_forces_kokkos(const torch::Tensor& forces,
                                 const torch::Tensor& mag_forces,
                                 int inum, int ignum);

  friend void pair_virial_fdotr_compute<PairNEPSpinKokkos>(PairNEPSpinKokkos*);
};

}  // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use chosen neighbor list style with pair spin/nep/kk

Self-explanatory.

E: Pair style spin/nep/kk requires newton pair on

Self-explanatory.

*/
