#!/bin/bash

# Install/unInstall package files in LAMMPS src dir
# Note: fix_nve_spin_sib and fix_langevin_spin_sib are now in USER-SPIN-ML package

if (test $1 = 1) then

  # Install: copy files to src directory

  # Pair style
  cp pair_nep_spin.cpp ..
  cp pair_nep_spin.h ..

  # NEP core files
  cp nep_spin_data.cpp ..
  cp nep_spin_data.h ..
  cp nep_types.h ..
  cp descriptor.cpp ..
  cp descriptor.h ..
  cp math_utils.cpp ..
  cp math_utils.h ..
  cp cg_coefficients.h ..
  cp model.cpp ..
  cp model.h ..
  cp neighbor_list.cpp ..
  cp neighbor_list.h ..
  cp xyz_reader.cpp ..
  cp xyz_reader.h ..

  # RK4 integrator (alternative to SIB)
  cp fix_nve_spin_rk4.cpp ..
  cp fix_nve_spin_rk4.h ..
  cp fix_langevin_spin_rk4.cpp ..
  cp fix_langevin_spin_rk4.h ..

  # Kokkos files - check if KOKKOS package is installed
  if (test -e ../kokkos.h) then
    cp pair_nep_spin_kokkos.cpp ..
    cp pair_nep_spin_kokkos.h ..
  fi

elif (test $1 = 0) then

  # Uninstall: remove files from src directory

  # Pair style
  rm -f ../pair_nep_spin.cpp
  rm -f ../pair_nep_spin.h

  # NEP core files
  rm -f ../nep_spin_data.cpp
  rm -f ../nep_spin_data.h
  rm -f ../nep_types.h
  rm -f ../descriptor.cpp
  rm -f ../descriptor.h
  rm -f ../math_utils.cpp
  rm -f ../math_utils.h
  rm -f ../cg_coefficients.h
  rm -f ../model.cpp
  rm -f ../model.h
  rm -f ../neighbor_list.cpp
  rm -f ../neighbor_list.h
  rm -f ../xyz_reader.cpp
  rm -f ../xyz_reader.h

  # RK4 integrator
  rm -f ../fix_nve_spin_rk4.cpp
  rm -f ../fix_nve_spin_rk4.h
  rm -f ../fix_langevin_spin_rk4.cpp
  rm -f ../fix_langevin_spin_rk4.h

  # Kokkos files
  rm -f ../pair_nep_spin_kokkos.cpp
  rm -f ../pair_nep_spin_kokkos.h

fi
