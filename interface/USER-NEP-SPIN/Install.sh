#!/bin/bash

# Install/unInstall package files in LAMMPS src dir
# Note: fix_nve_spin_sib and fix_langevin_spin_sib are now in USER-SPIN-ML package

if (test $1 = 1) then

  # Install: copy files to src directory

  # Base files (always installed)
  cp pair_nep_spin.cpp ..
  cp pair_nep_spin.h ..
  cp nep_spin_data.cpp ..
  cp nep_spin_data.h ..

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

  # Base files
  rm -f ../pair_nep_spin.cpp
  rm -f ../pair_nep_spin.h
  rm -f ../nep_spin_data.cpp
  rm -f ../nep_spin_data.h

  # RK4 integrator
  rm -f ../fix_nve_spin_rk4.cpp
  rm -f ../fix_nve_spin_rk4.h
  rm -f ../fix_langevin_spin_rk4.cpp
  rm -f ../fix_langevin_spin_rk4.h

  # Kokkos files
  rm -f ../pair_nep_spin_kokkos.cpp
  rm -f ../pair_nep_spin_kokkos.h

fi
