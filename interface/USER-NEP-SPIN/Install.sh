#!/bin/bash

# Install/unInstall package files in LAMMPS src dir

if (test $1 = 1) then

  # Install: copy files to src directory

  # Base files (always installed)
  cp pair_nep_spin.cpp ..
  cp pair_nep_spin.h ..
  cp fix_nve_spin_sib.cpp ..
  cp fix_nve_spin_sib.h ..
  cp fix_langevin_spin_sib.cpp ..
  cp fix_langevin_spin_sib.h ..
  cp descriptor.cpp ..
  cp descriptor.h ..
  cp math_utils.cpp ..
  cp math_utils.h ..
  cp nep_types.h ..
  cp cg_coefficients.h ..

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
  rm -f ../fix_nve_spin_sib.cpp
  rm -f ../fix_nve_spin_sib.h
  rm -f ../fix_langevin_spin_sib.cpp
  rm -f ../fix_langevin_spin_sib.h
  rm -f ../descriptor.cpp
  rm -f ../descriptor.h
  rm -f ../math_utils.cpp
  rm -f ../math_utils.h
  rm -f ../nep_types.h
  rm -f ../cg_coefficients.h

  # Kokkos files
  rm -f ../pair_nep_spin_kokkos.cpp
  rm -f ../pair_nep_spin_kokkos.h

fi
