#!/bin/bash

# Install/unInstall package files in LAMMPS src dir

if (test $1 = 1) then

  # Install: copy files to src directory
  cp pair_nep_spin.cpp ..
  cp pair_nep_spin.h ..
  cp descriptor.cpp ..
  cp descriptor.h ..
  cp math_utils.cpp ..
  cp math_utils.h ..
  cp nep_types.h ..
  cp cg_coefficients.h ..

elif (test $1 = 0) then

  # Uninstall: remove files from src directory
  rm -f ../pair_nep_spin.cpp
  rm -f ../pair_nep_spin.h
  rm -f ../descriptor.cpp
  rm -f ../descriptor.h
  rm -f ../math_utils.cpp
  rm -f ../math_utils.h
  rm -f ../nep_types.h
  rm -f ../cg_coefficients.h

fi
