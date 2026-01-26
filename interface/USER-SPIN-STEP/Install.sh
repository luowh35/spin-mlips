#!/bin/bash

# Install/unInstall package files in LAMMPS src dir
# Note: fix_nve_spin_sib and fix_langevin_spin_sib are now in USER-SPIN-ML package

if (test $1 = 1) then

  # Install: copy files to src directory

  # Base files (always installed)
  cp step_utils.cpp ..
  cp step_utils.h ..
  cp pair_spin_step.cpp ..
  cp pair_spin_step.h ..

elif (test $1 = 0) then

  # Uninstall: remove files from src directory

  # Base files
  rm -f ../step_utils.cpp
  rm -f ../step_utils.h
  rm -f ../pair_spin_step.cpp
  rm -f ../pair_spin_step.h

fi
