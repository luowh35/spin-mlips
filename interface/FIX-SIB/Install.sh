#!/bin/bash

# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# enforce using portable C locale
LC_ALL=C
export LC_ALL

# arg1 = file, arg2 = move file, arg3 = copy file
action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test -n "$3") then
        cp $1 ../$3
      fi
    fi
  fi
}

# list of files

action fix_nve_spin_sib.cpp
action fix_nve_spin_sib.h
action fix_landau_spin.cpp
action fix_landau_spin.h
action fix_langevin_spin_sib.cpp
action fix_langevin_spin_sib.h
action fix_glangevin_spin_sib.cpp
action fix_glangevin_spin_sib.h
action pair_spin_ml.h
