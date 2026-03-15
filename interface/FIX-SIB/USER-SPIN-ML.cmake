# USER-SPIN-ML package configuration
# Shared components for Machine Learning Spin potentials in LAMMPS
#
# This package provides:
# - pair_spin_ml.h: Abstract base class for ML spin pair styles
# - fix nve/spin/sib: SIB integrator for spin-lattice dynamics
# - fix langevin/spin/sib: Langevin thermostat for SIB method
#
# These components are shared by:
# - USER-NEP-SPIN (pair_style spin/nep)
# - USER-SPIN-STEP (pair_style spin/step)
# - Any future ML spin potential packages
#
# Requirements:
# - LAMMPS SPIN package (for atom_style spin)
#
# Usage:
#   cmake -DPKG_SPIN=ON -DPKG_USER-SPIN-ML=ON ...
#
# Note: This package should be enabled automatically when enabling
# USER-NEP-SPIN or USER-SPIN-STEP packages.

# =============================================================================
# Package Dependencies
# =============================================================================

# Require SPIN package
if(NOT PKG_SPIN)
  message(FATAL_ERROR "USER-SPIN-ML package requires SPIN package. "
                      "Please enable with -DPKG_SPIN=ON")
endif()

# =============================================================================
# Source Files
# =============================================================================

set(SPIN_ML_SOURCES
  ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML/fix_nve_spin_sib.cpp
  ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML/fix_langevin_spin_sib.cpp
  ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML/fix_glangevin_spin_sib.cpp
)

# Header files (for IDE integration)
set(SPIN_ML_HEADERS
  ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML/pair_spin_ml.h
  ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML/fix_nve_spin_sib.h
  ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML/fix_langevin_spin_sib.h
  ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML/fix_glangevin_spin_sib.h
)

# =============================================================================
# Build Configuration
# =============================================================================

# Add sources to LAMMPS
target_sources(lammps PRIVATE ${SPIN_ML_SOURCES})

# Add USER-SPIN-ML include directory (for pair_spin_ml.h)
target_include_directories(lammps PUBLIC ${LAMMPS_SOURCE_DIR}/USER-SPIN-ML)

# Add preprocessor definitions
target_compile_definitions(lammps PRIVATE USE_SPIN_ML)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "USER-SPIN-ML Configuration Summary:")
message(STATUS "  Components: fix nve/spin/sib, fix langevin/spin/sib, fix glangevin/spin/sib")
message(STATUS "  Base class: pair_spin_ml.h (header-only)")
message(STATUS "  Source files: ${SPIN_ML_SOURCES}")
message(STATUS "")
