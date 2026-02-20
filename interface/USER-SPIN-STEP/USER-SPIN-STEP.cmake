# USER-SPIN-STEP package configuration
# SPIN-STEP: Machine Learning Spin potential for LAMMPS
#
# This package provides pair_style spin/step for magnetic systems
# using STEP models based on e3nn equivariant neural networks.
#
# The fix nve/spin/sib and fix langevin/spin/sib are provided by
# the shared USER-SPIN-ML package.
#
# Requirements:
# - LAMMPS SPIN package (for atom_style spin and fix nve/spin)
# - USER-SPIN-ML package (for fix nve/spin/sib and fix langevin/spin/sib)
# - LibTorch (PyTorch C++ API) >= 2.0
# - C++17 compatible compiler
#
# Usage:
#   cmake -DPKG_SPIN=ON -DPKG_USER-SPIN-ML=ON -DPKG_USER-SPIN-STEP=ON \
#         -DCMAKE_PREFIX_PATH=/path/to/libtorch \
#         ../cmake
#
# Or if using conda-installed PyTorch:
#   LIBTORCH_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
#   cmake -DPKG_SPIN=ON -DPKG_USER-SPIN-ML=ON -DPKG_USER-SPIN-STEP=ON \
#         -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH \
#         ../cmake

# =============================================================================
# Package Dependencies
# =============================================================================

# Require SPIN package
if(NOT PKG_SPIN)
  message(FATAL_ERROR "USER-SPIN-STEP package requires SPIN package. "
                      "Please enable with -DPKG_SPIN=ON")
endif()

# Require USER-SPIN-ML package (provides fix nve/spin/sib, fix langevin/spin/sib, pair_spin_ml.h)
if(NOT PKG_USER-SPIN-ML)
  message(FATAL_ERROR "USER-SPIN-STEP package requires USER-SPIN-ML package. "
                      "Please enable with -DPKG_USER-SPIN-ML=ON")
endif()

# =============================================================================
# Find PyTorch (LibTorch)
# =============================================================================

# Workaround for PyTorch MKL paths issue
if(NOT DEFINED MKL_INCLUDE_DIR)
  set(MKL_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Dummy MKL include dir")
endif()

find_package(Torch REQUIRED)

if(Torch_FOUND)
  message(STATUS "USER-SPIN-STEP: Found PyTorch ${Torch_VERSION}")
  message(STATUS "USER-SPIN-STEP: Torch libraries: ${TORCH_LIBRARIES}")
  message(STATUS "USER-SPIN-STEP: Torch include dirs: ${TORCH_INCLUDE_DIRS}")

  # Clean up all invalid paths from torch-related targets
  foreach(_target torch torch_library c10 c10_cuda)
    if(TARGET ${_target})
      get_target_property(_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      if(_includes)
        list(FILTER _includes EXCLUDE REGEX ".*NOTFOUND.*")
        set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_includes}")
      endif()
    endif()
  endforeach()

  # =============================================================================
  # Source Files
  # =============================================================================

  set(SPIN_STEP_SOURCES
    ${LAMMPS_SOURCE_DIR}/USER-SPIN-STEP/pair_spin_step.cpp
    ${LAMMPS_SOURCE_DIR}/USER-SPIN-STEP/step_utils.cpp
  )

  # =============================================================================
  # Build Configuration
  # =============================================================================

  # Add sources to LAMMPS
  target_sources(lammps PRIVATE ${SPIN_STEP_SOURCES})

  # Link LibTorch
  target_link_libraries(lammps PUBLIC ${TORCH_LIBRARIES})
  target_include_directories(lammps PUBLIC ${TORCH_INCLUDE_DIRS})

  # C++17 required for LibTorch
  target_compile_features(lammps PUBLIC cxx_std_17)

  # Compiler flags
  target_compile_options(lammps PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
  )

  # Add USER-SPIN-STEP include directory
  target_include_directories(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/USER-SPIN-STEP)

  # Add preprocessor definitions
  target_compile_definitions(lammps PRIVATE USE_SPIN_STEP)

  # =============================================================================
  # CUDA Support
  # =============================================================================

  if(TORCH_CUDA_AVAILABLE)
    message(STATUS "USER-SPIN-STEP: CUDA support enabled")
    target_compile_definitions(lammps PRIVATE TORCH_CUDA_AVAILABLE)
  else()
    message(STATUS "USER-SPIN-STEP: CPU-only mode")
  endif()

  # =============================================================================
  # Summary
  # =============================================================================

  message(STATUS "")
  message(STATUS "USER-SPIN-STEP Configuration Summary:")
  message(STATUS "  PyTorch Version: ${Torch_VERSION}")
  message(STATUS "  CUDA Available: ${TORCH_CUDA_AVAILABLE}")
  message(STATUS "  Source Files: ${SPIN_STEP_SOURCES}")
  message(STATUS "  Depends on: USER-SPIN-ML (fix nve/spin/sib, fix langevin/spin/sib)")
  message(STATUS "")

else()
  message(FATAL_ERROR
    "USER-SPIN-STEP: PyTorch (LibTorch) not found.\n"
    "Please set CMAKE_PREFIX_PATH to LibTorch location.\n"
    "If using conda-installed PyTorch, try:\n"
    "  LIBTORCH_PATH=$(python -c \"import torch; print(torch.utils.cmake_prefix_path)\")\n"
    "  cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ...")
endif()
