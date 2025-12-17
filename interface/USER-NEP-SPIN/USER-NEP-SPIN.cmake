# USER-NEP-SPIN package configuration

# Require SPIN package
if(NOT PKG_SPIN)
  message(FATAL_ERROR "USER-NEP-SPIN package requires SPIN package. Please enable -DPKG_SPIN=ON")
endif()

# Workaround for PyTorch MKL paths issue
# Create a dummy MKL_INCLUDE_DIR if it doesn't exist
if(NOT DEFINED MKL_INCLUDE_DIR)
  set(MKL_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Dummy MKL include dir")
endif()

# Find PyTorch (LibTorch)
find_package(Torch REQUIRED)

if(Torch_FOUND)
  message(STATUS "USER-NEP-SPIN: Found PyTorch ${Torch_VERSION}")
  message(STATUS "USER-NEP-SPIN: Torch libraries: ${TORCH_LIBRARIES}")
  message(STATUS "USER-NEP-SPIN: Torch include dirs: ${TORCH_INCLUDE_DIRS}")

  # Clean up all invalid paths from all torch-related targets
  foreach(_target torch torch_library c10 c10_cuda)
    if(TARGET ${_target})
      get_target_property(_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      if(_includes)
        list(FILTER _includes EXCLUDE REGEX ".*NOTFOUND.*")
        set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_includes}")
      endif()
    endif()
  endforeach()

  # Add sources
  target_sources(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/USER-NEP-SPIN/pair_nep_spin.cpp
    ${LAMMPS_SOURCE_DIR}/USER-NEP-SPIN/nep_descriptor.cpp
    ${LAMMPS_SOURCE_DIR}/USER-NEP-SPIN/nep_math_utils.cpp
  )

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

  # Add USER-NEP-SPIN include directory
  target_include_directories(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/USER-NEP-SPIN)

  # Add preprocessor definitions
  target_compile_definitions(lammps PRIVATE USE_NEP_SPIN)

  # CUDA support (optional)
  if(TORCH_CUDA_AVAILABLE)
    message(STATUS "USER-NEP-SPIN: CUDA support enabled")
    target_compile_definitions(lammps PRIVATE TORCH_CUDA_AVAILABLE)
  else()
    message(STATUS "USER-NEP-SPIN: CPU-only mode")
  endif()

else()
  message(FATAL_ERROR "USER-NEP-SPIN: PyTorch (LibTorch) not found. Please set CMAKE_PREFIX_PATH to LibTorch location.")
endif()
