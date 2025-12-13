cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nep_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Users need to set LibTorch path
# Example: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
if(NOT CMAKE_PREFIX_PATH)
    message(WARNING "CMAKE_PREFIX_PATH is not set. Please specify the path to LibTorch, e.g., -DCMAKE_PREFIX_PATH=/path/to/libtorch")
endif()

# Find dependencies
find_package(Torch REQUIRED)
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, using header-only Eigen")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eigen-3.4.0)
endif()

# Set C++ compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files list
set(SOURCES
    src/main.cpp
    src/xyz_reader.cpp
    src/neighbor_list.cpp
    src/math_utils.cpp
    src/descriptor.cpp
    src/model.cpp
)

# Executable
add_executable(nep_inference ${SOURCES})

# Link libraries
if(Eigen3_FOUND)
    target_link_libraries(nep_inference "${TORCH_LIBRARIES}" Eigen3::Eigen)
else()
    target_link_libraries(nep_inference "${TORCH_LIBRARIES}")
endif()

# Set rpath to find LibTorch dynamic libraries
set_property(TARGET nep_inference PROPERTY CXX_STANDARD 17)

# Enable CUDA support if available
if(TORCH_CUDA_FOUND)
    message(STATUS "CUDA found, enabling CUDA support")
    enable_language(CUDA)
endif()

# Copy test data to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/example/test.xyz
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Output compilation information
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
