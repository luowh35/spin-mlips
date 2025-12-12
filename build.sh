#!/bin/bash
# MagneticNEP C++ Build Script

set -e

# ============================================================================
# Configuration - Modify according to your setup
# ============================================================================

# LibTorch path (required)
LIBTORCH_PATH="$(pwd)/dependencies/libtorch"

# Eigen3 path (required)
EIGEN3_PATH="$(pwd)/dependencies/eigen-3.4.0"

# Build type: Release or Debug
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Parallel compilation jobs
JOBS="${JOBS:-4}"

# ============================================================================
# Build
# ============================================================================

echo "============================================================"
echo "MagneticNEP C++ Build Script"
echo "============================================================"
echo ""

# Check LibTorch path
if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "Error: LibTorch path does not exist: $LIBTORCH_PATH"
    echo "Please set the correct LibTorch path in this script or via:"
    echo "  export LIBTORCH_PATH=/path/to/libtorch"
    exit 1
fi

echo "LibTorch path: $LIBTORCH_PATH"

# Check Eigen3 path
if [ ! -d "$EIGEN3_PATH" ]; then
    echo "Error: Eigen3 path does not exist: $EIGEN3_PATH"
    echo "Please set the correct Eigen3 path in this script or via:"
    echo "  export EIGEN3_PATH=/path/to/eigen-3.4.0"
    exit 1
fi

echo "Eigen3 path: $EIGEN3_PATH"
echo ""

echo "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Parallel jobs: $JOBS"
echo ""

# Create build directory
echo "[1/3] Creating build directory..."
mkdir -p build
cd build

# Configure CMake
echo "[2/3] Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH \
      -DEIGEN3_INCLUDE_DIR=$EIGEN3_PATH \
      ..

# Compile
echo "[3/3] Compiling..."
make -j$JOBS

echo ""
echo "============================================================"
echo "Build successful!"
echo "============================================================"
echo ""
echo "Executable: build/nep_inference"
echo ""
echo "Run example:"
echo "  cd build"
echo "  ./nep_inference"
echo ""
