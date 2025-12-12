#!/bin/bash
# Automatic installation of all dependencies (no sudo required)
# This script will download LibTorch and Eigen3 to the current directory

set -e

echo "============================================================"
echo "MagneticNEP C++ Dependency Installation Script"
echo "============================================================"
echo ""
echo "This script will download the following dependencies (no sudo required):"
echo "  1. LibTorch 2.0.1 (CPU version, ~180MB compressed, ~700MB extracted)"
echo "  2. Eigen3 3.4.0 (header-only library, ~2.7MB)"
echo ""

# Ask to continue
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled"
    exit 0
fi

INSTALL_DIR="$(pwd)/dependencies"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo ""
echo "============================================================"
echo "[1/2] Downloading LibTorch 2.0.1"
echo "============================================================"

if [ -d "libtorch" ]; then
    echo "✓ LibTorch already exists, skipping download"
else
    echo "Downloading LibTorch (~180MB, may take a few minutes)..."
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip"

    if command -v wget &> /dev/null; then
        wget --progress=bar:force "$LIBTORCH_URL" -O libtorch.zip
    elif command -v curl &> /dev/null; then
        curl -L "$LIBTORCH_URL" -o libtorch.zip --progress-bar
    else
        echo "Error: wget or curl is required to download files"
        exit 1
    fi

    echo "Extracting LibTorch..."
    unzip -q libtorch.zip
    rm libtorch.zip
    echo "✓ LibTorch installation complete"
fi

LIBTORCH_PATH="$(pwd)/libtorch"

echo ""
echo "============================================================"
echo "[2/2] Downloading Eigen3 3.4.0"
echo "============================================================"

if [ -d "eigen-3.4.0" ]; then
    echo "✓ Eigen3 already exists, skipping download"
else
    echo "Downloading Eigen3 (~2.7MB)..."
    EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"

    if command -v wget &> /dev/null; then
        wget --progress=bar:force "$EIGEN_URL" -O eigen-3.4.0.tar.gz
    elif command -v curl &> /dev/null; then
        curl -L "$EIGEN_URL" -o eigen-3.4.0.tar.gz --progress-bar
    else
        echo "Error: wget or curl is required to download files"
        exit 1
    fi

    echo "Extracting Eigen3..."
    tar -xzf eigen-3.4.0.tar.gz
    rm eigen-3.4.0.tar.gz
    echo "✓ Eigen3 installation complete"
fi

EIGEN3_PATH="$(pwd)/eigen-3.4.0"

echo ""
echo "============================================================"
echo "✓ All dependencies installed successfully!"
echo "============================================================"
echo ""
echo "Installation locations:"
echo "  LibTorch: $LIBTORCH_PATH"
echo "  Eigen3:   $EIGEN3_PATH"
echo ""
echo "Environment variables (recommended to add to ~/.bashrc):"
echo "  export LIBTORCH_PATH='$LIBTORCH_PATH'"
echo "  export EIGEN3_PATH='$EIGEN3_PATH'"
echo ""
echo "Now you can compile MagneticNEP:"
echo "  export LIBTORCH_PATH='$LIBTORCH_PATH'"
echo "  export EIGEN3_PATH='$EIGEN3_PATH'"
echo "  ./build.sh"
echo ""

# Ask to build immediately
read -p "Build MagneticNEP now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ..
    export LIBTORCH_PATH="$LIBTORCH_PATH"
    export EIGEN3_PATH="$EIGEN3_PATH"
    ./build.sh
fi
