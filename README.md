# MagneticNEP C++ Inference Module

C++ inference implementation of MagneticNEP neural network potential, supporting energy, force, and magnetic force calculations for magnetic systems.

## File Structure

```
MagneticNEP_CPP/
├── README.md              # This document
├── CMakeLists.txt         # CMake build configuration
├── include/               # Header files
│   ├── nep_types.h            # Basic type definitions
│   ├── math_utils.h           # Math utility functions
│   ├── xyz_reader.h           # XYZ file reader
│   ├── neighbor_list.h        # Neighbor list builder
│   ├── cg_coefficients.h      # Clebsch-Gordan coefficients
│   ├── descriptor.h           # ACE descriptor computation
│   └── model.h                # Neural network model inference
├── src/                   # Source code
│   ├── math_utils.cpp         # Math utilities implementation
│   ├── xyz_reader.cpp         # XYZ reader implementation
│   ├── neighbor_list.cpp      # Neighbor list implementation
│   ├── descriptor.cpp         # ACE descriptor implementation
│   ├── model.cpp              # Model inference implementation
│   └── main.cpp               # Example program entry
├── examples/              # Example files
│   └── test.xyz               # Test structure file
└── data/                  # Model file directory (user provided)
    └── best_model_traced.pt   # TorchScript model file
```

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Compiler**: GCC 7.5+ or Clang 10+ (C++14 support required)
- **CMake**: 3.10 or higher

### Dependencies

#### 1. LibTorch (Required)
- **Version**: 2.0.0 or higher (2.0.1 recommended)
- **Type**: C++ CPU version (cxx11 ABI)
- **Download**: https://pytorch.org/get-started/locally/
- **Example** (Linux, CPU, cxx11):
  ```bash
  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
  unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip
  ```

#### 2. Eigen3 (Required)
- **Version**: 3.3.7 or higher
- **Type**: Header-only library (no compilation, no sudo needed)
- **Installation**:
  ```bash
  # Download without sudo (recommended)
  wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  tar -xzf eigen-3.4.0.tar.gz
  # Then set: export EIGEN3_PATH=$(pwd)/eigen-3.4.0
  ```

## Quick Build

### Method 1: Automatic Installation (Recommended, No Sudo Required)

If you haven't installed LibTorch and Eigen3, use the automatic installation script:

```bash
# Automatically download and install all dependencies (no sudo required)
chmod +x install_dependencies.sh
./install_dependencies.sh

# The script will download:
#   - LibTorch 2.0.1 (~180MB compressed)
#   - Eigen3 3.4.0 (~2.7MB)
# And automatically build MagneticNEP
```

### Method 2: Manual Build (Dependencies Already Installed)

If you have already installed LibTorch and Eigen3, use the provided build script:

```bash
# The build script is already configured
chmod +x build.sh
./build.sh
```

Or manually build:

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure CMake (set LibTorch path)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DEIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0 \
      ..

# 3. Build
make -j4

# 4. Run
./nep_inference
```

## Usage Example

### Prepare Model File

Convert your trained PyTorch model to TorchScript format:

```python
import torch

# Load your model
model = YourMagneticNEPModel(...)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Convert to TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save('best_model_traced.pt')
```

Place `best_model_traced.pt` in the `data/` directory.

### Run Inference

```bash
cd build
./nep_inference

# Default uses:
# - Model: ../example/best_model_traced.pt
# - Structure: ../example/test.xyz
```


## Common Issues

### Build Errors

**Q: "Could not find Torch"**
- A: Ensure `CMAKE_PREFIX_PATH` is set to the LibTorch directory

**Q: "undefined reference to torch::xxx"**
- A: Check that you're using the correct ABI version (cxx11 vs pre-cxx11)

**Q: "Eigen/Core: No such file"**
- A: Install Eigen3 or set `EIGEN3_INCLUDE_DIR`

### Runtime Errors

**Q: "error loading model"**
- A: Ensure the model file is in TorchScript format (.pt file)

**Q: Energy/forces inconsistent with Python version**
- A: Check that descriptor configuration matches training parameters exactly

**Q: "Descriptor dimension mismatch"**
- A: Model expects different descriptor dimension, check n_max, l_max parameters

## Performance Optimization

- Build in Release mode: `-DCMAKE_BUILD_TYPE=Release`
- Enable compiler optimizations: `-O3 -march=native`
- Multi-threaded LibTorch: set `torch::set_num_threads(4)`
- Batch computation: process multiple structures at once

## License and Citation

If you use this code, please cite the relevant papers.

## Technical Support

For questions, please contact: [Your contact information]

---

