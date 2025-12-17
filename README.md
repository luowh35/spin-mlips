# NEP-SPIN_CPU

# What does this repository contain?

* C++ inference implementation of MagneticNEP neural network potential, supporting energy, force, and magnetic force calculations for magnetic systems.

* An interface of the `NEP-SPIN` class to the CPU version of LAMMPS (https://github.com/lammps/lammps). **It can be run with MPI**.

# The standalone C++ implementation of NEP-SPIN

* The `NEP3` C++ class is defined in the following three files:
  * `src/cg_coefficients.h `
  * `src/descriptor.h`
  * `src/math_utils.cpp`
  * `src/model.cpp`
  * `src/neighbor_list.cpp`
  * `src/nep_types.h`
  * `src/xyz_reader.h`
  * `src/descriptor.cpp`
  * `src/main.cpp`
  * `src/math_utils.h`
  * `src/model.h`
  * `src/neighbor_list.h`
  * `src/xyz_reader.cpp`

                
                       
  

* The following folders contain some testing code and results:
  * `example/`
  

# The NEP-LAMMPS interface

## Build the NEP-LAMMPS interface

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

* step 1: Copy the files in `src/` into `interface/USER-NEP-SPIN/` such that you have the following files in `interface/USER-NEP-SPIN/`:
  ```shell
  make interface #copy the files into the interface/USER-NEP-SPIN/
  make clean-interface #remove the files
  ```
  
* Step 2: Now you can copy the `USER-NEP-SPIN/` folder into `YOUR_LAMMPS_PATH/src/` and start to compile LAMMPS in your favorite way. 
  modify the Makefile (src/MAKE/Makefile.mpi):
  ```shell
  CCFLAGS = -g -O3 -std=c++17
  LINKFLAGS = -g -O3 -std=c++17
  
  ...

  TORCH_PATH = /home/wjw106/luowh/nep_spin/libtorch

  TORCH_INC = -I$(TORCH_PATH)/include \
              -I$(TORCH_PATH)/include/torch/csrc/api/include

  TORCH_PATH_LIB = -L$(TORCH_PATH)/lib

  TORCH_LIB = -ltorch \
              -ltorch_cpu \
              -lc10 \
              -Wl,-rpath,$(TORCH_PATH)/lib
  
  JPG_INC = $(TORCH_INC)
  JPG_PATH = $(TORCH_PATH_LIB)
  JPG_LIB = $(TORCH_LIB)
  ```
  after modified the makefile, in your/lammps/src:
  ```shell
  make yes-spin
  make yes-user-nep-spin
  make mpi -j4
  ```

## Use the NEP-LAMMPS interface

* `atom_style` can only be `spin`
* `units` must be `metal`
* Specify the `pair_style` in the following way:
  ```shell
  pair_style nep   # YOUR_NEP_MODEL_FILE.txt is your NEP model file (with path)
  pair_coeff * * YOUR_NEP_MODEL_FILE.pt Cr I                        # This format is fixed
  ```
  

# Citation

* If you directly or indirectly use the `NEP-SPIN` class here, you are suggested to cite the following paper:


