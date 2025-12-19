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
### Build LAMMPS using `make` 
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

  TORCH_PATH = /home//your_libtorch_path/

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

### Building LAMMPS Using `cmake` (prefered)

If you need to add GPU support, it is recommended to compile the Kokko package using CMake.
For the Kokko library, you need to download the CUDA version of the libtorch library.

```bash
# CUDA 12.1 version
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cu121.zip
```

* step 1: Copy the files in `src/` into `interface/USER-NEP-SPIN/` such that you have the following files in `interface/USER-NEP-SPIN/`:
  ```shell
  make interface #copy the files into the interface/USER-NEP-SPIN/
  make clean-interface #remove the files
  cp interface/USER-NEP-SPIN/ your/lammps/src/ -rf
  ```
* step 2: building lammps using cmake:

```bash
# ====== Modify these paths and GPU architecture ======
LAMMPS_DIR=/path/to/lammps
LIBTORCH_PATH=/path/to/libtorch
# GPU Arch: V100=VOLTA70, A100=AMPERE80, RTX3090=AMPERE86, RTX4090=ADA89, H100=HOPPER90
GPU_ARCH=AMPERE80
# =====================================================

cd ${LAMMPS_DIR}
mkdir -p build_kokkos && cd build_kokkos

cmake ../cmake \
    -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH} \
    -DCMAKE_CXX_FLAGS="-I${LIBTORCH_PATH}/include -I${LIBTORCH_PATH}/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1" \
    -DPKG_SPIN=ON \
    -DPKG_USER-NEP-SPIN=ON \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_${GPU_ARCH}=ON \
    -DCMAKE_CXX_COMPILER=${LAMMPS_DIR}/lib/kokkos/bin/nvcc_wrapper \
    -DPKG_MOLECULE=ON \
    -DPKG_KSPACE=ON \
    -DPKG_MANYBODY=ON \
    -DBUILD_MPI=ON \
    -DBUILD_OMP=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

# Verify
./lmp -h | grep spin/nep
```


## Use the NEP-LAMMPS interface

* `atom_style` can only be `spin`
* `units` must be `metal`
* Specify the `pair_style` in the following way:
  ```shell
  pair_style spin/nep   # YOUR_NEP_MODEL_FILE.txt is your NEP model file (with path)
  pair_coeff * * YOUR_NEP_MODEL_FILE.pt Cr I                        # This format is fixed
  ```
When using Kokkos version, specify GPU devices:

```bash
# Single GPU
mpirun -np 1 ./lmp -k on g 1 -sf kk -i input.in

# Multiple GPUs (one GPU per MPI process)
mpirun -np 4 ./lmp -k on g 4 -sf kk -i input.in
```
  

# Citation

* If you directly or indirectly use the `NEP-SPIN` class here, you are suggested to cite the following paper:


