# Spin-MLIPs

# What does this repository contain?

* C++ inference implementation of MagneticNEP neural network potential, supporting energy, force, and magnetic force calculations for magnetic systems.

* LAMMPS interface packages for machine-learning spin potentials:
  - `USER-NEP-SPIN` — `pair_style spin/nep` (NEP-SPIN model)
  - `USER-SPIN-STEP` — `pair_style spin/step` (STEP model based on e3nn equivariant neural networks)
  - `FIX-SIB` (USER-SPIN-ML) — shared SIB integrators and thermostats for ML spin potentials

* All interfaces can be run with MPI.

# The standalone C++ implementation of NEP-SPIN

* The `NEP3` C++ class is defined in the following files:
  * `src/cg_coefficients.h`
  * `src/descriptor.h`
  * `src/descriptor.cpp`
  * `src/math_utils.h`
  * `src/math_utils.cpp`
  * `src/model.h`
  * `src/model.cpp`
  * `src/neighbor_list.h`
  * `src/neighbor_list.cpp`
  * `src/nep_types.h`
  * `src/xyz_reader.h`
  * `src/xyz_reader.cpp`
  * `src/main.cpp`

* The following folders contain some testing code and results:
  * `example/`

# LAMMPS Interface Packages

The `interface/` directory contains three packages:

| Package | Directory | Provides |
|---------|-----------|----------|
| USER-NEP-SPIN | `interface/USER-NEP-SPIN/` | `pair_style spin/nep` (with Kokkos GPU support) |
| USER-SPIN-STEP | `interface/USER-SPIN-STEP/` | `pair_style spin/step` |
| USER-SPIN-ML (FIX-SIB) | `interface/FIX-SIB/` | `fix nve/spin/sib`, `fix langevin/spin/sib`, `fix glangevin/spin/sib`, `fix landau/spin`, `pair_spin_ml.h` base class |

USER-SPIN-ML is a shared dependency — both USER-NEP-SPIN and USER-SPIN-STEP rely on it for the SIB integrator and thermostats.

## Fix Styles (provided by USER-SPIN-ML)

### fix nve/spin/sib

Time integrator for coupled spin-lattice dynamics using the Semi-Implicit B (SIB) predictor-corrector method.

```
fix ID group nve/spin/sib keyword value ...
```

Optional keywords:
- `lattice yes/no` — enable/disable lattice dynamics (default: yes). Use `lattice no` for pure spin dynamics with frozen atomic positions.

This fix automatically discovers and uses:
- `PairSpinML` derivatives (`spin/step`, `spin/nep`)
- Standard `PairSpin` styles (`spin/exchange`, `spin/dmi`, etc.)
- `fix precession/spin` (external magnetic field)
- `fix langevin/spin/sib` (fixed-length thermostat)
- `fix glangevin/spin/sib` (variable-length thermostat)
- `fix landau/spin` (on-site Landau potential)

### fix langevin/spin/sib

Stochastic Langevin thermostat for fixed-length spin dynamics. Spin magnitude `|m|` is constant; only the spin direction evolves.

```
fix ID group langevin/spin/sib temp alpha_t seed
```

Parameters:
- `temp` — spin bath temperature (K)
- `alpha_t` — transverse (Gilbert) damping coefficient (dimensionless, typical: 0.01–0.1)
- `seed` — random number generator seed

### fix glangevin/spin/sib

Generalized Langevin thermostat for variable-length spin dynamics. Both spin direction and magnitude `|m|` evolve, enabling longitudinal spin fluctuations.

```
fix ID group glangevin/spin/sib temp alpha_t tau_L seed
```

Parameters:
- `temp` — spin bath temperature (K)
- `alpha_t` — transverse (Gilbert) damping coefficient (dimensionless, typical: 0.01–0.1)
- `tau_L` — longitudinal relaxation time (ps, typical: 0.01–0.1). Controls how fast `|m|` relaxes.
- `seed` — random number generator seed

This fix requires a potential that provides a `|m|`-dependent energy surface (e.g. `fix landau/spin` or a `PairSpinML` that implements `distribute_full_mag_forces()`).

### fix landau/spin

On-site Landau potential for variable-length spin dynamics.

```
fix ID group landau/spin a2 val [a4 val] [a6 val] [a8 val] ...
```

The potential energy per atom is: `E_L(m) = a2*m^2 + a4*m^4 + a6*m^6 + ...`

Multiple instances can be used with different groups to assign different Landau coefficients to different atom types. The global Landau energy is accessible via `f_<fix-ID>` for thermo output.

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Compiler**: GCC 7.5+ or Clang 10+ (C++17 support required)
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

#### 2. Eigen3 (Required for USER-NEP-SPIN)
- **Version**: 3.3.7 or higher
- **Type**: Header-only library (no compilation, no sudo needed)
- **Installation**:
  ```bash
  wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  tar -xzf eigen-3.4.0.tar.gz
  # Then set: export EIGEN3_PATH=$(pwd)/eigen-3.4.0
  ```

## Building LAMMPS

### Using `cmake` (preferred)

Step 1: Copy interface packages into your LAMMPS source tree:

```bash
# For NEP-SPIN:
make interface  # copies src/ files into interface/USER-NEP-SPIN/
cp -rf interface/USER-NEP-SPIN/ /path/to/lammps/src/

# For SPIN-STEP:
cp -rf interface/USER-SPIN-STEP/ /path/to/lammps/src/

# For FIX-SIB (USER-SPIN-ML, required by both):
cp -rf interface/FIX-SIB/ /path/to/lammps/src/USER-SPIN-ML/
```

Step 2: Build with cmake.

For CPU-only build (using NEP-SPIN as example):

```bash
LAMMPS_DIR=/path/to/lammps
LIBTORCH_PATH=/path/to/libtorch

cd ${LAMMPS_DIR}
mkdir -p build && cd build

cmake ../cmake \
    -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH} \
    -DCMAKE_CXX_FLAGS="-I${LIBTORCH_PATH}/include -I${LIBTORCH_PATH}/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1" \
    -DPKG_SPIN=ON \
    -DPKG_USER-SPIN-ML=ON \
    -DPKG_USER-NEP-SPIN=ON \
    -DPKG_MOLECULE=ON \
    -DPKG_KSPACE=ON \
    -DPKG_MANYBODY=ON \
    -DBUILD_MPI=ON \
    -DBUILD_OMP=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

For SPIN-STEP, replace `-DPKG_USER-NEP-SPIN=ON` with `-DPKG_USER-SPIN-STEP=ON`. To enable both:

```bash
    -DPKG_USER-SPIN-ML=ON \
    -DPKG_USER-NEP-SPIN=ON \
    -DPKG_USER-SPIN-STEP=ON \
```

For GPU (Kokkos) build, download the CUDA version of libtorch and add Kokkos flags:

```bash
# CUDA 12.1 version
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cu121.zip
```

```bash
# GPU Arch: V100=VOLTA70, A100=AMPERE80, RTX3090=AMPERE86, RTX4090=ADA89, H100=HOPPER90
GPU_ARCH=AMPERE80

cmake ../cmake \
    -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH} \
    -DCMAKE_CXX_FLAGS="-I${LIBTORCH_PATH}/include -I${LIBTORCH_PATH}/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1" \
    -DPKG_SPIN=ON \
    -DPKG_USER-SPIN-ML=ON \
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
```

### Using `make`

Step 1: Set up the environment and install PyTorch:

```bash
# Load compilers (adjust to your cluster environment)
source /opt/ohpc/pub/apps/intel/oneapi/setvars.sh
source /opt/rh/devtoolset-7/enable

# Create conda environment with PyTorch + CUDA
conda create -n libtorch pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
conda activate libtorch
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Get the torch installation path (you will need this for Makefile.mpi)
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
echo $TORCH_PATH   # e.g. /home/user/anaconda3/envs/libtorch/lib/python3.x/site-packages/torch
```

Step 2: Copy interface packages into your LAMMPS source tree:

```bash
make interface  # copy src files into interface/USER-NEP-SPIN/
cp -rf interface/USER-NEP-SPIN/ /path/to/lammps/src/
cp -rf interface/USER-SPIN-STEP/ /path/to/lammps/src/
cp -rf interface/FIX-SIB/ /path/to/lammps/src/USER-SPIN-ML/
```

Step 3: Modify `src/MAKE/Makefile.mpi`.

Find and change `CCFLAGS` and `LINKFLAGS` to enable C++17:

```makefile
CCFLAGS = -g -O3 -std=c++17
LINKFLAGS = -g -O3 -std=c++17
```

Then find the `JPG_INC`, `JPG_PATH`, `JPG_LIB` lines (near the bottom) and replace them with the following. Set `TORCH_PATH` to the output of `python -c "import torch; print(torch.__path__[0])"`:

```makefile
# ---- PyTorch (LibTorch) ----
TORCH_PATH = /home/user/anaconda3/envs/libtorch/lib/python3.x/site-packages/torch

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

Step 4: Build LAMMPS:

```bash
cd /path/to/lammps/src
make yes-spin
make yes-user-spin-ml
make yes-user-nep-spin    # and/or: make yes-user-spin-step
make mpi -j 24
```

## Usage

* `atom_style` must be `spin`
* `units` must be `metal`

### pair_style spin/nep

```
pair_style spin/nep
pair_coeff * * model.pt Cr I
```

### pair_style spin/step

```
pair_style spin/step
pair_coeff * * model.pt
```

### Example: Fixed-Length Spin Dynamics

```
atom_style    spin
pair_style    spin/step
pair_coeff    * * model.pt

fix nve       all nve/spin/sib lattice no
fix thermo    all langevin/spin/sib 300.0 0.05 12345

thermo        100
run           10000
```

### Example: Variable-Length Spin Dynamics with Landau Potential

```
atom_style    spin
pair_style    spin/step
pair_coeff    * * model.pt

fix landau    all landau/spin a2 1.0 a4 -0.5
fix nve       all nve/spin/sib lattice no
fix thermo    all glangevin/spin/sib 300.0 0.05 0.05 12345

thermo_style  custom step temp f_landau
thermo        100
run           10000
```

### Example: Mixed ML + Analytical Pair Styles

```
atom_style    spin
pair_style    hybrid/overlay spin/step spin/exchange
pair_coeff    * * spin/step model.pt
pair_coeff    * * spin/exchange 3.0 0.02 0.5

fix ext       all precession/spin zeeman 0.0 0.0 1.0
fix nve       all nve/spin/sib
fix thermo    all langevin/spin/sib 300.0 0.05 12345

thermo        100
run           10000
```

### Running with Kokkos (GPU)

```bash
# Single GPU
mpirun -np 1 ./lmp -k on g 1 -sf kk -i input.in

# Multiple GPUs (one GPU per MPI process)
mpirun -np 4 ./lmp -k on g 4 -sf kk -i input.in
```

## References

[1] J.H. Mentink, M.V. Tretyakov, A. Fasolino, M.I. Katsnelson, T. Rasing,
    "Stable and fast semi-implicit integration of the stochastic Landau-Lifshitz
    equation", J. Phys.: Condens. Matter 22, 176001 (2010).

[2] J. Tranchida, S.J. Plimpton, P. Thibaudeau, A.P. Thompson,
    "Massively Parallel Symplectic Algorithm for Coupled Magnetic Spin Dynamics
    and Molecular Dynamics", J. Comput. Phys. 372, 406–425 (2018).

[3] P.-W. Ma, S.L. Dudarev, "Longitudinal magnetic fluctuations in Langevin
    spin dynamics", Phys. Rev. B 86, 054416 (2012).

# Citation

* If you directly or indirectly use the `NEP-SPIN` class here, you are suggested to cite the following paper:


