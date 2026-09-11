# MPI review validation — 2026-09-11

Environment: LAMMPS 2 Aug 2023 Update 3, LibTorch/PyTorch 2.9.1+cu128,
MPICH, C++17. All numerical runs below explicitly used `device cpu`.
The local machine has only one physical GPU; multiple physical GPUs were not tested.

## Reproduced and fixed defects

1. SIB `setup()` called the ML pair a second time after Verlet setup had already
   computed forces. A `run 0` with a SIB integrator produced approximately twice
   the atomic and magnetic forces of the same configuration without that fix.
   Removed the redundant compute call in both NVE and Nose–Hoover SIB setup.
2. The second SIB half-step ran immediately after position/box updates, before
   LAMMPS migration and neighbor rebuilding. It could miss newly entering
   neighbors. Moved this half-step to `pre_force`, preserved pre-existing magnetic
   force contributions, and added scratch-array growth after migration.
   An independent two-atom reference exercises both a new neighbor and migration:
   the old version differed by up to 0.01587386 in magnetic moment components;
   the fixed version passed at atol=rtol=2e-6 for 1/2/4 ranks.
3. The regression data writer had magnitude and spin direction columns reversed.
   Corrected the LAMMPS spin data ordering and added a check that the moments read
   back from the simulation match the intended input before testing derivatives.

Also changed rank-local device/load failures to MPI-aborting error handling and
removed the silent 5 Å fallback for a missing model cutoff. Export metadata must
provide a positive finite `r_max`; an undersized guessed cutoff is unsafe.

## Passed tests

- Nonlinear 3-hop spin model vs independent full-cell PyTorch graph/autograd:
  1/2/4 ranks; batch_size 0 and 5; periodic orthogonal and triclinic boxes;
  empty ranks and isolated atoms. Energy, atomic force, magnetic force and global
  pressure agree within the specified FP32 tolerances.
  Maximum absolute atomic force difference: **3.71e-8 eV/Å**.
- Short deterministic NVE SIB trajectories, with transverse updates and with
  zero-temperature longitudinal glangevin updates, agree across 1/2/4 ranks.
- NVE/NVT/NPT SIB setup preserves initial forces and thermodynamic output.
- NVT/NPT execute both spin half-steps, compared with frozen-lattice NVE;
  the NPT barostat is effectively stationary in this focused check.
- New-neighbor and atom-migration reference described above passes for 1/2/4 ranks.
- Both SIB regressions explicitly fail against the pre-fix SIB implementation.
- Real single-element Fe export `Fe0702.pt` (2 layers, r_max=4.5 Å):
  1/2/4 CPU ranks, batch_size 0/5, orthogonal/triclinic static checks and short
  transverse/longitudinal NVE SIB trajectories passed.
  Maximum absolute atomic force difference: **1.85e-5 eV/Å**.
  This CUDA-traced export also exercises remapping graph Device constants to CPU.
- C++ compilation/linking (including rebuilt NVT/NPT subclasses), Python test
  execution and `git diff --check` passed.

Final fixture regression artifacts on the review machine:
`/tmp/step-review/cpu-baseline/suite-i1snq4yd/`.
Earlier real-model artifacts are under `/tmp/step-review/cpu-baseline/run*/`;
these temporary directories are not part of the source distribution.

## Reproduce on another machine

From the repository root, with the updated LAMMPS executable:

```bash
python interface/USER-SPIN-STEP/tests/test_mpi.py \
  --lammps /path/to/lmp --device cpu --workdir /path/to/results
```

For a single-element Fe export, append `--model /path/to/Fe.pt`.
An optional `--baseline /path/to/old-lmp` checks that the old SIB implementation
fails the two targeted regressions (use the fixture suite, without `--model`).
Each invocation creates its own `suite-*` directory to avoid collisions.

## Limits

These are correctness checks on small systems, not multi-GPU performance or memory
benchmarks. Physical GPU placement, CUDA execution across multiple devices,
multi-node runs, long NPT trajectories with appreciable cell changes, and large
production systems remain to be validated on the target server. Per-atom virial
is not implemented by this pair style. Nonzero-temperature stochastic trajectories
are not expected to be identical across different MPI decompositions.
