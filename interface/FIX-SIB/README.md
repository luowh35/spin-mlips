# USER-SPIN-ML Package

This package provides spin dynamics integrators and thermostats designed to work
with machine-learning spin potentials (e.g. STEP, NEP-SPIN) in LAMMPS. It also
supports standard `PairSpin` styles (spin/exchange, spin/dmi, etc.) and can mix
both ML and analytical pair styles in the same simulation.

All fix styles use the Semi-Implicit B (SIB) predictor-corrector method [1,2]
for unconditionally stable spin integration.

## Pair Style

### pair_style spin/step, spin/nep, ...

Any pair style that inherits from `PairSpinML` can be used. The base class
defines the interface:

- `recompute_forces()` — recompute magnetic forces at current spin configuration
- `distribute_cached_mag_forces()` — projected (transverse) forces → `fm`
- `distribute_full_mag_forces()` — unprojected forces → `fm_full` (for
  variable-length spin dynamics)

Standard `PairSpin` styles are also detected and their per-atom contributions
are accumulated automatically.

## Fix Styles

### fix nve/spin/sib

Time integrator for coupled spin-lattice dynamics using the SIB method.

```
fix ID group nve/spin/sib keyword value ...
```

Optional keywords:
- `lattice yes/no` — enable/disable lattice dynamics (default: yes).
  Use `lattice no` for pure spin dynamics with frozen atomic positions.

Algorithm per timestep:
1. Velocity half-step
2. SIB spin half-step (2 force evaluations: predictor + corrector)
3. Position full step
4. SIB spin half-step (2 force evaluations: predictor + corrector)
5. Velocity half-step (in `final_integrate`)

This fix automatically discovers and uses:
- `PairSpinML` derivatives (spin/step, spin/nep)
- Standard `PairSpin` styles (spin/exchange, spin/dmi, etc.)
- `fix precession/spin` (external magnetic field, multiple allowed)
- `fix langevin/spin/sib` (fixed-length thermostat)
- `fix glangevin/spin/sib` (variable-length thermostat)
- `fix landau/spin` (on-site Landau potential, multiple allowed)

---

### fix langevin/spin/sib

Stochastic Langevin thermostat for fixed-length spin dynamics. Spin magnitude
`|m|` is constant; only the spin direction evolves.

```
fix ID group langevin/spin/sib temp alpha_t seed
```

Parameters:
- `temp` — spin bath temperature (K)
- `alpha_t` — transverse (Gilbert) damping coefficient (dimensionless, typical: 0.01–0.1)
- `seed` — random number generator seed

The same noise realization is used in both the predictor and corrector steps of
the SIB method, ensuring correct Stratonovich interpretation.

---

### fix glangevin/spin/sib

Generalized Langevin thermostat for variable-length spin dynamics. Both spin
direction and magnitude `|m|` evolve, enabling longitudinal spin fluctuations.

```
fix ID group glangevin/spin/sib temp alpha_t gamma_L seed
```

Parameters:
- `temp` — spin bath temperature (K)
- `alpha_t` — transverse (Gilbert) damping coefficient (dimensionless, typical: 0.01–0.1)
- `gamma_L` — longitudinal mobility coefficient in `dm/dt = gamma_L * H_parallel`,
  with units `1/(eV*ps)`. For Fe, a SPILADY-consistent value is about `235.75`.
- `seed` — random number generator seed

The longitudinal dynamics uses a Heun (trapezoidal) predictor-corrector scheme:
the predictor does an Euler step in `|m|`, then the corrector averages the
parallel effective fields from both evaluations.

This fix requires a potential that provides a `|m|`-dependent energy surface to
drive the longitudinal dynamics. Use `fix landau/spin` for an on-site Landau
polynomial, or a `PairSpinML` that implements `distribute_full_mag_forces()`.

---

### fix landau/spin

On-site Landau potential for variable-length spin dynamics. Provides a
`|m|`-dependent energy landscape that drives longitudinal spin relaxation.

```
fix ID group landau/spin a2 val [a4 val] [a6 val] [a8 val] ...
```

The potential energy per atom is an even-order polynomial in spin magnitude:

```
E_L(m) = a2*m^2 + a4*m^4 + a6*m^6 + a8*m^8 + ...
```

The longitudinal effective field (accumulated into `fm_full`) is:

```
H_L = -dE/d|m| = -(2*a2*m + 4*a4*m^3 + 6*a6*m^5 + 8*a8*m^7 + ...)
```

Parameters:
- `a2`, `a4`, `a6`, `a8`, ... — coefficients for each even-order term.
  Only specified terms are included; unspecified orders default to zero.
  At least one term is required. Arbitrary even orders are supported.

The global Landau energy is accessible as a scalar for thermo output:

```
thermo_style custom step temp f_landau
```

where `landau` is the fix ID.

Multiple `fix landau/spin` instances can be used with different groups to assign
different Landau coefficients to different atom types.

## Example: Variable-Length Spin Dynamics with Landau Potential

```
atom_style    spin
pair_style    spin/step
pair_coeff    * * model.pt

# Landau potential: double-well in |m| with minimum near m ≈ 1
fix landau    all landau/spin a2 1.0 a4 -0.5

# SIB integrator (frozen lattice)
fix nve       all nve/spin/sib lattice no

# Generalized Langevin thermostat with longitudinal relaxation
fix thermo    all glangevin/spin/sib 300.0 0.05 235.75 12345

# Output
thermo_style  custom step temp f_landau
thermo        100
run           10000
```

## Example: Fixed-Length Spin Dynamics

```
atom_style    spin
pair_style    spin/step
pair_coeff    * * model.pt

fix nve       all nve/spin/sib lattice no
fix thermo    all langevin/spin/sib 300.0 0.05 12345

thermo        100
run           10000
```

## Example: Mixed ML + Analytical Pair Styles

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

## References

[1] J.H. Mentink, M.V. Tretyakov, A. Fasolino, M.I. Katsnelson, T. Rasing,
    "Stable and fast semi-implicit integration of the stochastic Landau-Lifshitz
    equation", J. Phys.: Condens. Matter 22, 176001 (2010).

[2] J. Tranchida, S.J. Plimpton, P. Thibaudeau, A.P. Thompson,
    "Massively Parallel Symplectic Algorithm for Coupled Magnetic Spin Dynamics
    and Molecular Dynamics", J. Comput. Phys. 372, 406–425 (2018).

[3] P.-W. Ma, S.L. Dudarev, "Longitudinal magnetic fluctuations in Langevin
    spin dynamics", Phys. Rev. B 86, 054416 (2012).
