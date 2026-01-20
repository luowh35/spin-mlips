// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   NEP-SPIN RK4 Method with Spin-Lattice Coupling
------------------------------------------------------------------------- */

#include "fix_nve_spin_rk4.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "fix_langevin_spin_rk4.h"
#include "fix_precession_spin.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "pair_nep_spin.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_nve_spin_rk4[] =
  "fix nve/spin/rk4 command:\n\n"
  "@article{tranchida2018massively,\n"
  "title={Massively Parallel Symplectic Algorithm for Coupled Magnetic Spin "
  "   Dynamics and Molecular Dynamics},\n"
  "author={Tranchida, J and Plimpton, S J and Thibaudeau, P and Thompson, A P},\n"
  "journal={Journal of Computational Physics},\n"
  "volume={372},\n"
  "pages={406--425},\n"
  "year={2018},\n"
  "doi={10.1016/j.jcp.2018.06.042}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

FixNVESpinRK4::FixNVESpinRK4(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  pair_nep_spin(nullptr), locklangevinspin_rk4(nullptr),
  lockprecessionspin(nullptr), s_save(nullptr), k1(nullptr),
  k2(nullptr), k3(nullptr), k4(nullptr), xi(nullptr),
  reuse_noise(false)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_nve_spin_rk4);

  if (narg < 3) error->all(FLERR, "Illegal fix nve/spin/rk4 command");

  time_integrate = 1;
  lattice_flag = 1;
  nlocal_max = 0;

  // Initialize flags
  nprecspin = nlangspin_rk4 = 0;
  precession_spin_flag = 0;
  maglangevin_rk4_flag = 0;

  // Check if atom map is defined
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Fix nve/spin/rk4 requires an atom map, see atom_modify");

  // Parse optional arguments
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "lattice") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix nve/spin/rk4 command");
      const std::string latarg = arg[iarg + 1];
      if ((latarg == "no") || (latarg == "off") || (latarg == "false") || (latarg == "frozen"))
        lattice_flag = 0;
      else if ((latarg == "yes") || (latarg == "on") || (latarg == "true") || (latarg == "moving"))
        lattice_flag = 1;
      else
        error->all(FLERR, "Illegal fix nve/spin/rk4 command");
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix nve/spin/rk4 command");
    }
  }

  // Check if atom/spin style is defined
  if (!atom->sp_flag)
    error->all(FLERR, "Fix nve/spin/rk4 requires atom/spin style");
}

/* ---------------------------------------------------------------------- */

FixNVESpinRK4::~FixNVESpinRK4()
{
  memory->destroy(s_save);
  memory->destroy(k1);
  memory->destroy(k2);
  memory->destroy(k3);
  memory->destroy(k4);
  memory->destroy(xi);
  delete[] locklangevinspin_rk4;
  delete[] lockprecessionspin;
}

/* ---------------------------------------------------------------------- */

int FixNVESpinRK4::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::init()
{
  // Set timesteps
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dts = 0.5 * update->dt;

  // Find NEP-SPIN pair style
  pair_nep_spin = dynamic_cast<PairNEPSpin *>(force->pair_match("spin/nep", 0, 0));
  if (!pair_nep_spin)
    error->all(FLERR, "Fix nve/spin/rk4 requires pair_style spin/nep");

  // Find fix precession/spin styles
  int iforce;
  nprecspin = 0;
  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (utils::strmatch(modify->fix[iforce]->style, "^precession/spin")) {
      nprecspin++;
    }
  }

  if (nprecspin > 0) {
    delete[] lockprecessionspin;
    lockprecessionspin = new FixPrecessionSpin *[nprecspin];

    int count = 0;
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (utils::strmatch(modify->fix[iforce]->style, "^precession/spin")) {
        precession_spin_flag = 1;
        lockprecessionspin[count] = dynamic_cast<FixPrecessionSpin *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Find fix langevin/spin/rk4 styles
  nlangspin_rk4 = 0;
  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (strcmp(modify->fix[iforce]->style, "langevin/spin/rk4") == 0) {
      nlangspin_rk4++;
    }
  }

  if (nlangspin_rk4 > 0) {
    delete[] locklangevinspin_rk4;
    locklangevinspin_rk4 = new FixLangevinSpinRK4 *[nlangspin_rk4];

    int count = 0;
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (strcmp(modify->fix[iforce]->style, "langevin/spin/rk4") == 0) {
        maglangevin_rk4_flag = 1;
        locklangevinspin_rk4[count] = dynamic_cast<FixLangevinSpinRK4 *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  // Allocate arrays
  nlocal_max = atom->nlocal;
  grow_arrays();

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix nve/spin/rk4: RK4 method with spin-lattice coupling\n");
    utils::logmesg(lmp, "Fix nve/spin/rk4: dts = dt/2 = {} for half-step updates\n", dts);
    utils::logmesg(lmp, "Fix nve/spin/rk4: 8 NN calls per timestep (4 per half-step)\n");
    if (lattice_flag)
      utils::logmesg(lmp, "Fix nve/spin/rk4: Lattice dynamics enabled (spin-lattice coupling)\n");
    else
      utils::logmesg(lmp, "Fix nve/spin/rk4: Lattice frozen (pure spin dynamics)\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::setup(int vflag)
{
  // Initial force computation
  pair_nep_spin->compute(1, 1);
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::grow_arrays()
{
  if (atom->nlocal > nlocal_max) {
    nlocal_max = atom->nlocal + 100;
  }
  memory->grow(s_save, nlocal_max, 3, "fix_nve_spin_rk4:s_save");
  memory->grow(k1, nlocal_max, 3, "fix_nve_spin_rk4:k1");
  memory->grow(k2, nlocal_max, 3, "fix_nve_spin_rk4:k2");
  memory->grow(k3, nlocal_max, 3, "fix_nve_spin_rk4:k3");
  memory->grow(k4, nlocal_max, 3, "fix_nve_spin_rk4:k4");
  memory->grow(xi, nlocal_max, 3, "fix_nve_spin_rk4:xi");
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::initial_integrate(int /*vflag*/)
{
  double dtfm;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *type = atom->type;
  int *mask = atom->mask;

  // Grow arrays if needed
  if (nlocal > nlocal_max) grow_arrays();

  // Step 1: Velocity half-step update
  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (rmass)
          dtfm = dtf / rmass[i];
        else
          dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
  }

  // Step 2: First RK4 half-step for spin (dt/2)
  rk4_spin_half_step();

  // Step 3: Position full-step update
  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }
    }
  }

  // Step 4: Second RK4 half-step for spin (dt/2)
  rk4_spin_half_step();
}

/* ----------------------------------------------------------------------
   Perform one RK4 half-step update (dt/2) for spins

   IMPORTANT: For spin-lattice coupling, we use recompute_forces() instead
   of compute() during spin updates. This is because:
   - compute() calculates both atomic forces (f) and magnetic forces (fm)
   - recompute_forces() only calculates magnetic forces (fm)
------------------------------------------------------------------------- */

void FixNVESpinRK4::rk4_spin_half_step()
{
  double **sp = atom->sp;
  double **fm = atom->fm;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *mask = atom->mask;

  // Save current spins and clear noise
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      s_save[i][0] = sp[i][0];
      s_save[i][1] = sp[i][1];
      s_save[i][2] = sp[i][2];

      xi[i][0] = 0.0;
      xi[i][1] = 0.0;
      xi[i][2] = 0.0;
    }
  }

  // Stage k1
  prepare_thermal_field();
  reuse_noise = false;
  comm->forward_comm();
  pair_nep_spin->recompute_forces();
  distribute_magnetic_forces();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double fmi[3];
      compute_omega(i, sp[i], fmi);

      double cx = sp[i][1] * fmi[2] - sp[i][2] * fmi[1];
      double cy = sp[i][2] * fmi[0] - sp[i][0] * fmi[2];
      double cz = sp[i][0] * fmi[1] - sp[i][1] * fmi[0];

      k1[i][0] = -dts * cx;
      k1[i][1] = -dts * cy;
      k1[i][2] = -dts * cz;
    }
  }

  // Stage k2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      sp[i][0] = s_save[i][0] + 0.5 * k1[i][0];
      sp[i][1] = s_save[i][1] + 0.5 * k1[i][1];
      sp[i][2] = s_save[i][2] + 0.5 * k1[i][2];

      double norm = sqrt(sp[i][0]*sp[i][0] + sp[i][1]*sp[i][1] + sp[i][2]*sp[i][2]);
      if (norm > 1e-10) {
        sp[i][0] /= norm;
        sp[i][1] /= norm;
        sp[i][2] /= norm;
      }
    }
  }

  reuse_noise = true;
  comm->forward_comm();
  pair_nep_spin->recompute_forces();
  distribute_magnetic_forces();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double fmi[3];
      compute_omega(i, sp[i], fmi);

      double cx = sp[i][1] * fmi[2] - sp[i][2] * fmi[1];
      double cy = sp[i][2] * fmi[0] - sp[i][0] * fmi[2];
      double cz = sp[i][0] * fmi[1] - sp[i][1] * fmi[0];

      k2[i][0] = -dts * cx;
      k2[i][1] = -dts * cy;
      k2[i][2] = -dts * cz;
    }
  }

  // Stage k3
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      sp[i][0] = s_save[i][0] + 0.5 * k2[i][0];
      sp[i][1] = s_save[i][1] + 0.5 * k2[i][1];
      sp[i][2] = s_save[i][2] + 0.5 * k2[i][2];

      double norm = sqrt(sp[i][0]*sp[i][0] + sp[i][1]*sp[i][1] + sp[i][2]*sp[i][2]);
      if (norm > 1e-10) {
        sp[i][0] /= norm;
        sp[i][1] /= norm;
        sp[i][2] /= norm;
      }
    }
  }

  comm->forward_comm();
  pair_nep_spin->recompute_forces();
  distribute_magnetic_forces();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double fmi[3];
      compute_omega(i, sp[i], fmi);

      double cx = sp[i][1] * fmi[2] - sp[i][2] * fmi[1];
      double cy = sp[i][2] * fmi[0] - sp[i][0] * fmi[2];
      double cz = sp[i][0] * fmi[1] - sp[i][1] * fmi[0];

      k3[i][0] = -dts * cx;
      k3[i][1] = -dts * cy;
      k3[i][2] = -dts * cz;
    }
  }

  // Stage k4
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      sp[i][0] = s_save[i][0] + k3[i][0];
      sp[i][1] = s_save[i][1] + k3[i][1];
      sp[i][2] = s_save[i][2] + k3[i][2];

      double norm = sqrt(sp[i][0]*sp[i][0] + sp[i][1]*sp[i][1] + sp[i][2]*sp[i][2]);
      if (norm > 1e-10) {
        sp[i][0] /= norm;
        sp[i][1] /= norm;
        sp[i][2] /= norm;
      }
    }
  }

  comm->forward_comm();
  pair_nep_spin->recompute_forces();
  distribute_magnetic_forces();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double fmi[3];
      compute_omega(i, sp[i], fmi);

      double cx = sp[i][1] * fmi[2] - sp[i][2] * fmi[1];
      double cy = sp[i][2] * fmi[0] - sp[i][0] * fmi[2];
      double cz = sp[i][0] * fmi[1] - sp[i][1] * fmi[0];

      k4[i][0] = -dts * cx;
      k4[i][1] = -dts * cy;
      k4[i][2] = -dts * cz;

      // Expose the effective field (including noise and damping) for diagnostics
      fm[i][0] = fmi[0];
      fm[i][1] = fmi[1];
      fm[i][2] = fmi[2];
    }
  }

  // Final RK4 update
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      sp[i][0] = s_save[i][0]
                 + (k1[i][0] + 2.0 * k2[i][0] + 2.0 * k3[i][0] + k4[i][0]) / 6.0;
      sp[i][1] = s_save[i][1]
                 + (k1[i][1] + 2.0 * k2[i][1] + 2.0 * k3[i][1] + k4[i][1]) / 6.0;
      sp[i][2] = s_save[i][2]
                 + (k1[i][2] + 2.0 * k2[i][2] + 2.0 * k3[i][2] + k4[i][2]) / 6.0;

      double norm = sqrt(sp[i][0]*sp[i][0] + sp[i][1]*sp[i][1] + sp[i][2]*sp[i][2]);
      if (norm > 1e-10) {
        sp[i][0] /= norm;
        sp[i][1] /= norm;
        sp[i][2] /= norm;
      }
    }
  }

  // Communicate updated spins
  comm->forward_comm();
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::compute_omega(int i, double *spi, double *fmi)
{
  double **fm = atom->fm;

  fmi[0] = fm[i][0];
  fmi[1] = fm[i][1];
  fmi[2] = fm[i][2];

  if (precession_spin_flag) {
    for (int k = 0; k < nprecspin; k++) {
      lockprecessionspin[k]->compute_single_precession(i, spi, fmi);
    }
  }

  if (maglangevin_rk4_flag) {
    for (int k = 0; k < nlangspin_rk4; k++) {
      if (reuse_noise)
        locklangevinspin_rk4[k]->compute_single_langevin_reuse_noise(i, spi, fmi, xi[i]);
      else
        locklangevinspin_rk4[k]->compute_single_langevin_store_noise(i, spi, fmi, xi[i]);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::prepare_thermal_field()
{
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      xi[i][0] = 0.0;
      xi[i][1] = 0.0;
      xi[i][2] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::distribute_magnetic_forces()
{
  // Clear fm array first (it accumulates)
  double **fm = atom->fm;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      fm[i][0] = 0.0;
      fm[i][1] = 0.0;
      fm[i][2] = 0.0;
    }
  }

  pair_nep_spin->distribute_cached_mag_forces();
}

/* ---------------------------------------------------------------------- */

void FixNVESpinRK4::final_integrate()
{
  double dtfm;

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *type = atom->type;
  int *mask = atom->mask;

  // Update half v for all particles
  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (rmass)
          dtfm = dtf / rmass[i];
        else
          dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
  }
}
