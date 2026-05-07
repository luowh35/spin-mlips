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
   Shared Nose-Hoover / Parrinello-Rahman + SIB integrator base class
------------------------------------------------------------------------- */

#include "fix_nh_spin_sib.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "fix_glangevin_spin_sib.h"
#include "fix_landau_spin.h"
#include "fix_langevin_spin_sib.h"
#include "fix_precession_spin.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_spin.h"
#include "pair_spin_ml.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

enum {NOBIAS, BIAS};
enum {NONE, XYZ, XY, YZ, XZ};
enum {ISO, ANISO, TRICLINIC};

static const char cite_fix_spin_sib_nh[] =
  "fix */spin/sib command:\n\n"
  "@article{mentink2010stable,\n"
  "title={Stable and fast semi-implicit integration of the stochastic "
  "   Landau-Lifshitz equation},\n"
  "author={Mentink, J. H. and Tretyakov, M. V. and Fasolino, A. and "
  "   Katsnelson, M. I. and Rasing, T.},\n"
  "journal={Journal of Physics: Condensed Matter},\n"
  "volume={22},\n"
  "pages={176001},\n"
  "year={2010},\n"
  "doi={10.1088/0953-8984/22/17/176001}\n"
  "}\n\n"
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

static int spin_sib_nh_narg(int narg, char **arg)
{
  int stripped = narg;
  for (int iarg = 3; iarg < narg; iarg++) {
    if (strcmp(arg[iarg], "lattice") == 0) {
      if (iarg + 1 >= narg) return narg;
      stripped -= 2;
      iarg++;
    }
  }
  return stripped;
}

static char **spin_sib_nh_args(int narg, char **arg)
{
  static std::vector<char *> stripped;
  stripped.clear();
  stripped.reserve(narg);
  for (int iarg = 0; iarg < narg; iarg++) {
    if (iarg >= 3 && strcmp(arg[iarg], "lattice") == 0 && iarg + 1 < narg) {
      iarg++;
      continue;
    }
    stripped.push_back(arg[iarg]);
  }
  return stripped.data();
}

static bool pair_ml_has_longitudinal_force(PairSpinML *pair_spin_ml)
{
  return pair_spin_ml && pair_spin_ml->has_longitudinal_force();
}

FixNHSpinSIB::FixNHSpinSIB(LAMMPS *lmp, int narg, char **arg) :
  FixNH(lmp, spin_sib_nh_narg(narg, arg), spin_sib_nh_args(narg, arg)),
  dts(0.0),
  nlocal_max(0),
  npairspin(0),
  nprecspin(0),
  nlandauspin(0),
  pair_spin_ml(nullptr),
  spin_pairs(nullptr),
  lockprecessionspin(nullptr),
  locklangevinspin_sib(nullptr),
  lockglangevinspin_sib(nullptr),
  locklandauspin(nullptr),
  s_save(nullptr),
  mag_save(nullptr),
  H_par_save(nullptr),
  noise_vec(nullptr),
  noise_L_vec(nullptr),
  fm_full(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_spin_sib_nh);

  lattice_flag = 1;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "lattice") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix {} command", style);
      const std::string latarg = arg[iarg + 1];
      if ((latarg == "no") || (latarg == "off") || (latarg == "false") || (latarg == "frozen"))
        lattice_flag = 0;
      else if ((latarg == "yes") || (latarg == "on") || (latarg == "true") ||
               (latarg == "moving"))
        lattice_flag = 1;
      else
        error->all(FLERR, "Illegal fix {} command", style);
      iarg += 2;
    } else {
      iarg++;
    }
  }

  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Fix {} requires an atom map, see atom_modify", style);

  if (!atom->sp_flag) error->all(FLERR, "Fix {} requires atom/spin style", style);
}

FixNHSpinSIB::~FixNHSpinSIB()
{
  memory->destroy(s_save);
  memory->destroy(mag_save);
  memory->destroy(H_par_save);
  memory->destroy(noise_vec);
  memory->destroy(noise_L_vec);
  memory->destroy(fm_full);
  delete[] spin_pairs;
  delete[] lockprecessionspin;
  delete[] locklandauspin;
}

int FixNHSpinSIB::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  if (pre_exchange_flag) mask |= PRE_EXCHANGE;
  return mask;
}

void FixNHSpinSIB::init()
{
  FixNH::init();

  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Fix {} does not yet support rRESPA", style);

  dts = 0.5 * update->dt;

  pair_spin_ml = dynamic_cast<PairSpinML *>(force->pair_match("spin/step", 0, 0));
  if (!pair_spin_ml)
    pair_spin_ml = dynamic_cast<PairSpinML *>(force->pair_match("spin/nep", 0, 0));

  delete[] spin_pairs;
  spin_pairs = nullptr;
  npairspin = 0;

  bool is_hybrid = (force->pair_match("^hybrid", 0) != nullptr);
  if (is_hybrid) {
    for (int i = 1; force->pair_match("^spin", 0, i); i++) {
      Pair *p = force->pair_match("^spin", 0, i);
      if (dynamic_cast<PairSpinML *>(p) == nullptr) npairspin++;
    }
    if (npairspin > 0) {
      spin_pairs = new PairSpin *[npairspin];
      int count = 0;
      for (int i = 1; force->pair_match("^spin", 0, i); i++) {
        Pair *p = force->pair_match("^spin", 0, i);
        if (dynamic_cast<PairSpinML *>(p) == nullptr)
          spin_pairs[count++] = dynamic_cast<PairSpin *>(p);
      }
    }
  } else {
    Pair *p = force->pair_match("^spin", 0, 0);
    if (p && dynamic_cast<PairSpinML *>(p) == nullptr) {
      npairspin = 1;
      spin_pairs = new PairSpin *[1];
      spin_pairs[0] = dynamic_cast<PairSpin *>(p);
    }
  }

  if (!pair_spin_ml && npairspin == 0)
    error->all(FLERR, "Fix {} requires at least one spin pair style "
               "(PairSpinML or standard PairSpin)", style);

  delete[] lockprecessionspin;
  lockprecessionspin = nullptr;
  nprecspin = 0;
  for (int iforce = 0; iforce < modify->nfix; iforce++)
    if (utils::strmatch(modify->fix[iforce]->style, "^precession/spin")) nprecspin++;

  if (nprecspin > 0) {
    lockprecessionspin = new FixPrecessionSpin *[nprecspin];
    int count = 0;
    for (int iforce = 0; iforce < modify->nfix; iforce++)
      if (utils::strmatch(modify->fix[iforce]->style, "^precession/spin"))
        lockprecessionspin[count++] = dynamic_cast<FixPrecessionSpin *>(modify->fix[iforce]);
  }

  locklangevinspin_sib = nullptr;
  {
    int count = 0;
    for (int iforce = 0; iforce < modify->nfix; iforce++) {
      if (strcmp(modify->fix[iforce]->style, "langevin/spin/sib") == 0) {
        if (count > 0) error->all(FLERR, "Only one fix langevin/spin/sib is allowed");
        locklangevinspin_sib = dynamic_cast<FixLangevinSpinSIB *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  lockglangevinspin_sib = nullptr;
  {
    int count = 0;
    for (int iforce = 0; iforce < modify->nfix; iforce++) {
      if (strcmp(modify->fix[iforce]->style, "glangevin/spin/sib") == 0) {
        if (count > 0) error->all(FLERR, "Only one fix glangevin/spin/sib is allowed");
        lockglangevinspin_sib = dynamic_cast<FixGLangevinSpinSIB *>(modify->fix[iforce]);
        count++;
      }
    }
  }

  delete[] locklandauspin;
  locklandauspin = nullptr;
  nlandauspin = 0;
  for (int iforce = 0; iforce < modify->nfix; iforce++)
    if (strcmp(modify->fix[iforce]->style, "landau/spin") == 0) nlandauspin++;

  if (nlandauspin > 0) {
    locklandauspin = new FixLandauSpin *[nlandauspin];
    int count = 0;
    for (int iforce = 0; iforce < modify->nfix; iforce++)
      if (strcmp(modify->fix[iforce]->style, "landau/spin") == 0)
        locklandauspin[count++] = dynamic_cast<FixLandauSpin *>(modify->fix[iforce]);
  }

  nlocal_max = atom->nlocal;
  grow_arrays();

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix {}: SIB spin update embedded in Nose-Hoover integrator\n", style);
    utils::logmesg(lmp, "Fix {}: dts = dt/2 = {} for half-step spin updates\n", style, dts);
    if (pair_spin_ml)
      utils::logmesg(lmp, "Fix {}: PairSpinML detected (spin/step or spin/nep)\n", style);
    if (npairspin > 0)
      utils::logmesg(lmp, "Fix {}: {} standard PairSpin style(s) detected\n", style, npairspin);
    if (lattice_flag)
      utils::logmesg(lmp, "Fix {}: Lattice dynamics enabled (spin-lattice coupling)\n", style);
    else
      utils::logmesg(lmp, "Fix {}: Lattice frozen (pure spin dynamics)\n", style);
    if (lockglangevinspin_sib)
      utils::logmesg(lmp, "Fix {}: Variable-length spin mode enabled\n", style);
    if (nlandauspin > 0)
      utils::logmesg(lmp, "Fix {}: {} fix landau/spin style(s) detected\n", style, nlandauspin);
  }

  if (lockglangevinspin_sib && nlandauspin == 0 &&
      !pair_ml_has_longitudinal_force(pair_spin_ml))
    error->warning(FLERR, "Fix {}: glangevin/spin/sib detected but no longitudinal "
                   "driving force is available. The active PairSpinML style does not "
                   "provide distribute_full_mag_forces(), and no fix landau/spin was "
                   "found. Longitudinal dynamics has no driving force.", style);
}

void FixNHSpinSIB::setup(int vflag)
{
  FixNH::setup(vflag);
  if (pair_spin_ml) pair_spin_ml->compute(1, 1);
}

void FixNHSpinSIB::grow_arrays()
{
  if (atom->nlocal > nlocal_max) nlocal_max = atom->nlocal + 100;
  memory->grow(s_save, nlocal_max, 3, "fix_nh_spin_sib:s_save");
  memory->grow(mag_save, nlocal_max, "fix_nh_spin_sib:mag_save");
  memory->grow(H_par_save, nlocal_max, "fix_nh_spin_sib:H_par_save");
  memory->grow(noise_vec, nlocal_max, 3, "fix_nh_spin_sib:noise_vec");
  memory->grow(noise_L_vec, nlocal_max, "fix_nh_spin_sib:noise_L_vec");
  memory->grow(fm_full, nlocal_max, 3, "fix_nh_spin_sib:fm_full");
}

void FixNHSpinSIB::initial_integrate(int /*vflag*/)
{
  if (pstat_flag && mpchain) nhc_press_integrate();

  if (tstat_flag) {
    compute_temp_target();
    nhc_temp_integrate();
  }

  if (pstat_flag) {
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep + 1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  if (lattice_flag) nve_v();

  sib_spin_half_step();

  if (pstat_flag) remap();
  if (lattice_flag) nve_x();
  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }

  sib_spin_half_step();
}

void FixNHSpinSIB::final_integrate()
{
  if (lattice_flag) nve_v();

  if (which == BIAS && neighbor->ago == 0) t_current = temperature->compute_scalar();

  if (pstat_flag) nh_v_press();

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO)
      pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep + 1);
  }

  if (pstat_flag) nh_omega_dot();

  if (tstat_flag) nhc_temp_integrate();
  if (pstat_flag && mpchain) nhc_press_integrate();
}

void FixNHSpinSIB::sib_spin_half_step()
{
  double **sp = atom->sp;
  double **fm = atom->fm;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *mask = atom->mask;

  if (nlocal > nlocal_max) grow_arrays();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      s_save[i][0] = sp[i][0];
      s_save[i][1] = sp[i][1];
      s_save[i][2] = sp[i][2];
      mag_save[i] = sp[i][3];
      H_par_save[i] = 0.0;
      noise_vec[i][0] = 0.0;
      noise_vec[i][1] = 0.0;
      noise_vec[i][2] = 0.0;
      noise_L_vec[i] = 0.0;
    }
  }

  comm->forward_comm();
  if (pair_spin_ml) pair_spin_ml->recompute_forces();
  distribute_magnetic_forces();

  if (lockglangevinspin_sib) {
    for (int i = 0; i < nlocal; i++)
      fm_full[i][0] = fm_full[i][1] = fm_full[i][2] = 0.0;
    if (pair_spin_ml) pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);
    for (int k = 0; k < nlandauspin; k++)
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          locklandauspin[k]->compute_single_landau(i, sp[i][3], fm_full[i]);
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (lockglangevinspin_sib) {
        double spi_L[3] = {sp[i][0], sp[i][1], sp[i][2]};
        lockglangevinspin_sib->compute_longitudinal_predictor(
            i, spi_L, fm_full[i], H_par_save[i], dts);
      }

      double spi[3] = {sp[i][0], sp[i][1], sp[i][2]};
      double fmi[3] = {fm[i][0], fm[i][1], fm[i][2]};

      for (int k = 0; k < nprecspin; k++)
        lockprecessionspin[k]->compute_single_precession(i, spi, fmi);

      if (locklangevinspin_sib)
        locklangevinspin_sib->compute_single_langevin_store_noise(i, spi, fmi, noise_vec[i]);
      if (lockglangevinspin_sib)
        lockglangevinspin_sib->compute_single_langevin_store_noise(i, spi, fmi, noise_vec[i]);

      double F_pred[3] = {fmi[0] * dts, fmi[1] * dts, fmi[2] * dts};
      double s_pred[3];
      solve_implicit_sib(s_save[i], F_pred, s_pred);

      double s_mid[3];
      s_mid[0] = 0.5 * (s_save[i][0] + s_pred[0]);
      s_mid[1] = 0.5 * (s_save[i][1] + s_pred[1]);
      s_mid[2] = 0.5 * (s_save[i][2] + s_pred[2]);

      sp[i][0] = s_mid[0];
      sp[i][1] = s_mid[1];
      sp[i][2] = s_mid[2];
    }
  }

  comm->forward_comm();
  if (pair_spin_ml) pair_spin_ml->recompute_forces();
  distribute_magnetic_forces();

  if (lockglangevinspin_sib) {
    for (int i = 0; i < nlocal; i++)
      fm_full[i][0] = fm_full[i][1] = fm_full[i][2] = 0.0;
    if (pair_spin_ml) pair_spin_ml->distribute_full_mag_forces(fm_full, nlocal);
    for (int k = 0; k < nlandauspin; k++)
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          locklandauspin[k]->compute_single_landau(i, sp[i][3], fm_full[i]);
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double spi[3] = {sp[i][0], sp[i][1], sp[i][2]};
      double fmi[3] = {fm[i][0], fm[i][1], fm[i][2]};

      for (int k = 0; k < nprecspin; k++)
        lockprecessionspin[k]->compute_single_precession(i, spi, fmi);

      if (locklangevinspin_sib)
        locklangevinspin_sib->compute_single_langevin_reuse_noise(i, spi, fmi, noise_vec[i]);
      if (lockglangevinspin_sib)
        lockglangevinspin_sib->compute_single_langevin_reuse_noise(i, spi, fmi, noise_vec[i]);

      double F_corr[3] = {fmi[0] * dts, fmi[1] * dts, fmi[2] * dts};
      double s_new[3];
      solve_implicit_sib(s_save[i], F_corr, s_new);

      fm[i][0] = fmi[0];
      fm[i][1] = fmi[1];
      fm[i][2] = fmi[2];

      sp[i][0] = s_new[0];
      sp[i][1] = s_new[1];
      sp[i][2] = s_new[2];

      if (lockglangevinspin_sib) {
        lockglangevinspin_sib->compute_longitudinal_corrector(
            i, spi, fm_full[i], H_par_save[i], mag_save[i], noise_L_vec[i], dts);
      }
    }
  }

  comm->forward_comm();
}

void FixNHSpinSIB::solve_implicit_sib(double *s_in, double *F_vec, double *s_out)
{
  double Fx = F_vec[0];
  double Fy = F_vec[1];
  double Fz = F_vec[2];

  double sx = s_in[0];
  double sy = s_in[1];
  double sz = s_in[2];

  double F_sq = Fx * Fx + Fy * Fy + Fz * Fz;
  double denom = 1.0 + 0.25 * F_sq;

  double cross_x = sy * Fz - sz * Fy;
  double cross_y = sz * Fx - sx * Fz;
  double cross_z = sx * Fy - sy * Fx;

  double dot_FX = Fx * sx + Fy * sy + Fz * sz;

  double Mx = (sx - 0.5 * cross_x + 0.25 * dot_FX * Fx) / denom;
  double My = (sy - 0.5 * cross_y + 0.25 * dot_FX * Fy) / denom;
  double Mz = (sz - 0.5 * cross_z + 0.25 * dot_FX * Fz) / denom;

  s_out[0] = 2.0 * Mx - sx;
  s_out[1] = 2.0 * My - sy;
  s_out[2] = 2.0 * Mz - sz;

  double norm = sqrt(s_out[0] * s_out[0] + s_out[1] * s_out[1] + s_out[2] * s_out[2]);
  if (norm > 1e-10) {
    s_out[0] /= norm;
    s_out[1] /= norm;
    s_out[2] /= norm;
  }
}

void FixNHSpinSIB::distribute_magnetic_forces()
{
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

  if (pair_spin_ml) pair_spin_ml->distribute_cached_mag_forces();

  for (int k = 0; k < npairspin; k++)
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) spin_pairs[k]->compute_single_pair(i, fm[i]);
}
