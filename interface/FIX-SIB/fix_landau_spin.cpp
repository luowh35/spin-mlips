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
   On-site Landau potential for variable-length spin dynamics:
     E_L(m) = sum_k a_{n_k} * m^{n_k}   (n_k = 2, 4, 6, 8, ...)
     -dE/d|m| = -sum_k n_k * a_{n_k} * m^{n_k - 1}

   Usage:
     fix ID group landau/spin a2 0.5 a4 -0.3 a6 0.1
------------------------------------------------------------------------- */

#include "fix_landau_spin.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixLandauSpin::FixLandauSpin(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  nterms(0), powers(nullptr), coeffs(nullptr), landau_energy(0.0)
{
  if (narg < 5)
    error->all(FLERR, "Illegal fix landau/spin command: need at least one a<n> coefficient");

  // Enable scalar output for thermo
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;

  // Parse keyword-value pairs: a2 val a4 val a6 val ...
  std::vector<int> pw;
  std::vector<double> cf;

  int iarg = 3;
  while (iarg < narg) {
    if (arg[iarg][0] != 'a')
      error->all(FLERR, "Illegal fix landau/spin command: expected keyword a<n>");

    int n = atoi(&arg[iarg][1]);
    if (n < 2 || n % 2 != 0)
      error->all(FLERR, "Illegal fix landau/spin command: power must be even >= 2");

    if (iarg + 1 >= narg)
      error->all(FLERR, "Illegal fix landau/spin command: missing value after {}", arg[iarg]);

    double val = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
    pw.push_back(n);
    cf.push_back(val);
    iarg += 2;
  }

  nterms = pw.size();
  powers = new int[nterms];
  coeffs = new double[nterms];
  for (int k = 0; k < nterms; k++) {
    powers[k] = pw[k];
    coeffs[k] = cf[k];
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "Fix landau/spin: {} terms\n", nterms);
    for (int k = 0; k < nterms; k++)
      utils::logmesg(lmp, "  a{} = {}\n", powers[k], coeffs[k]);
  }
}

/* ---------------------------------------------------------------------- */

FixLandauSpin::~FixLandauSpin()
{
  delete[] powers;
  delete[] coeffs;
}

/* ---------------------------------------------------------------------- */

int FixLandauSpin::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLandauSpin::post_force(int /*vflag*/)
{
  double **sp = atom->sp;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  double elocal = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      elocal += compute_single_landau_energy(sp[i][3]);
    }
  }
  landau_energy = elocal;
}

/* ---------------------------------------------------------------------- */

double FixLandauSpin::compute_scalar()
{
  double eall;
  MPI_Allreduce(&landau_energy, &eall, 1, MPI_DOUBLE, MPI_SUM, world);
  return eall;
}

/* ----------------------------------------------------------------------
   Compute longitudinal effective field H_L = -dE/d|m| and accumulate
   into fm_full_i as H_L * s_hat (unit spin direction).
------------------------------------------------------------------------- */

void FixLandauSpin::compute_single_landau(int i, double mag, double *fm_full_i)
{
  double **sp = atom->sp;

  // H_L = -dE/d|m| = -sum_k n_k * a_{n_k} * m^{n_k - 1}
  double H_L = 0.0;
  for (int k = 0; k < nterms; k++) {
    int n = powers[k];
    H_L -= n * coeffs[k] * pow(mag, n - 1);
  }

  // Accumulate as H_L * s_hat (unit spin direction vector)
  fm_full_i[0] += H_L * sp[i][0];
  fm_full_i[1] += H_L * sp[i][1];
  fm_full_i[2] += H_L * sp[i][2];
}

/* ---------------------------------------------------------------------- */

double FixLandauSpin::compute_single_landau_energy(double mag)
{
  double e = 0.0;
  for (int k = 0; k < nterms; k++) {
    e += coeffs[k] * pow(mag, powers[k]);
  }
  return e;
}
