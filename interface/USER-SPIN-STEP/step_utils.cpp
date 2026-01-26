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
   SPIN-STEP Utility Functions Implementation
   Utility functions for SPIN-STEP (E3nn Magnetic Atomic SPIN) potential
------------------------------------------------------------------------- */

#include "step_utils.h"

#include <regex>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace LAMMPS_NS {
namespace step {

// =============================================================================
// Element to Atomic Number Conversion
// =============================================================================

// Static map for element symbols to atomic numbers
static const std::unordered_map<std::string, int> ELEMENT_TO_Z = {
    {"H", 1},   {"He", 2},  {"Li", 3},  {"Be", 4},  {"B", 5},
    {"C", 6},   {"N", 7},   {"O", 8},   {"F", 9},   {"Ne", 10},
    {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},
    {"S", 16},  {"Cl", 17}, {"Ar", 18}, {"K", 19},  {"Ca", 20},
    {"Sc", 21}, {"Ti", 22}, {"V", 23},  {"Cr", 24}, {"Mn", 25},
    {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
    {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
    {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39},  {"Zr", 40},
    {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45},
    {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50},
    {"Sb", 51}, {"Te", 52}, {"I", 53},  {"Xe", 54}, {"Cs", 55},
    {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60},
    {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65},
    {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70},
    {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74},  {"Re", 75},
    {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80},
    {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85},
    {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90},
    {"Pa", 91}, {"U", 92}
};

// Static map for atomic numbers to element symbols
static const std::unordered_map<int, std::string> Z_TO_ELEMENT = {
    {1, "H"},   {2, "He"},  {3, "Li"},  {4, "Be"},  {5, "B"},
    {6, "C"},   {7, "N"},   {8, "O"},   {9, "F"},   {10, "Ne"},
    {11, "Na"}, {12, "Mg"}, {13, "Al"}, {14, "Si"}, {15, "P"},
    {16, "S"},  {17, "Cl"}, {18, "Ar"}, {19, "K"},  {20, "Ca"},
    {21, "Sc"}, {22, "Ti"}, {23, "V"},  {24, "Cr"}, {25, "Mn"},
    {26, "Fe"}, {27, "Co"}, {28, "Ni"}, {29, "Cu"}, {30, "Zn"},
    {31, "Ga"}, {32, "Ge"}, {33, "As"}, {34, "Se"}, {35, "Br"},
    {36, "Kr"}, {37, "Rb"}, {38, "Sr"}, {39, "Y"},  {40, "Zr"},
    {41, "Nb"}, {42, "Mo"}, {43, "Tc"}, {44, "Ru"}, {45, "Rh"},
    {46, "Pd"}, {47, "Ag"}, {48, "Cd"}, {49, "In"}, {50, "Sn"},
    {51, "Sb"}, {52, "Te"}, {53, "I"},  {54, "Xe"}, {55, "Cs"},
    {56, "Ba"}, {57, "La"}, {58, "Ce"}, {59, "Pr"}, {60, "Nd"},
    {61, "Pm"}, {62, "Sm"}, {63, "Eu"}, {64, "Gd"}, {65, "Tb"},
    {66, "Dy"}, {67, "Ho"}, {68, "Er"}, {69, "Tm"}, {70, "Yb"},
    {71, "Lu"}, {72, "Hf"}, {73, "Ta"}, {74, "W"},  {75, "Re"},
    {76, "Os"}, {77, "Ir"}, {78, "Pt"}, {79, "Au"}, {80, "Hg"},
    {81, "Tl"}, {82, "Pb"}, {83, "Bi"}, {84, "Po"}, {85, "At"},
    {86, "Rn"}, {87, "Fr"}, {88, "Ra"}, {89, "Ac"}, {90, "Th"},
    {91, "Pa"}, {92, "U"}
};

int element_to_number(const std::string& elem) {
  // Normalize element symbol (capitalize first letter, lowercase rest)
  std::string normalized = elem;
  if (!normalized.empty()) {
    normalized[0] = std::toupper(normalized[0]);
    for (size_t i = 1; i < normalized.size(); ++i) {
      normalized[i] = std::tolower(normalized[i]);
    }
  }

  auto it = ELEMENT_TO_Z.find(normalized);
  if (it != ELEMENT_TO_Z.end()) {
    return it->second;
  }
  return -1;  // Unknown element
}

std::string number_to_element(int z) {
  auto it = Z_TO_ELEMENT.find(z);
  if (it != Z_TO_ELEMENT.end()) {
    return it->second;
  }
  return "X";  // Unknown element
}

// =============================================================================
// Magnetic Force Projection
// =============================================================================

torch::Tensor project_forces_perpendicular(
    const torch::Tensor& forces,
    const torch::Tensor& magmoms,
    float epsilon) {

  // Handle empty tensors
  if (forces.numel() == 0 || magmoms.numel() == 0) {
    return forces;
  }

  // Calculate magnetic moment magnitudes
  auto m_norms = magmoms.norm(2, /*dim=*/-1, /*keepdim=*/true);

  // Safe normalization (prevent division by zero)
  auto m_norms_safe = m_norms.clamp_min(epsilon);
  auto m_unit = magmoms / m_norms_safe;

  // Calculate parallel component: (F · M_hat) * M_hat
  auto dot_product = (forces * m_unit).sum(/*dim=*/-1, /*keepdim=*/true);
  auto f_parallel = dot_product * m_unit;

  // Subtract parallel component to get perpendicular component
  auto f_perp = forces - f_parallel;

  // Zero out forces for atoms with negligible magnetic moment
  auto mask = (m_norms > epsilon).to(forces.dtype());
  return f_perp * mask;
}

// =============================================================================
// JSON Configuration Parsing
// =============================================================================

float extract_float(const std::string& json, const std::string& key, float default_val) {
  std::regex pattern("\"" + key + "\"\\s*:\\s*([\\d.eE+-]+)");
  std::smatch match;
  if (std::regex_search(json, match, pattern)) {
    try {
      return std::stof(match[1].str());
    } catch (...) {
      return default_val;
    }
  }
  return default_val;
}

int extract_int(const std::string& json, const std::string& key, int default_val) {
  std::regex pattern("\"" + key + "\"\\s*:\\s*(-?\\d+)");
  std::smatch match;
  if (std::regex_search(json, match, pattern)) {
    try {
      return std::stoi(match[1].str());
    } catch (...) {
      return default_val;
    }
  }
  return default_val;
}

bool extract_bool(const std::string& json, const std::string& key, bool default_val) {
  std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (std::regex_search(json, match, pattern)) {
    return match[1].str() == "true";
  }
  return default_val;
}

std::string extract_string(const std::string& json, const std::string& key,
                           const std::string& default_val) {
  std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch match;
  if (std::regex_search(json, match, pattern)) {
    return match[1].str();
  }
  return default_val;
}

std::vector<std::string> extract_string_array(const std::string& json, const std::string& key) {
  std::vector<std::string> result;
  std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]");
  std::smatch match;
  if (std::regex_search(json, match, pattern)) {
    std::string arr = match[1].str();
    std::regex elem_pattern("\"([^\"]+)\"");
    std::sregex_iterator iter(arr.begin(), arr.end(), elem_pattern);
    std::sregex_iterator end;
    while (iter != end) {
      result.push_back((*iter)[1].str());
      ++iter;
    }
  }
  return result;
}

std::vector<int> extract_int_array(const std::string& json, const std::string& key) {
  std::vector<int> result;
  std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]");
  std::smatch match;
  if (std::regex_search(json, match, pattern)) {
    std::string arr = match[1].str();
    std::regex elem_pattern("(-?\\d+)");
    std::sregex_iterator iter(arr.begin(), arr.end(), elem_pattern);
    std::sregex_iterator end;
    while (iter != end) {
      try {
        result.push_back(std::stoi((*iter)[1].str()));
      } catch (...) {
        // Skip invalid integers
      }
      ++iter;
    }
  }
  return result;
}

std::unordered_map<int, int> extract_atom_types_map(const std::string& json) {
  std::unordered_map<int, int> result;

  // Match atom_types_map object: {"26": 0, "27": 1, ...}
  std::regex pattern("\"atom_types_map\"\\s*:\\s*\\{([^}]+)\\}");
  std::smatch match;
  if (std::regex_search(json, match, pattern)) {
    std::string obj = match[1].str();
    // Match individual key-value pairs: "26": 0
    std::regex kv_pattern("\"(\\d+)\"\\s*:\\s*(\\d+)");
    std::sregex_iterator iter(obj.begin(), obj.end(), kv_pattern);
    std::sregex_iterator end;
    while (iter != end) {
      try {
        int z = std::stoi((*iter)[1].str());
        int idx = std::stoi((*iter)[2].str());
        result[z] = idx;
      } catch (...) {
        // Skip invalid entries
      }
      ++iter;
    }
  }
  return result;
}

// =============================================================================
// NaN Handling
// =============================================================================

bool has_nan(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return false;
  }
  return torch::any(torch::isnan(tensor)).item<bool>();
}

torch::Tensor replace_nan(const torch::Tensor& tensor, const torch::Tensor& cache) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return tensor;
  }

  auto nan_mask = torch::isnan(tensor);
  if (!torch::any(nan_mask).item<bool>()) {
    return tensor;  // No NaN values
  }

  if (cache.defined() && cache.sizes() == tensor.sizes()) {
    // Replace NaN with cached values
    return torch::where(nan_mask, cache, tensor);
  } else {
    // Replace NaN with zeros
    return torch::where(nan_mask, torch::zeros_like(tensor), tensor);
  }
}

}  // namespace step
}  // namespace LAMMPS_NS
