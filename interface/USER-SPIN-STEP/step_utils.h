/* -*- c++ -*- ----------------------------------------------------------
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
   SPIN-STEP Utility Functions
   Utility functions for SPIN-STEP (E3nn Magnetic Atomic SPIN) potential
------------------------------------------------------------------------- */

#ifndef LMP_STEP_UTILS_H
#define LMP_STEP_UTILS_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace LAMMPS_NS {
namespace step {

// =============================================================================
// Element to Atomic Number Conversion
// =============================================================================

/**
 * @brief Convert element symbol to atomic number
 * @param elem Element symbol (e.g., "Fe", "Co", "Ni")
 * @return Atomic number (e.g., 26 for Fe)
 */
int element_to_number(const std::string& elem);

/**
 * @brief Convert atomic number to element symbol
 * @param z Atomic number
 * @return Element symbol
 */
std::string number_to_element(int z);

// =============================================================================
// Magnetic Force Projection
// =============================================================================

/**
 * @brief Project forces to be perpendicular to magnetic moments
 *
 * This is crucial for spin dynamics (LLG equation) to prevent
 * changes in magnetic moment magnitude.
 *
 * Formula: F_perp = F - (F · M_hat) * M_hat
 *
 * @param forces Raw magnetic forces [N, 3]
 * @param magmoms Magnetic moments [N, 3]
 * @param epsilon Small value for numerical stability
 * @return Projected forces [N, 3]
 */
torch::Tensor project_forces_perpendicular(
    const torch::Tensor& forces,
    const torch::Tensor& magmoms,
    float epsilon = 1e-12f);

// =============================================================================
// JSON Configuration Parsing
// =============================================================================

/**
 * @brief Extract float value from JSON string
 * @param json JSON string
 * @param key Key to search for
 * @param default_val Default value if key not found
 * @return Extracted float value
 */
float extract_float(const std::string& json, const std::string& key, float default_val);

/**
 * @brief Extract integer value from JSON string
 * @param json JSON string
 * @param key Key to search for
 * @param default_val Default value if key not found
 * @return Extracted integer value
 */
int extract_int(const std::string& json, const std::string& key, int default_val);

/**
 * @brief Extract boolean value from JSON string
 * @param json JSON string
 * @param key Key to search for
 * @param default_val Default value if key not found
 * @return Extracted boolean value
 */
bool extract_bool(const std::string& json, const std::string& key, bool default_val);

/**
 * @brief Extract string value from JSON string
 * @param json JSON string
 * @param key Key to search for
 * @param default_val Default value if key not found
 * @return Extracted string value
 */
std::string extract_string(const std::string& json, const std::string& key,
                           const std::string& default_val);

/**
 * @brief Extract array of strings from JSON
 * @param json JSON string
 * @param key Key to search for
 * @return Vector of strings
 */
std::vector<std::string> extract_string_array(const std::string& json, const std::string& key);

/**
 * @brief Extract array of integers from JSON
 * @param json JSON string
 * @param key Key to search for
 * @return Vector of integers
 */
std::vector<int> extract_int_array(const std::string& json, const std::string& key);

/**
 * @brief Extract atom_types_map from JSON (maps atomic number to type index)
 * @param json JSON string
 * @return Map from atomic number string to type index
 */
std::unordered_map<int, int> extract_atom_types_map(const std::string& json);

// =============================================================================
// Neighbor List Structure
// =============================================================================

/**
 * @brief Structure to hold neighbor list data
 */
struct NeighborListData {
  torch::Tensor edge_index;  // [2, n_edges] - (dst, src) pairs
  torch::Tensor shifts;      // [n_edges, 3] - periodic shift vectors
  int n_pairs;               // Number of neighbor pairs
};

// =============================================================================
// NaN Handling
// =============================================================================

/**
 * @brief Check if tensor contains NaN values
 * @param tensor Input tensor
 * @return True if any NaN values found
 */
bool has_nan(const torch::Tensor& tensor);

/**
 * @brief Replace NaN values with cached values or zeros
 * @param tensor Input tensor (modified in place)
 * @param cache Cached valid values (optional)
 * @return Tensor with NaN values replaced
 */
torch::Tensor replace_nan(const torch::Tensor& tensor,
                          const torch::Tensor& cache = torch::Tensor());

}  // namespace step
}  // namespace LAMMPS_NS

#endif  // LMP_STEP_UTILS_H
