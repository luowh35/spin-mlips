#ifndef NEP_TYPES_H
#define NEP_TYPES_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>
#include <tuple>

namespace nep {

// Neighbor list data structure
struct NeighborList {
    torch::Tensor center_indices;    // [M] Central atom indices
    torch::Tensor neighbor_indices;  // [M] Neighbor atom indices
    torch::Tensor shifts;            // [M, 3] Cell shifts
    torch::Tensor batch_idx;         // [N] Batch index (all 0 for single frame)
    int n_pairs;                     // Number of neighbor pairs

    NeighborList() : n_pairs(0) {}
};

// Descriptor configuration parameters
struct DescriptorConfig {
    std::vector<std::string> elements;
    float rc = 5.0f;                    // Cutoff radius
    int n_max = 3;                      // Radial basis function order
    int l_max = 4;                      // Angular basis function order
    int nu_max = 2;                     // Correlation order
    float m_cut = 4.0f;                 // Magnetic moment cutoff
    bool use_spin_invariants = true;    // Whether to use spin invariants
    float pos_scale = 1.0f;             // Position scaling factor
    float spin_scale = 1.0f;            // Spin scaling factor
    float epsilon = 1e-7f;              // Numerical stability constant
    float mag_noise_threshold = 0.35f;  // Magnetic moment noise threshold
    float pos_noise_threshold = 1e-8f;  // Position noise threshold
    float pole_threshold = 1e-6f;       // Pole determination threshold
};

// Clebsch-Gordan coefficient term
struct CGTerm {
    int m1, m2, m3;  // Quantum numbers
    float coeff;     // Coefficient value

    CGTerm(int m1_, int m2_, int m3_, float c)
        : m1(m1_), m2(m2_), m3(m3_), coeff(c) {}
};

// Utility function: convert element symbol to atomic number
inline int element_to_number(const std::string& element) {
    static const std::map<std::string, int> element_map = {
        {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},
        {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
        {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},
        {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20},
        {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25},
        {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
        {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
        {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40},
        {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45},
        {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50},
        {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55},
        {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}
    };

    auto it = element_map.find(element);
    if (it != element_map.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown element: " + element);
}

} // namespace nep

#endif // NEP_TYPES_H
