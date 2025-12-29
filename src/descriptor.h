#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include "nep_types.h"
#include "math_utils.h"
#include "cg_coefficients.h"
#include <torch/torch.h>
#include <map>
#include <vector>

namespace nep {

/**
 * MagneticACE descriptor generator
 * Computes atomic environment descriptors with spin information
 */
class MagneticACEDescriptor {
public:
    /**
     * Constructor
     * @param config Descriptor configuration parameters
     */
    MagneticACEDescriptor(const DescriptorConfig& config);

    /**
     * Compute descriptors from precomputed neighbor list
     * @param positions Atomic positions [N, 3]
     * @param numbers Atomic numbers [N]
     * @param magmoms Magnetic moments [N, 3]
     * @param neighbors Neighbor list
     * @param cell Cell matrix [1, 3, 3] or [3, 3]
     * @return Descriptors [N, descriptor_dim]
     */
    torch::Tensor compute_from_precomputed_neighbors(
        const torch::Tensor& positions,
        const torch::Tensor& numbers,
        const torch::Tensor& magmoms,
        const NeighborList& neighbors,
        const torch::Tensor& cell
    );

    /**
     * Get descriptor dimension
     */
    int get_descriptor_dimension() const { return descriptor_dimension_; }

private:
    // Configuration parameters
    float rc_, m_cut_, pos_scale_, spin_scale_, epsilon_;
    float mag_noise_threshold_, pos_noise_threshold_, pole_threshold_;
    int n_max_, l_max_, nu_max_, l_phi_max_;
    bool use_spin_invariants_;
    std::vector<std::string> elements_;
    std::map<int, int> number_map_;

    // Dimension information
    int n_channels_j_, n_channels_i_, n_channels_spin_;
    int n_angular_total_A_, n_angular_total_M_;
    int descriptor_dimension_;
    std::vector<int> angular_splits_A_, angular_splits_M_;

    // Initialization functions
    void init_angular_dimensions();
    void calculate_descriptor_dimension();

    // Basis functions
    torch::Tensor radial_basis(const torch::Tensor& r);
    torch::Tensor magnetic_basis(const torch::Tensor& m);
    torch::Tensor species_encoding(const torch::Tensor& numbers);

    // Tensor coupling
    std::map<int, torch::Tensor> couple_two_tensors(
        const std::map<int, torch::Tensor>& Y1, int l1,
        const std::map<int, torch::Tensor>& Y2, int l2
    );

    std::map<int, torch::Tensor> couple_phi_tensors(
        const std::map<int, torch::Tensor>& Y_r,
        const std::map<int, torch::Tensor>& Y_m
    );

    // Invariant computation
    std::vector<torch::Tensor> compute_invariants(
        const std::map<int, torch::Tensor>& T1,
        const std::map<int, torch::Tensor>& T2,
        bool use_D_AA,  // true=use D_AA coefficients (skip L=0), false=use D_MA (include L=0)
        bool is_self_coupling,  // true=apply upper triangular indexing
        int l_max_val,
        int n_atoms
    );

    // Utility functions
    torch::Tensor pack_tensors_to_matrix(
        const std::map<int, torch::Tensor>& tensor_dict,
        int l_max_val,
        int n_cols
    );

    std::map<int, torch::Tensor> unpack_tensor(
        const torch::Tensor& flat,
        const std::vector<int>& splits
    );
};

} // namespace nep

#endif // DESCRIPTOR_H
