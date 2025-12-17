#include "descriptor.h"
#include <iostream>
#include <stdexcept>

namespace nep {

MagneticACEDescriptor::MagneticACEDescriptor(const DescriptorConfig& config)
    : rc_(config.rc)
    , m_cut_(config.m_cut)
    , pos_scale_(config.pos_scale)
    , spin_scale_(config.spin_scale)
    , epsilon_(config.epsilon)
    , mag_noise_threshold_(config.mag_noise_threshold)
    , pos_noise_threshold_(config.pos_noise_threshold)
    , pole_threshold_(config.pole_threshold)
    , n_max_(config.n_max)
    , l_max_(config.l_max)
    , nu_max_(config.nu_max)
    , use_spin_invariants_(config.use_spin_invariants)
    , elements_(config.elements)
{
    l_phi_max_ = 2 * l_max_;

    // Initialize element mapping
    for (size_t i = 0; i < elements_.size(); ++i) {
        number_map_[element_to_number(elements_[i])] = i;
    }

    // Initialize angular dimensions
    init_angular_dimensions();

    // Calculate channels
    n_channels_j_ = n_max_ * n_max_ * elements_.size();
    n_channels_i_ = n_max_;
    n_channels_spin_ = n_max_ * elements_.size();

    // Calculate descriptor dimension
    calculate_descriptor_dimension();

    // Initialize CG coefficients
    CGCoefficients::initialize();
}

void MagneticACEDescriptor::init_angular_dimensions() {
    // Initialize angular dimensions for A (l_phi_max)
    int idx = 0;
    angular_splits_A_.clear();
    for (int l = 0; l <= l_phi_max_; ++l) {
        for (int m = -l; m <= l; ++m) {
            angular_dims_A_[{l, m}] = idx++;
        }
        angular_splits_A_.push_back(2*l + 1);
    }
    n_angular_total_A_ = idx;

    // Initialize angular dimensions for M (l_max)
    idx = 0;
    angular_splits_M_.clear();
    for (int l = 0; l <= l_max_; ++l) {
        for (int m = -l; m <= l; ++m) {
            angular_dims_M_[{l, m}] = idx++;
        }
        angular_splits_M_.push_back(2*l + 1);
    }
    n_angular_total_M_ = idx;
}

void MagneticACEDescriptor::calculate_descriptor_dimension() {
    int total_dim = 0;

    // nu=1: L=0 components of A_i and M_i
    total_dim += n_channels_j_ + n_channels_i_;

    // nu=2: Invariants
    if (nu_max_ >= 2) {
        // D_AA: Symmetric combinations, L=1 to l_phi_max (skipping L=0)
        int n_combos_AA = (n_channels_j_ * (n_channels_j_ + 1)) / 2;
        total_dim += n_combos_AA * l_phi_max_;  // l_phi_max L values (1,2,...,6)

        // D_MA: L=0 to l_max
        total_dim += n_channels_i_ * n_channels_j_ * (l_max_ + 1);

        // Spin invariants
        if (use_spin_invariants_) {
            int n_combos_SS = (n_channels_spin_ * (n_channels_spin_ + 1)) / 2;
            // D_SS: uses D_MA coefficients, so L=0 to l_max
            total_dim += n_combos_SS * (l_max_ + 1);
            // D_MS: L=0 to l_max
            total_dim += n_channels_i_ * n_channels_spin_ * (l_max_ + 1);
        }
    }

    descriptor_dimension_ = total_dim;
}

torch::Tensor MagneticACEDescriptor::compute_from_precomputed_neighbors(
    const torch::Tensor& positions,
    const torch::Tensor& numbers,
    const torch::Tensor& magmoms,
    const NeighborList& neighbors,
    const torch::Tensor& cell
) {
    int n_total_atoms = positions.size(0);
    int n_pairs = neighbors.n_pairs;
    auto device = positions.device();

    if (n_pairs == 0) {
        return torch::zeros({n_total_atoms, descriptor_dimension_},
                           torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // Handle magnetic moment dimension
    torch::Tensor magmoms_3d;
    if (magmoms.dim() == 1) {
        magmoms_3d = torch::zeros_like(positions);
        magmoms_3d.index_put_({torch::indexing::Slice(), 2}, magmoms);
    } else {
        magmoms_3d = magmoms;
    }

    // 1. Compute neighbor pair vectors
    auto pos_i = positions.index_select(0, neighbors.center_indices);
    auto pos_j = positions.index_select(0, neighbors.neighbor_indices);
    auto numbers_j = numbers.index_select(0, neighbors.neighbor_indices);

    // Handle cell
    torch::Tensor cell_for_pairs;
    if (cell.dim() == 3) {
        auto batch_idx_of_pairs = neighbors.batch_idx.index_select(0, neighbors.center_indices);
        cell_for_pairs = cell.index_select(0, batch_idx_of_pairs);
    } else {
        cell_for_pairs = cell.unsqueeze(0).expand({n_pairs, 3, 3});
    }

    // Compute Cartesian shifts
    auto cartesian_shifts = torch::einsum("ni,nij->nj", {neighbors.shifts, cell_for_pairs});

    // r_ij vectors
    auto r_ij_vec = pos_j - pos_i + cartesian_shifts;

    // 2. Compute spherical coordinates (positions)
    auto [theta_r, phi_r, is_zero_r] = MathUtils::cartesian_to_spherical(
        r_ij_vec, true, pos_noise_threshold_, mag_noise_threshold_, pole_threshold_, epsilon_
    );

    // 3. Compute spherical coordinates (neighbor magnetic moments)
    auto mag_j_3d = magmoms_3d.index_select(0, neighbors.neighbor_indices);
    auto [theta_m_j, phi_m_j, is_zero_m_j] = MathUtils::cartesian_to_spherical(
        mag_j_3d, false, pos_noise_threshold_, mag_noise_threshold_, pole_threshold_, epsilon_
    );

    // 4. Compute spherical harmonics
    auto Y_r_all = MathUtils::spherical_harmonics(l_max_, theta_r, phi_r, is_zero_r);
    auto Y_m_j_all = MathUtils::spherical_harmonics(l_max_, theta_m_j, phi_m_j, is_zero_m_j);

    // 5. Compute basis functions
    auto r_ij_dist = MathUtils::safe_norm(r_ij_vec, -1, false, epsilon_);
    auto mag_j_norm = MathUtils::safe_norm(mag_j_3d, -1, false, epsilon_);

    auto B_R = radial_basis(r_ij_dist);     // [n_max, n_pairs]
    auto B_Mj = magnetic_basis(mag_j_norm);  // [n_max, n_pairs]
    auto B_Z = species_encoding(numbers_j);  // [n_elements, n_pairs]

    // 6. Compute spherical harmonics for central atom magnetic moments
    auto [theta_m_i, phi_m_i, is_zero_m_i] = MathUtils::cartesian_to_spherical(
        magmoms_3d, false, pos_noise_threshold_, mag_noise_threshold_, pole_threshold_, epsilon_
    );
    auto Y_m_i_all = MathUtils::spherical_harmonics(l_max_, theta_m_i, phi_m_i, is_zero_m_i);

    auto mag_i_norm = MathUtils::safe_norm(magmoms_3d, -1, false, epsilon_);
    auto B_Mi = magnetic_basis(mag_i_norm);  // [n_max, n_atoms]

    // 7. Build phi tensor (non-angular part)
    // phi = B_R ⊗ B_Mj ⊗ B_Z
    auto non_angular_phi = torch::einsum("ip,jp,kp->pijk", {B_R, B_Mj, B_Z});
    non_angular_phi = non_angular_phi.reshape({n_pairs, -1}).t();  // [n_channels_j, n_pairs]

    // 8. Angular coupling
    auto angular_phi_dict = couple_phi_tensors(Y_r_all, Y_m_j_all);
    auto angular_phi = pack_tensors_to_matrix(angular_phi_dict, l_phi_max_, n_pairs);

    // phi_j = non_angular ⊗ angular
    auto phi_j_weighted = (non_angular_phi.unsqueeze(1) * angular_phi.unsqueeze(0))
                          .reshape({-1, n_pairs});

    // 9. Scatter_add aggregation to atoms
    auto A_i_flat = torch::zeros({n_channels_j_ * n_angular_total_A_, n_total_atoms},
                                 torch::TensorOptions().dtype(torch::kComplexFloat).device(device));

    A_i_flat.scatter_add_(1, neighbors.center_indices.unsqueeze(0).expand_as(phi_j_weighted),
                          phi_j_weighted);

    auto A_i = A_i_flat.reshape({n_channels_j_, n_angular_total_A_, n_total_atoms});

    // 10. Build M_i tensor
    auto angular_M = pack_tensors_to_matrix(Y_m_i_all, l_max_, n_total_atoms);
    auto M_i = B_Mi.unsqueeze(1) * angular_M.unsqueeze(0);  // [n_channels_i, n_angular_M, n_atoms]

    // 11. Unpack tensors
    auto A_i_dict = unpack_tensor(A_i, angular_splits_A_);
    auto M_i_dict = unpack_tensor(M_i, angular_splits_M_);

    // 12. Build descriptors
    std::vector<torch::Tensor> D_i_list;

    // nu=1: L=0 components
    if (A_i_dict.count(0) && A_i_dict[0].size(1) == 1) {
        D_i_list.push_back(A_i_dict[0].squeeze(1));
    }
    if (M_i_dict.count(0) && M_i_dict[0].size(1) == 1) {
        D_i_list.push_back(M_i_dict[0].squeeze(1));
    }

    // nu=2: Invariants
    if (nu_max_ >= 2) {
        auto D_AA_list = compute_invariants(A_i_dict, A_i_dict, true, true, l_phi_max_, n_total_atoms);
        auto D_MA_list = compute_invariants(M_i_dict, A_i_dict, false, false, l_max_, n_total_atoms);

        if (!D_AA_list.empty()) {
            D_i_list.push_back(pos_scale_ * torch::cat(D_AA_list, 0));
        }
        if (!D_MA_list.empty()) {
            D_i_list.push_back(pos_scale_ * torch::cat(D_MA_list, 0));
        }

        // Spin invariants
        if (use_spin_invariants_) {
            // Build S_i tensor
            std::map<int, torch::Tensor> S_i_dict;
            auto non_angular_spin = torch::einsum("ip,jp->pij", {B_Mj, B_Z});
            non_angular_spin = non_angular_spin.reshape({n_pairs, -1}).t();

            for (int l = 0; l <= l_max_; ++l) {
                int n_ang = 2*l + 1;
                auto phi_spin_l = non_angular_spin.unsqueeze(1) * Y_m_j_all.at(l).unsqueeze(0);

                auto S_flat = torch::zeros({n_channels_spin_ * n_ang, n_total_atoms},
                                          torch::TensorOptions().dtype(torch::kComplexFloat).device(device));
                S_flat.scatter_add_(1,
                                   neighbors.center_indices.unsqueeze(0).expand({n_channels_spin_ * n_ang, n_pairs}),
                                   phi_spin_l.reshape({-1, n_pairs}));

                S_i_dict[l] = S_flat.reshape({n_channels_spin_, n_ang, n_total_atoms});
            }

            auto D_SS_list = compute_invariants(S_i_dict, S_i_dict, false, true, l_max_, n_total_atoms);
            auto D_MS_list = compute_invariants(M_i_dict, S_i_dict, false, false, l_max_, n_total_atoms);

            if (!D_SS_list.empty()) {
                D_i_list.push_back(spin_scale_ * torch::cat(D_SS_list, 0));
            }
            if (!D_MS_list.empty()) {
                D_i_list.push_back(spin_scale_ * torch::cat(D_MS_list, 0));
            }
        }
    }

    if (D_i_list.empty()) {
        return torch::zeros({n_total_atoms, descriptor_dimension_},
                           torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // Concatenate all descriptors (still complex)
    auto descriptors_complex = torch::cat(D_i_list, 0).t();  // [n_atoms, descriptor_dim]

    // Take real part at the last step to preserve gradient chain
    auto descriptors = torch::real(descriptors_complex);

    return descriptors;
}

// ========== Basis Function Implementation ==========

torch::Tensor MagneticACEDescriptor::radial_basis(const torch::Tensor& r) {
    auto r_clamped = torch::clamp(r, 0.0, rc_);
    auto fc = 0.5 * (1.0 + torch::cos(M_PI * r_clamped / rc_));
    auto x = 2.0 * (r_clamped / rc_) - 1.0;
    return fc * MathUtils::chebyshev_basis(x, n_max_);
}

torch::Tensor MagneticACEDescriptor::magnetic_basis(const torch::Tensor& m) {
    auto m_clamped = torch::clamp(m, 0.0, m_cut_);
    auto x = 2.0 * (m_clamped / m_cut_) - 1.0;
    return MathUtils::chebyshev_basis(x, n_max_);
}

torch::Tensor MagneticACEDescriptor::species_encoding(const torch::Tensor& numbers) {
    int n_elem = elements_.size();
    int n_atoms = numbers.size(0);
    auto device = numbers.device();

    auto enc = torch::zeros({n_elem, n_atoms},
                           torch::TensorOptions().dtype(torch::kFloat32).device(device));

    for (const auto& pair : number_map_) {
        int atomic_num = pair.first;
        int elem_idx = pair.second;
        auto mask = (numbers == atomic_num);
        enc.index_put_({elem_idx}, mask.to(torch::kFloat32));
    }

    return enc;
}

// ========== Tensor Coupling ==========

std::map<int, torch::Tensor> MagneticACEDescriptor::couple_two_tensors(
    const std::map<int, torch::Tensor>& Y1, int l1,
    const std::map<int, torch::Tensor>& Y2, int l2
) {
    std::map<int, torch::Tensor> result;

    int n_pairs = Y1.at(l1).size(1);
    auto device = Y1.at(l1).device();

    for (int L = std::abs(l1 - l2); L <= l1 + l2; ++L) {
        if (!CGCoefficients::has_phi(l1, l2, L)) continue;

        auto& cg_terms = CGCoefficients::get_phi(l1, l2, L);

        result[L] = torch::zeros({2*L+1, n_pairs},
                                torch::TensorOptions().dtype(torch::kComplexFloat).device(device));

        for (const auto& term : cg_terms) {
            int idx1 = term.m1 + l1;
            int idx2 = term.m2 + l2;
            int idx_out = term.m3 + L;

            result[L].index_put_({idx_out},
                                result[L].index({idx_out}) +
                                term.coeff * Y1.at(l1).index({idx1}) * Y2.at(l2).index({idx2}));
        }
    }

    return result;
}

std::map<int, torch::Tensor> MagneticACEDescriptor::couple_phi_tensors(
    const std::map<int, torch::Tensor>& Y_r,
    const std::map<int, torch::Tensor>& Y_m
) {
    int n_pairs = Y_r.at(0).size(1);
    auto device = Y_r.at(0).device();

    std::map<int, torch::Tensor> phi_tensors;
    for (int l = 0; l <= l_phi_max_; ++l) {
        phi_tensors[l] = torch::zeros({2*l+1, n_pairs},
                                     torch::TensorOptions().dtype(torch::kComplexFloat).device(device));
    }

    for (int l_r = 0; l_r <= l_max_; ++l_r) {
        for (int l_m = 0; l_m <= l_max_; ++l_m) {
            auto coupled = couple_two_tensors(Y_r, l_r, Y_m, l_m);
            for (const auto& pair : coupled) {
                int L = pair.first;
                if (L <= l_phi_max_) {
                    phi_tensors[L] += pair.second;
                }
            }
        }
    }

    return phi_tensors;
}

// ========== Invariant Computation ==========

std::vector<torch::Tensor> MagneticACEDescriptor::compute_invariants(
    const std::map<int, torch::Tensor>& T1,
    const std::map<int, torch::Tensor>& T2,
    bool use_D_AA,
    bool is_self_coupling,
    int l_max_val,
    int n_atoms
) {
    std::vector<torch::Tensor> invariants;
    auto device = T1.begin()->second.device();

    // D_AA starts from L=1 (skips 0), D_MA starts from L=0
    int l_start = use_D_AA ? 1 : 0;

    for (int l = l_start; l <= l_max_val; ++l) {
        if (T1.count(l) == 0 || T2.count(l) == 0) continue;

        auto& cg_terms = use_D_AA ? CGCoefficients::get_D_AA(l, l, 0) :
                         CGCoefficients::get_D_MA(l, l, 0);

        int n_ch1 = T1.at(l).size(0);
        int n_ch2 = T2.at(l).size(0);

        auto inv = torch::zeros({n_ch1, n_ch2, n_atoms},
                               torch::TensorOptions().dtype(torch::kComplexFloat).device(device));

        for (const auto& term : cg_terms) {
            int m1_idx = term.m1 + l;
            int m2_idx = term.m2 + l;

            if (m1_idx >= T1.at(l).size(1) || m2_idx >= T2.at(l).size(1)) continue;

            auto t1 = T1.at(l).index({torch::indexing::Slice(), m1_idx, torch::indexing::Slice()});  // [n_ch1, n_atoms]
            auto t2 = T2.at(l).index({torch::indexing::Slice(), m2_idx, torch::indexing::Slice()});  // [n_ch2, n_atoms]

            // CG coupling: <T1 ⊗ T2> = Σ CG * T1 * T2
            // Consistent with Python version - no conjugation
            inv += term.coeff * t1.unsqueeze(1) * t2.unsqueeze(0);
        }

        if (is_self_coupling) {
            // Keep only upper triangular
            std::vector<int64_t> idx0_vec, idx1_vec;
            for (int i = 0; i < n_ch1; ++i) {
                for (int j = i; j < n_ch2; ++j) {
                    idx0_vec.push_back(i);
                    idx1_vec.push_back(j);
                }
            }
            auto idx0 = torch::tensor(idx0_vec, torch::TensorOptions().dtype(torch::kInt64).device(device));
            auto idx1 = torch::tensor(idx1_vec, torch::TensorOptions().dtype(torch::kInt64).device(device));
            inv = inv.index({idx0, idx1});
        } else {
            inv = inv.reshape({-1, n_atoms});
        }

        invariants.push_back(inv);
    }

    return invariants;
}

// ========== Utility Functions ==========

torch::Tensor MagneticACEDescriptor::pack_tensors_to_matrix(
    const std::map<int, torch::Tensor>& tensor_dict,
    int l_max_val,
    int n_cols
) {
    std::vector<torch::Tensor> tensors;
    auto device = tensor_dict.begin()->second.device();

    for (int l = 0; l <= l_max_val; ++l) {
        if (tensor_dict.count(l)) {
            tensors.push_back(tensor_dict.at(l));
        } else {
            tensors.push_back(torch::zeros({2*l+1, n_cols},
                                          torch::TensorOptions().dtype(torch::kComplexFloat).device(device)));
        }
    }

    return torch::cat(tensors, 0);
}

std::map<int, torch::Tensor> MagneticACEDescriptor::unpack_tensor(
    const torch::Tensor& flat,
    const std::vector<int>& splits
) {
    auto split_tensors = flat.split_with_sizes(torch::IntArrayRef(std::vector<int64_t>(splits.begin(), splits.end())), 1);
    std::map<int, torch::Tensor> result;

    for (size_t l = 0; l < splits.size(); ++l) {
        result[l] = split_tensors[l];
    }

    return result;
}

} // namespace nep
