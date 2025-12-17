#include "math_utils.h"
#include <cmath>
#include <iostream>

namespace nep {

// =============================================================================
// Chebyshev Polynomials
// =============================================================================

torch::Tensor MathUtils::chebyshev_basis(const torch::Tensor& x, int n_max) {
    if (x.dim() == 0) {
        // Scalar input
        auto x_expanded = x.unsqueeze(0);
        return chebyshev_basis(x_expanded, n_max).squeeze(1);
    }

    std::vector<torch::Tensor> T;
    T.push_back(torch::ones_like(x));  // T_0(x) = 1
    if (n_max > 1) {
        T.push_back(x);  // T_1(x) = x
    }

    // Recurrence: T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    for (int n = 2; n < n_max; ++n) {
        T.push_back(2 * x * T[n-1] - T[n-2]);
    }

    return torch::stack(T, 0);
}

// =============================================================================
// Safe Norm Computation
// =============================================================================

torch::Tensor MathUtils::safe_norm(
    const torch::Tensor& x,
    int dim,
    bool keepdim,
    float epsilon
) {
    return torch::sqrt(torch::sum(x * x, dim, keepdim) + epsilon);
}

// =============================================================================
// Cartesian to Spherical Coordinate Conversion
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MathUtils::cartesian_to_spherical(
    const torch::Tensor& vec,
    bool is_position,
    float pos_threshold,
    float mag_threshold,
    float pole_threshold,
    float epsilon
) {
    // vec: [..., 3]
    auto device = vec.device();

    // 1. Compute norm
    auto r = safe_norm(vec, -1, false, epsilon);

    // 2. Determine zero vectors
    float threshold = is_position ? pos_threshold : mag_threshold;
    auto is_zero = r < threshold;

    // 3. Determine poles (vectors close to 0 in xy plane)
    auto vec_xy = vec.index({"...", torch::indexing::Slice(0, 2)});
    auto r_xy = safe_norm(vec_xy, -1, false, epsilon);
    auto is_pole = r_xy < pole_threshold;

    // 4. Safe handling to avoid division by zero
    auto safe_mask = is_zero | is_pole;

    auto vx = torch::where(safe_mask,
                           torch::ones({1}, vec.options()),
                           vec.index({"...", 0}));
    auto vy = torch::where(safe_mask,
                           torch::zeros({1}, vec.options()),
                           vec.index({"...", 1}));
    auto vz = torch::where(safe_mask,
                           torch::zeros({1}, vec.options()),
                           vec.index({"...", 2}));

    // 5. Compute r_xy (for theta)
    auto vx_sq = vx * vx;
    auto vy_sq = vy * vy;
    r_xy = torch::sqrt(vx_sq + vy_sq + epsilon);

    // 6. Compute theta and phi
    auto theta = torch::atan2(r_xy, vz);
    auto phi = torch::atan2(vy, vx);

    // 7. Set theta=0, phi=0 for zero vectors and poles
    theta = torch::where(is_zero, torch::zeros_like(theta), theta);
    phi = torch::where(is_zero | is_pole, torch::zeros_like(phi), phi);

    return std::make_tuple(theta, phi, is_zero);
}

// =============================================================================
// Associated Legendre Polynomial P_l^m(x)
// =============================================================================

std::map<std::pair<int, int>, torch::Tensor>
MathUtils::associated_legendre(int l_max, const torch::Tensor& x) {
    std::map<std::pair<int, int>, torch::Tensor> P;

    // P_0^0 = 1
    P[{0, 0}] = torch::ones_like(x);

    if (l_max == 0) return P;

    auto sin_theta = torch::sqrt(torch::clamp(1.0 - x*x, 0.0, 1.0));

    // Recursive computation
    for (int l = 1; l <= l_max; ++l) {
        // P_l^l = -(2l-1) * sin(theta) * P_{l-1}^{l-1}
        P[{l, l}] = -(2*l - 1) * sin_theta * P[{l-1, l-1}];

        // P_l^{l-1} = (2l-1) * x * P_{l-1}^{l-1}
        if (l > 0) {
            P[{l, l-1}] = (2*l - 1) * x * P[{l-1, l-1}];
        }

        // P_l^m for m < l-1
        if (l > 1) {
            for (int m = l - 2; m >= 0; --m) {
                float denominator = l - m;
                P[{l, m}] = ((2*l - 1) * x * P[{l-1, m}] - (l + m - 1) * P[{l-2, m}]) / (denominator + 1e-9);
            }
        }
    }

    return P;
}

// =============================================================================
// Spherical Harmonics Y_l^m(theta, phi)
// =============================================================================

std::map<int, torch::Tensor>
MathUtils::spherical_harmonics(
    int l_max,
    const torch::Tensor& theta,
    const torch::Tensor& phi,
    const torch::Tensor& is_zero
) {
    auto device = theta.device();
    auto x = torch::cos(theta);

    // Compute Associated Legendre polynomials
    auto P_lm = associated_legendre(l_max, x);

    // Store Y_l^m
    std::map<int, torch::Tensor> Y_lm;

    // For each l, compute all m values of Y_l^m
    for (int l = 0; l <= l_max; ++l) {
        std::vector<torch::Tensor> Y_l_components;

        for (int m = -l; m <= l; ++m) {
            int abs_m = std::abs(m);

            // Compute normalization factor
            // N_l^m = sqrt[(2l+1)/(4π) * (l-|m|)!/(l+|m|)!]
            // Use lgamma to avoid factorial overflow
            float log_fact_l_minus_m = std::lgamma(l - abs_m + 1);
            float log_fact_l_plus_m = std::lgamma(l + abs_m + 1);
            float log_norm_factor_sq = std::log(2*l + 1) + log_fact_l_minus_m
                                      - (std::log(4.0 * M_PI) + log_fact_l_plus_m);
            float norm_factor = std::exp(0.5 * log_norm_factor_sq);

            // Y_l^m = N * P_l^|m| * exp(i*m*phi)
            // exp(i*m*phi) = cos(m*phi) + i*sin(m*phi)
            auto m_phi = m * phi;
            auto exp_im_phi = torch::complex(torch::cos(m_phi), torch::sin(m_phi));

            // Convert P to complex type
            auto P_complex = torch::complex(P_lm[{l, abs_m}], torch::zeros_like(P_lm[{l, abs_m}]));

            torch::Tensor Y_lm_val;

            // For m >= 0: Y_l^m = norm * P_l^m * exp(i*m*phi)
            // For m < 0: Y_l^{-|m|} = (-1)^|m| * conj(Y_l^{|m|})
            //          = (-1)^|m| * norm * P_l^|m| * exp(-i*|m|*phi)
            // Note: exp(i*m*phi) with m<0 already gives exp(-i*|m|*phi)
            // So we only need to multiply by (-1)^|m| phase factor
            if (m >= 0) {
                Y_lm_val = norm_factor * P_complex * exp_im_phi;
            } else {
                float phase = std::pow(-1.0f, abs_m);
                Y_lm_val = phase * norm_factor * P_complex * exp_im_phi;
            }

            Y_l_components.push_back(Y_lm_val);
        }

        // Stack into [2l+1, N] shape
        Y_lm[l] = torch::stack(Y_l_components, 0);
    }

    // Apply zero vector mask
    auto mask = (~is_zero).to(torch::kComplexFloat);
    Y_lm[0] = Y_lm[0];  // Y_0^0 is always valid

    for (int l = 1; l <= l_max; ++l) {
        Y_lm[l] = Y_lm[l] * mask;
    }

    return Y_lm;
}

} // namespace nep
