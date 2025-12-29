#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <torch/torch.h>
#include <map>

namespace nep {

/**
 * Mathematical utility functions collection
 * Including spherical harmonics, Chebyshev polynomials, etc.
 */
class MathUtils {
public:
    /**
     * Compute Chebyshev polynomial basis functions
     * T_0(x) = 1, T_1(x) = x, T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
     * @param x Input tensor
     * @param n_max Maximum order
     * @return [n_max, ...] tensor
     */
    static torch::Tensor chebyshev_basis(const torch::Tensor& x, int n_max);

    /**
     * Cartesian to spherical coordinates conversion
     * @param vec Input vector [..., 3]
     * @param is_position Whether it is a position vector (affects threshold)
     * @return (theta, phi, is_zero) Spherical coordinates and zero vector mask
     */
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    cartesian_to_spherical(
        const torch::Tensor& vec,
        bool is_position,
        float pos_threshold,
        float mag_threshold,
        float pole_threshold,
        float epsilon
    );

    /**
     * Compute spherical harmonics Y_l^m(theta, phi)
     * @param l_max Maximum angular momentum
     * @param theta Polar angle
     * @param phi Azimuthal angle
     * @param is_zero Zero vector mask
     * @return Dictionary {l: tensor[2l+1, N]}
     */
    static std::map<int, torch::Tensor> spherical_harmonics(
        int l_max,
        const torch::Tensor& theta,
        const torch::Tensor& phi,
        const torch::Tensor& is_zero
    );

    /**
     * Compute associated Legendre polynomials P_l^m(x)
     * Using recurrence relations
     */
    static std::map<std::pair<int, int>, torch::Tensor> associated_legendre(
        int l_max,
        const torch::Tensor& x
    );

    /**
     * Safe norm computation (avoiding numerical issues)
     */
    static torch::Tensor safe_norm(
        const torch::Tensor& x,
        int dim,
        bool keepdim,
        float epsilon
    );

    /**
     * Project forces perpendicular to magnetic moments
     * F_perp = F - (F · n) * n, where n is the unit vector of magmoms
     * @param forces Input forces tensor [N, 3]
     * @param magmoms Magnetic moments tensor [N, 3]
     * @param epsilon Small value to prevent division by zero
     * @return Projected forces tensor [N, 3]
     */
    static torch::Tensor project_forces_perpendicular(
        const torch::Tensor& forces,
        const torch::Tensor& magmoms,
        float epsilon = 1e-12f
    );
};

} // namespace nep

#endif // MATH_UTILS_H
