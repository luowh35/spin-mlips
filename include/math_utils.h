#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <torch/torch.h>
#include <map>

namespace nep {

/**
 * 数学工具函数集合
 * 包含球谐函数、Chebyshev多项式等
 */
class MathUtils {
public:
    /**
     * 计算Chebyshev多项式基函数
     * T_0(x) = 1, T_1(x) = x, T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
     * @param x 输入tensor
     * @param n_max 最高阶数
     * @return [n_max, ...] tensor
     */
    static torch::Tensor chebyshev_basis(const torch::Tensor& x, int n_max);

    /**
     * 笛卡尔坐标转球坐标
     * @param vec 输入向量 [..., 3]
     * @param is_position 是否为位置向量（影响阈值）
     * @return (theta, phi, is_zero) 球坐标和零向量mask
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
     * 计算球谐函数 Y_l^m(theta, phi)
     * @param l_max 最大角动量
     * @param theta 极角
     * @param phi 方位角
     * @param is_zero 零向量mask
     * @return 字典 {l: tensor[2l+1, N]}
     */
    static std::map<int, torch::Tensor> spherical_harmonics(
        int l_max,
        const torch::Tensor& theta,
        const torch::Tensor& phi,
        const torch::Tensor& is_zero
    );

    /**
     * 计算Legendre多项式 P_l^m(x)
     * 使用递推关系
     */
    static std::map<std::pair<int, int>, torch::Tensor> associated_legendre(
        int l_max,
        const torch::Tensor& x
    );

    /**
     * 安全的范数计算（避免数值问题）
     */
    static torch::Tensor safe_norm(
        const torch::Tensor& x,
        int dim,
        bool keepdim,
        float epsilon
    );
};

} // namespace nep

#endif // MATH_UTILS_H
