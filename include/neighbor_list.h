#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include "nep_types.h"
#include <torch/torch.h>

namespace nep {

/**
 * 邻居列表构建器
 * 模拟ASE的build_neighbor_list功能
 */
class NeighborListBuilder {
public:
    /**
     * 构建邻居列表
     * @param positions 原子位置 [N, 3]
     * @param cell 晶胞矩阵 [3, 3]
     * @param pbc 周期性边界条件 [3]
     * @param cutoff 截断半径
     * @param self_interaction 是否包含自相互作用
     * @param bothways 是否包含对称邻居对
     * @return 邻居列表
     */
    static NeighborList build(
        const torch::Tensor& positions,
        const torch::Tensor& cell,
        const torch::Tensor& pbc,
        float cutoff,
        bool self_interaction = false,
        bool bothways = true
    );

private:
    /**
     * 计算晶胞镜像的偏移量
     */
    static std::vector<std::vector<int>> get_cell_shifts(const torch::Tensor& pbc);

    /**
     * 计算两原子间的最短距离（考虑PBC）
     */
    static float minimum_image_distance(
        const torch::Tensor& pos_i,
        const torch::Tensor& pos_j,
        const torch::Tensor& cell,
        const std::vector<int>& shift,
        torch::Tensor& min_shift
    );
};

} // namespace nep

#endif // NEIGHBOR_LIST_H
