#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include "nep_types.h"
#include <torch/torch.h>

namespace nep {

/**
 * Neighbor list builder
 * Emulates ASE's build_neighbor_list functionality
 */
class NeighborListBuilder {
public:
    /**
     * Build neighbor list
     * @param positions Atomic positions [N, 3]
     * @param cell Cell matrix [3, 3]
     * @param pbc Periodic boundary conditions [3]
     * @param cutoff Cutoff radius
     * @param self_interaction Whether to include self-interaction
     * @param bothways Whether to include symmetric neighbor pairs
     * @return Neighbor list
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
     * Compute cell image shifts
     */
    static std::vector<std::vector<int>> get_cell_shifts(const torch::Tensor& pbc);

    /**
     * Compute minimum image distance between two atoms (considering PBC)
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
