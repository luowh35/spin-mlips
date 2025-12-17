#ifndef MODEL_H
#define MODEL_H

#include "nep_types.h"
#include <torch/script.h>
#include <torch/torch.h>

namespace nep {

/**
 * MagneticNEP model loader and inference engine
 */
class MagneticNEPModel {
public:
    /**
     * Constructor: load TorchScript model
     * @param model_path Model file path
     * @param device Execution device
     */
    MagneticNEPModel(
        const std::string& model_path,
        const torch::Device& device = torch::kCPU
    );

    /**
     * Forward pass: compute energy only
     * @param descriptors Input descriptors [N, descriptor_dim]
     * @return Atomic energies [N, 1]
     */
    torch::Tensor forward(const torch::Tensor& descriptors);

    /**
     * Forward pass (with gradient): for force calculation
     * @param descriptors Input descriptors [N, descriptor_dim]
     * @return Atomic energies [N, 1]
     */
    torch::Tensor forward_with_grad(const torch::Tensor& descriptors);

    /**
     * Prediction with gradients: compute energy, forces, and magnetic forces
     * @param descriptors Input descriptors
     * @param positions Atomic positions (requires gradient)
     * @param magmoms Magnetic moments (requires gradient)
     * @return (total_energy, forces, mag_forces)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    predict_with_gradients(
        const torch::Tensor& descriptors,
        const torch::Tensor& positions,
        const torch::Tensor& magmoms
    );

    /**
     * Get device
     */
    torch::Device device() const { return device_; }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
};

} // namespace nep

#endif // MODEL_H
