#include "model.h"
#include <iostream>
#include <stdexcept>

namespace nep {

MagneticNEPModel::MagneticNEPModel(
    const std::string& model_path,
    const torch::Device& device
) : device_(device) {
    try {
        // Load TorchScript model
        module_ = torch::jit::load(model_path);
        module_.to(device_);
        module_.eval();  // Set to evaluation mode
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

torch::Tensor MagneticNEPModel::forward(const torch::Tensor& descriptors) {
    // Ensure input is on the correct device
    auto desc_device = descriptors.to(device_);

    // Disable gradient computation (inference mode)
    torch::NoGradGuard no_grad;

    // Call model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(desc_device);

    auto output = module_.forward(inputs).toTensor();

    return output;
}

torch::Tensor MagneticNEPModel::forward_with_grad(const torch::Tensor& descriptors) {
    // Ensure input is on the correct device
    auto desc_device = descriptors.to(device_);

    // Don't disable gradient (for force calculation)
    // Call model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(desc_device);

    auto output = module_.forward(inputs).toTensor();

    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MagneticNEPModel::predict_with_gradients(
    const torch::Tensor& descriptors,
    const torch::Tensor& positions,
    const torch::Tensor& magmoms
) {
    // Note: This function assumes descriptors is already a function of positions and magmoms
    // and that positions and magmoms already have requires_grad=true

    // Forward propagation to compute energy
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(descriptors);

    auto atomic_energies = module_.forward(inputs).toTensor();
    auto total_energy = atomic_energies.sum();

    // Backpropagation
    total_energy.backward();

    // Extract gradients
    // Forces = -dE/dR
    auto forces = -positions.grad().clone();

    // Magnetic forces = dE/dM
    auto mag_forces = magmoms.grad().clone();

    return std::make_tuple(total_energy, forces, mag_forces);
}

} // namespace nep
