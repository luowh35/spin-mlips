#ifndef MODEL_H
#define MODEL_H

#include "nep_types.h"
#include <torch/script.h>
#include <torch/torch.h>

namespace nep {

/**
 * MagneticNEP模型加载器和推理器
 */
class MagneticNEPModel {
public:
    /**
     * 构造函数：加载TorchScript模型
     * @param model_path 模型文件路径
     * @param device 运行设备
     */
    MagneticNEPModel(
        const std::string& model_path,
        const torch::Device& device = torch::kCPU
    );

    /**
     * 前向传播：仅计算能量
     * @param descriptors 输入描述符 [N, descriptor_dim]
     * @return 原子能量 [N, 1]
     */
    torch::Tensor forward(const torch::Tensor& descriptors);

    /**
     * 前向传播（保留梯度）：用于力计算
     * @param descriptors 输入描述符 [N, descriptor_dim]
     * @return 原子能量 [N, 1]
     */
    torch::Tensor forward_with_grad(const torch::Tensor& descriptors);

    /**
     * 带梯度的预测：计算能量、力和磁力
     * @param descriptors 输入描述符
     * @param positions 原子位置（需要梯度）
     * @param magmoms 磁矩（需要梯度）
     * @return (total_energy, forces, mag_forces)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    predict_with_gradients(
        const torch::Tensor& descriptors,
        const torch::Tensor& positions,
        const torch::Tensor& magmoms
    );

    /**
     * 获取设备
     */
    torch::Device device() const { return device_; }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
};

} // namespace nep

#endif // MODEL_H
