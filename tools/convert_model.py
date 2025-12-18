#!/usr/bin/env python3
"""
模型转换脚本：将PyTorch模型转换为TorchScript格式供C++使用
"""

import sys
import os
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# 确保可以导入pre.py中的类
sys.path.insert(0, os.path.dirname(__file__))
from pre import MagneticNEP, CONFIG, MagneticACEDescriptor

def convert_model(model_path, output_path, device='cpu'):
    """
    将PyTorch模型转换为TorchScript

    Args:
        model_path: 输入的.pth模型路径
        output_path: 输出的.pt TorchScript模型路径
        device: 运行设备
    """
    print(f"Starting model conversion...")
    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")
    print(f"Device: {device}")

    # 1. 初始化描述符获取输入维度
    print("\n[1/5] Initializing descriptor to get input dimension...")
    desc_gen = MagneticACEDescriptor(**CONFIG["descriptor"])
    input_dim = desc_gen.descriptor_dimension
    print(f"  Descriptor dimension: {input_dim}")

    # 2. 创建模型实例
    print("\n[2/5] Creating model instance...")
    model = MagneticNEP(
        input_dim=input_dim,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        dropout_rate=CONFIG["model"]["dropout_rate"]
    )
    print(f"  Model architecture: {CONFIG['model']['hidden_dim']}-dim hidden layer")

    # 3. 加载训练好的权重
    print("\n[3/5] Loading trained weights...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 设置为评估模式
    print("  Weights loaded successfully")

    # 4. 转换为TorchScript
    print("\n[4/5] Converting to TorchScript...")

    # 创建示例输入（批次大小为10）
    dummy_input = torch.randn(10, input_dim, device=device)

    try:
        # 方法1：trace模式（推荐，更稳定）
        print("  Using torch.jit.trace mode...")
        traced_model = torch.jit.trace(model, dummy_input)
        print("  Tracing successful")
    except Exception as e:
        print(f"  Tracing failed: {e}")
        print("  Trying torch.jit.script mode...")
        try:
            # 方法2：script模式（备选）
            traced_model = torch.jit.script(model)
            print("  Scripting successful")
        except Exception as e2:
            print(f"  Scripting also failed: {e2}")
            raise RuntimeError("Both trace and script modes failed")

    # 5. 保存TorchScript模型
    print("\n[5/5] Saving TorchScript model...")
    traced_model.save(output_path)
    print(f"  Model saved to: {output_path}")

    # 6. 验证转换
    print("\n[Verification] Testing converted model...")
    test_input = torch.randn(5, input_dim, device=device)

    # Python模型输出
    with torch.no_grad():
        python_output = model(test_input)

    # TorchScript模型输出
    with torch.no_grad():
        script_output = traced_model(test_input)

    # 比较输出
    max_diff = torch.max(torch.abs(python_output - script_output)).item()
    print(f"  Max difference between Python and TorchScript: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✓ Conversion verified successfully!")
    else:
        print(f"  ⚠ Warning: Large difference detected ({max_diff:.2e})")

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nYou can now use the model in C++ with LibTorch:")
    print(f"  torch::jit::load(\"{output_path}\")")

    return traced_model

def main():
    """主函数"""
    # 配置路径
    model_path = "best_model.pth"
    output_path = "./best_model_new.pt"

    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 执行转换
    try:
        convert_model(model_path, output_path, device)
        return 0
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
