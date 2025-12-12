#ifndef NEP_TYPES_H
#define NEP_TYPES_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>
#include <tuple>

namespace nep {

// 原子系统数据结构
struct AtomicSystem {
    torch::Tensor positions;      // [N, 3] Atomic positions
    torch::Tensor numbers;        // [N] Atomic numbers
    torch::Tensor magmoms;        // [N, 3] Magnetic moments (3D vectors)
    torch::Tensor cell;           // [3, 3] Cell matrix
    torch::Tensor pbc;            // [3] Periodic boundary conditions
    int n_atoms;                  // 原子数量
    std::vector<std::string> elements;  // 元素符号列表

    // 可选的参考数据（用于验证）
    torch::Tensor ref_energy;     // 标量
    torch::Tensor ref_forces;     // [N, 3] Reference forces
    torch::Tensor ref_magforces;  // [N, 3] Reference magnetic forces
    bool has_ref_data = false;
};

// 邻居列表数据结构
struct NeighborList {
    torch::Tensor center_indices;    // [M] Central atom indices
    torch::Tensor neighbor_indices;  // [M] Neighbor atom indices
    torch::Tensor shifts;            // [M, 3] Cell shifts
    torch::Tensor batch_idx;         // [N] Batch index (all 0 for single frame)
    int n_pairs;                     // 邻居对数量

    NeighborList() : n_pairs(0) {}
};

// 描述符配置参数
struct DescriptorConfig {
    std::vector<std::string> elements;
    float rc = 5.0f;                    // 截断半径
    int n_max = 3;                      // 径向基函数阶数
    int l_max = 4;                      // 角向基函数阶数
    int nu_max = 2;                     // 相关阶数
    float m_cut = 4.0f;                 // 磁矩截断值
    bool use_spin_invariants = true;    // 是否使用自旋不变量
    float pos_scale = 1.0f;             // 位置缩放因子
    float spin_scale = 1.0f;            // 自旋缩放因子
    float epsilon = 1e-7f;              // 数值稳定性常数
    float mag_noise_threshold = 0.35f;  // 磁矩噪声阈值
    float pos_noise_threshold = 1e-8f;  // 位置噪声阈值
    float pole_threshold = 1e-6f;       // 极点判定阈值
};

// Clebsch-Gordan coefficient term
struct CGTerm {
    int m1, m2, m3;  // 量子数
    float coeff;     // 系数值

    CGTerm(int m1_, int m2_, int m3_, float c)
        : m1(m1_), m2(m2_), m3(m3_), coeff(c) {}
};

// 预测结果
struct PredictionResult {
    float total_energy;           // 总能量
    torch::Tensor atomic_energies; // [N] Atomic energies
    torch::Tensor forces;          // [N, 3] Forces on atoms
    torch::Tensor mag_forces;      // [N, 3] Magnetic forces

    // 计算时间（毫秒）
    double descriptor_time_ms = 0.0;
    double inference_time_ms = 0.0;
    double gradient_time_ms = 0.0;
    double total_time_ms = 0.0;
};

// 工具函数：将元素符号转换为原子序数
inline int element_to_number(const std::string& element) {
    static const std::map<std::string, int> element_map = {
        {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},
        {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
        {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},
        {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20},
        {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25},
        {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
        {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
        {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40},
        {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45},
        {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50},
        {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55},
        {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}
    };

    auto it = element_map.find(element);
    if (it != element_map.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown element: " + element);
}

// 工具函数：将原子序数转换为元素符号
inline std::string number_to_element(int number) {
    static const std::vector<std::string> elements = {
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd"
    };

    if (number > 0 && number < static_cast<int>(elements.size())) {
        return elements[number];
    }
    return "Unknown";
}

} // namespace nep

#endif // NEP_TYPES_H
