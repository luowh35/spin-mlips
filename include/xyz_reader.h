#ifndef XYZ_READER_H
#define XYZ_READER_H

#include "nep_types.h"
#include <string>
#include <vector>
#include <fstream>

namespace nep {

/**
 * XYZ文件读取器
 * 支持扩展XYZ格式，包含Lattice、Properties等信息
 */
class XYZReader {
public:
    /**
     * 读取XYZ文件中的单帧
     * @param filename 文件路径
     * @param frame_idx 帧索引（默认0）
     * @param device 设备（CPU或CUDA）
     * @return 原子系统数据
     */
    static AtomicSystem read_frame(
        const std::string& filename,
        int frame_idx = 0,
        const torch::Device& device = torch::kCPU
    );

    /**
     * 读取XYZ文件中的所有帧
     * @param filename 文件路径
     * @param device 设备（CPU或CUDA）
     * @return 原子系统数据列表
     */
    static std::vector<AtomicSystem> read_all_frames(
        const std::string& filename,
        const torch::Device& device = torch::kCPU
    );

private:
    /**
     * 解析扩展XYZ格式的注释行
     * 提取Lattice、Energy、Virial、Properties等信息
     */
    static void parse_comment_line(
        const std::string& line,
        AtomicSystem& system
    );

    /**
     * 解析Lattice字符串
     * 格式: Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"
     */
    static torch::Tensor parse_lattice(const std::string& lattice_str);

    /**
     * 解析Properties字符串
     * 格式: Properties=species:S:1:pos:R:3:force:R:3:magnetic_moment:R:3
     * 返回列名和类型
     */
    static std::vector<std::string> parse_properties(const std::string& prop_str);

    /**
     * 解析原子数据行
     */
    static void parse_atom_line(
        const std::string& line,
        const std::vector<std::string>& properties,
        std::vector<std::string>& elements,
        std::vector<float>& positions,
        std::vector<float>& magmoms,
        std::vector<float>& forces
    );

    /**
     * 字符串工具：分割字符串
     */
    static std::vector<std::string> split(const std::string& s, char delimiter);

    /**
     * 字符串工具：去除首尾空格
     */
    static std::string trim(const std::string& s);

    /**
     * 提取引号内的内容
     * 例如: Lattice="1 0 0" -> "1 0 0"
     */
    static std::string extract_quoted(const std::string& s);
};

} // namespace nep

#endif // XYZ_READER_H
