#ifndef XYZ_READER_H
#define XYZ_READER_H

#include "nep_types.h"
#include <string>
#include <vector>
#include <fstream>

namespace nep {

/**
 * XYZ file reader
 * Supports extended XYZ format, including Lattice, Properties, etc.
 */
class XYZReader {
public:
    /**
     * Read a single frame from XYZ file
     * @param filename File path
     * @param frame_idx Frame index (default 0)
     * @param device Device (CPU or CUDA)
     * @return Atomic system data
     */
    static AtomicSystem read_frame(
        const std::string& filename,
        int frame_idx = 0,
        const torch::Device& device = torch::kCPU
    );

    /**
     * Read all frames from XYZ file
     * @param filename File path
     * @param device Device (CPU or CUDA)
     * @return List of atomic system data
     */
    static std::vector<AtomicSystem> read_all_frames(
        const std::string& filename,
        const torch::Device& device = torch::kCPU
    );

private:
    /**
     * Parse extended XYZ format comment line
     * Extract Lattice, Energy, Virial, Properties, etc.
     */
    static void parse_comment_line(
        const std::string& line,
        AtomicSystem& system
    );

    /**
     * Parse Lattice string
     * Format: Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"
     */
    static torch::Tensor parse_lattice(const std::string& lattice_str);

    /**
     * Parse Properties string
     * Format: Properties=species:S:1:pos:R:3:force:R:3:magnetic_moment:R:3
     * Returns column names and types
     */
    static std::vector<std::string> parse_properties(const std::string& prop_str);

    /**
     * Parse atom data line
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
     * String utility: split string
     */
    static std::vector<std::string> split(const std::string& s, char delimiter);

    /**
     * String utility: trim leading and trailing whitespace
     */
    static std::string trim(const std::string& s);

    /**
     * Extract content within quotes
     * Example: Lattice="1 0 0" -> "1 0 0"
     */
    static std::string extract_quoted(const std::string& s);
};

} // namespace nep

#endif // XYZ_READER_H
