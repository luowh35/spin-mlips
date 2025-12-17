#include "xyz_reader.h"
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace nep {

AtomicSystem XYZReader::read_frame(const std::string& filename, int frame_idx, const torch::Device& device) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    int current_frame = 0;

    while (std::getline(file, line)) {
        // Read number of atoms
        line = trim(line);
        if (line.empty()) continue;

        int n_atoms = std::stoi(line);

        // Read comment line
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file");
        }

        AtomicSystem system;
        system.n_atoms = n_atoms;

        // Parse comment line (contains Lattice, Properties, etc.)
        parse_comment_line(line, system);

        // Parse Properties field to get column order
        std::vector<std::string> properties;
        if (line.find("Properties=") != std::string::npos) {
            size_t start = line.find("Properties=");
            size_t end = line.find(" ", start);
            if (end == std::string::npos) end = line.length();
            std::string prop_str = line.substr(start, end - start);
            properties = parse_properties(prop_str);
        }

        // Read atom data
        std::vector<std::string> elements;
        std::vector<float> positions, magmoms, forces;

        for (int i = 0; i < n_atoms; ++i) {
            if (!std::getline(file, line)) {
                throw std::runtime_error("Unexpected end of file while reading atoms");
            }
            parse_atom_line(line, properties, elements, positions, magmoms, forces);
        }

        // If this is the target frame, convert to tensor and return
        if (current_frame == frame_idx) {
            // Convert elements to numbers
            std::vector<int64_t> numbers;
            for (const auto& elem : elements) {
                numbers.push_back(element_to_number(elem));
            }

            // Create tensors
            system.positions = torch::from_blob(positions.data(), {n_atoms, 3}, torch::kFloat32)
                                .clone().to(device);
            system.numbers = torch::from_blob(numbers.data(), {n_atoms}, torch::kInt64)
                              .clone().to(device);
            system.magmoms = torch::from_blob(magmoms.data(), {n_atoms, 3}, torch::kFloat32)
                              .clone().to(device);
            system.elements = elements;

            // NaN cleaning
            system.positions = torch::nan_to_num(system.positions, 0.0, 0.0, 0.0);
            system.magmoms = torch::nan_to_num(system.magmoms, 0.0, 0.0, 0.0);

            // If there is reference data
            if (!forces.empty()) {
                system.ref_forces = torch::from_blob(forces.data(), {n_atoms, 3}, torch::kFloat32)
                                    .clone().to(device);
                system.has_ref_data = true;
            }

            file.close();
            return system;
        }

        current_frame++;
    }

    file.close();
    throw std::runtime_error("Frame index " + std::to_string(frame_idx) + " not found");
}

std::vector<AtomicSystem> XYZReader::read_all_frames(const std::string& filename, const torch::Device& device) {
    std::vector<AtomicSystem> systems;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        int n_atoms = std::stoi(line);

        // Read comment line
        if (!std::getline(file, line)) break;

        AtomicSystem system;
        system.n_atoms = n_atoms;
        parse_comment_line(line, system);

        // Parse Properties
        std::vector<std::string> properties;
        if (line.find("Properties=") != std::string::npos) {
            size_t start = line.find("Properties=");
            size_t end = line.find(" ", start);
            if (end == std::string::npos) end = line.length();
            std::string prop_str = line.substr(start, end - start);
            properties = parse_properties(prop_str);
        }

        // Read atom data
        std::vector<std::string> elements;
        std::vector<float> positions, magmoms, forces;

        for (int i = 0; i < n_atoms; ++i) {
            if (!std::getline(file, line)) break;
            parse_atom_line(line, properties, elements, positions, magmoms, forces);
        }

        // Convert to tensor
        std::vector<int64_t> numbers;
        for (const auto& elem : elements) {
            numbers.push_back(element_to_number(elem));
        }

        system.positions = torch::from_blob(positions.data(), {n_atoms, 3}, torch::kFloat32)
                            .clone().to(device);
        system.numbers = torch::from_blob(numbers.data(), {n_atoms}, torch::kInt64)
                          .clone().to(device);
        system.magmoms = torch::from_blob(magmoms.data(), {n_atoms, 3}, torch::kFloat32)
                          .clone().to(device);
        system.elements = elements;

        // NaN cleanup
        system.positions = torch::nan_to_num(system.positions, 0.0, 0.0, 0.0);
        system.magmoms = torch::nan_to_num(system.magmoms, 0.0, 0.0, 0.0);

        if (!forces.empty()) {
            system.ref_forces = torch::from_blob(forces.data(), {n_atoms, 3}, torch::kFloat32)
                                .clone().to(device);
            system.has_ref_data = true;
        }

        systems.push_back(system);
    }

    file.close();
    return systems;
}

void XYZReader::parse_comment_line(const std::string& line, AtomicSystem& system) {
    // Parse Lattice
    if (line.find("Lattice=") != std::string::npos) {
        size_t start = line.find("Lattice=");
        size_t end = line.find("\"", start + 9);
        if (end != std::string::npos) {
            std::string lattice_str = line.substr(start, end - start + 1);
            system.cell = parse_lattice(lattice_str);
        }
    }

    // Default PBC to all True
    system.pbc = torch::ones({3}, torch::kBool);

    // Parse Energy (if available)
    if (line.find("Energy=") != std::string::npos) {
        size_t start = line.find("Energy=") + 7;
        size_t end = line.find(" ", start);
        if (end == std::string::npos) end = line.length();
        std::string energy_str = line.substr(start, end - start);
        try {
            float energy = std::stof(energy_str);
            system.ref_energy = torch::tensor({energy});
            system.has_ref_data = true;
        } catch (...) {}
    }
}

torch::Tensor XYZReader::parse_lattice(const std::string& lattice_str) {
    std::string quoted = extract_quoted(lattice_str);
    auto values = split(quoted, ' ');

    if (values.size() != 9) {
        throw std::runtime_error("Invalid lattice format, expected 9 values");
    }

    std::vector<float> cell_data;
    for (const auto& v : values) {
        cell_data.push_back(std::stof(v));
    }

    // Create 3x3 matrix
    return torch::from_blob(cell_data.data(), {3, 3}, torch::kFloat32).clone();
}

std::vector<std::string> XYZReader::parse_properties(const std::string& prop_str) {
    // Properties=species:S:1:pos:R:3:force:R:3:magnetic_moment:R:3
    std::vector<std::string> result;

    std::string content = prop_str.substr(11); // Remove "Properties="
    auto parts = split(content, ':');

    // Parse format: name:type:count
    for (size_t i = 0; i < parts.size();) {
        if (i + 2 < parts.size()) {
            std::string name = parts[i];
            std::string type = parts[i + 1];
            int count = std::stoi(parts[i + 2]);

            for (int j = 0; j < count; ++j) {
                result.push_back(name);
            }
            i += 3;
        } else {
            break;
        }
    }

    return result;
}

void XYZReader::parse_atom_line(
    const std::string& line,
    const std::vector<std::string>& properties,
    std::vector<std::string>& elements,
    std::vector<float>& positions,
    std::vector<float>& magmoms,
    std::vector<float>& forces
) {
    // Split by TAB first
    auto tokens = split(trim(line), '\t');

    // Filter empty tokens
    std::vector<std::string> filtered;
    for (const auto& t : tokens) {
        if (!trim(t).empty()) {
            filtered.push_back(trim(t));
        }
    }

    if (filtered.empty()) return;

    // First token is element symbol
    elements.push_back(filtered[0]);

    // Second TAB field contains position and force (space-separated)
    if (filtered.size() >= 2) {
        auto pos_force_tokens = split(filtered[1], ' ');
        std::vector<std::string> pos_force_filtered;
        for (const auto& t : pos_force_tokens) {
            if (!trim(t).empty()) {
                pos_force_filtered.push_back(trim(t));
            }
        }

        // Position: first 3
        if (pos_force_filtered.size() >= 3) {
            positions.push_back(std::stof(pos_force_filtered[0]));
            positions.push_back(std::stof(pos_force_filtered[1]));
            positions.push_back(std::stof(pos_force_filtered[2]));
        }

        // Force: 4th-6th
        if (pos_force_filtered.size() >= 6) {
            forces.push_back(std::stof(pos_force_filtered[3]));
            forces.push_back(std::stof(pos_force_filtered[4]));
            forces.push_back(std::stof(pos_force_filtered[5]));
        }
    }

    // Magnetic moment: 3rd-5th TAB fields
    if (filtered.size() >= 5) {
        magmoms.push_back(std::stof(filtered[2]));
        magmoms.push_back(std::stof(filtered[3]));
        magmoms.push_back(std::stof(filtered[4]));
    } else {
        // Default magnetic moment to 0
        magmoms.push_back(0.0f);
        magmoms.push_back(0.0f);
        magmoms.push_back(0.0f);
    }
}

std::vector<std::string> XYZReader::split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string XYZReader::trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        start++;
    }

    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return std::string(start, end + 1);
}

std::string XYZReader::extract_quoted(const std::string& s) {
    size_t first = s.find('"');
    size_t last = s.rfind('"');

    if (first != std::string::npos && last != std::string::npos && first < last) {
        return s.substr(first + 1, last - first - 1);
    }

    return "";
}

} // namespace nep
