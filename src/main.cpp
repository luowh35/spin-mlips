#include "model.h"
#include "descriptor.h"
#include "neighbor_list.h"
#include "xyz_reader.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace nep;

int main(int argc, char* argv[]) {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "MagneticNEP C++ Inference" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // ========== Configuration ==========
    std::string model_path = "../example/best_model_new.pt";
    std::string xyz_path = "../example/test.xyz";

    // Device selection
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "✓ CUDA available, using GPU" << std::endl;
    } else {
        std::cout << "✓ Using CPU" << std::endl;
    }

    // Descriptor configuration (consistent with Python version)
    DescriptorConfig config;
    config.elements = {"Cr", "I"};
    config.rc = 4.7f;
    config.n_max = 5;
    config.l_max = 3;
    config.nu_max = 2;
    config.m_cut = 3.5f;
    config.use_spin_invariants = true;
    config.pos_scale = 200.0f;
    config.spin_scale = 1.0f;
    config.epsilon = 1e-7f;
    config.mag_noise_threshold = 0.35f;
    config.pos_noise_threshold = 1e-8f;
    config.pole_threshold = 1e-6f;

    try {
        // ========== Initialization ==========
        std::cout << "\n[1/6] Loading model..." << std::endl;
        MagneticNEPModel model(model_path, device);

        std::cout << "\n[2/6] Initializing descriptor..." << std::endl;
        MagneticACEDescriptor descriptor(config);

        std::cout << "\n[3/6] Reading XYZ file..." << std::endl;
        auto system = XYZReader::read_frame(xyz_path, 0, device);
        std::cout << "  ✓ Read " << system.n_atoms << " atoms" << std::endl;
        std::cout << "  ✓ Elements: ";
        for (const auto& elem : config.elements) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;

        // Enable gradient tracking to compute forces and magnetic forces
        system.positions.set_requires_grad(true);
        system.magmoms.set_requires_grad(true);

        // Verify magnetic moment data
        auto magmom_norms = torch::norm(system.magmoms, 2, 1);
        std::cout << "  ✓ Max magnetic moment: " << magmom_norms.max().item<float>() << " μB" << std::endl;

        // ========== Build neighbor list ==========
        std::cout << "\n[4/6] Building neighbor list..." << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();

        auto neighbors = NeighborListBuilder::build(
            system.positions,
            system.cell,
            system.pbc,
            config.rc,
            false,  // no self-interaction
            true    // bothways
        );

        auto t_neighbor = std::chrono::high_resolution_clock::now();
        auto dt_neighbor = std::chrono::duration_cast<std::chrono::milliseconds>(t_neighbor - t_start).count();

        std::cout << "  ✓ Found " << neighbors.n_pairs << " neighbor pairs" << std::endl;
        std::cout << "  ✓ Time: " << dt_neighbor << " ms" << std::endl;

        // Export neighbor list to file for comparison
        std::ofstream neighbor_file("cpp_neighbor_list.txt");
        neighbor_file << neighbors.n_pairs << "\n";
        auto center_acc = neighbors.center_indices.accessor<int64_t, 1>();
        auto neighbor_acc = neighbors.neighbor_indices.accessor<int64_t, 1>();
        auto shifts_acc = neighbors.shifts.accessor<float, 2>();
        for (int i = 0; i < neighbors.n_pairs; ++i) {
            neighbor_file << center_acc[i] << " "
                         << neighbor_acc[i] << " "
                         << static_cast<int>(shifts_acc[i][0]) << " "
                         << static_cast<int>(shifts_acc[i][1]) << " "
                         << static_cast<int>(shifts_acc[i][2]) << "\n";
        }
        neighbor_file.close();

        // ========== Compute descriptors ==========
        std::cout << "\n[5/6] Computing descriptors..." << std::endl;

        auto t_desc_start = std::chrono::high_resolution_clock::now();

        auto descriptors = descriptor.compute_from_precomputed_neighbors(
            system.positions,
            system.numbers,
            system.magmoms,
            neighbors,
            system.cell
        );

        auto t_desc_end = std::chrono::high_resolution_clock::now();
        auto dt_desc = std::chrono::duration_cast<std::chrono::milliseconds>(t_desc_end - t_desc_start).count();

        std::cout << "  ✓ Descriptor shape: " << descriptors.sizes() << std::endl;
        std::cout << "  ✓ Time: " << dt_desc << " ms" << std::endl;

        // Export descriptors to file for comparison
        std::ofstream desc_file("cpp_descriptors.txt");
        desc_file << std::setprecision(10);
        auto desc_acc = descriptors.accessor<float, 2>();
        for (int i = 0; i < system.n_atoms; ++i) {
            for (int j = 0; j < descriptors.size(1); ++j) {
                desc_file << desc_acc[i][j];
                if (j < descriptors.size(1) - 1) desc_file << " ";
            }
            desc_file << "\n";
        }
        desc_file.close();
        std::cout << "  ✓ Descriptors exported to cpp_descriptors.txt" << std::endl;

        // ========== Model inference ==========
        std::cout << "\n[6/6] Running model inference..." << std::endl;
        auto t_infer_start = std::chrono::high_resolution_clock::now();

        auto atomic_energies = model.forward_with_grad(descriptors);
        if (atomic_energies.dim() == 2) {
            atomic_energies = atomic_energies.squeeze();
        }
        auto total_energy = atomic_energies.sum();

        auto t_infer_end = std::chrono::high_resolution_clock::now();
        auto dt_infer = std::chrono::duration_cast<std::chrono::milliseconds>(t_infer_end - t_infer_start).count();

        std::cout << "  ✓ Time: " << dt_infer << " ms" << std::endl;

        // ========== Compute gradients (Forces & Magnetic Forces) ==========
        std::cout << "\n[7/7] Computing gradients..." << std::endl;
        auto t_grad_start = std::chrono::high_resolution_clock::now();

        // Compute gradients: dE/d(positions) and dE/d(magmoms)
        auto grads = torch::autograd::grad(
            {total_energy},
            {system.positions, system.magmoms},
            /*grad_outputs=*/{},
            /*retain_graph=*/false,
            /*create_graph=*/false,
            /*allow_unused=*/false
        );

        // Forces = -dE/d(positions)
        auto forces = -grads[0];
        // Magnetic forces = dE/d(magmoms) (Note: magnetic forces don't take negative sign)
        auto mag_forces = grads[1];

        auto t_grad_end = std::chrono::high_resolution_clock::now();
        auto dt_grad = std::chrono::duration_cast<std::chrono::milliseconds>(t_grad_end - t_grad_start).count();

        std::cout << "  ✓ Time: " << dt_grad << " ms" << std::endl;

        // ========== Output results ==========
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PREDICTION RESULTS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nTotal Energy: " << total_energy.item<float>() << " eV" << std::endl;

        std::cout << "\nAtomic Energies (eV):" << std::endl;
        auto atomic_energies_acc = atomic_energies.accessor<float, 1>();
        for (int i = 0; i < system.n_atoms; ++i) {
            std::cout << "  Atom " << std::setw(2) << i << ": "
                     << std::setw(12) << atomic_energies_acc[i] << std::endl;
        }

        // Output Forces
        std::cout << "\nForces (eV/Å):" << std::endl;
        auto forces_acc = forces.accessor<float, 2>();
        for (int i = 0; i < system.n_atoms; ++i) {
            std::cout << std::setw(12) << forces_acc[i][0] << " "
                     << std::setw(12) << forces_acc[i][1] << " "
                     << std::setw(12) << forces_acc[i][2] << std::endl;
        }

        // Output Magnetic Forces
        std::cout << "\nMagnetic Forces (eV/μB):" << std::endl;
        auto mag_forces_acc = mag_forces.accessor<float, 2>();
        for (int i = 0; i < system.n_atoms; ++i) {
            std::cout << std::setw(12) << mag_forces_acc[i][0] << " "
                     << std::setw(12) << mag_forces_acc[i][1] << " "
                     << std::setw(12) << mag_forces_acc[i][2] << std::endl;
        }

        // ========== Performance summary ==========
        auto t_total = std::chrono::high_resolution_clock::now();
        auto dt_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_total - t_start).count();

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PERFORMANCE SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "  Neighbor list:  " << std::setw(6) << dt_neighbor << " ms" << std::endl;
        std::cout << "  Descriptors:    " << std::setw(6) << dt_desc << " ms" << std::endl;
        std::cout << "  Inference:      " << std::setw(6) << dt_infer << " ms" << std::endl;
        std::cout << "  Gradients:      " << std::setw(6) << dt_grad << " ms" << std::endl;
        std::cout << "  " << std::string(50, '-') << std::endl;
        std::cout << "  Total:          " << std::setw(6) << dt_total << " ms" << std::endl;

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "✓ Inference completed successfully!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
