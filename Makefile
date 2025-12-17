# Makefile for MagneticNEP_CPP
# Provides targets for copying source files to LAMMPS interface directory

# Directories
SRC_DIR = src
INTERFACE_DIR = interface/USER-NEP-SPIN

# Source files to copy (excluding main.cpp)
SRC_FILES = cg_coefficients.h \
            descriptor.cpp \
            descriptor.h \
            math_utils.cpp \
            math_utils.h \
            model.cpp \
            model.h \
            neighbor_list.cpp \
            neighbor_list.h \
            nep_types.h \
            xyz_reader.cpp \
            xyz_reader.h

# LAMMPS interface files (already in interface directory)
LAMMPS_FILES = pair_nep_spin.cpp \
               pair_nep_spin.h \
               Install.sh \
               USER-NEP-SPIN.cmake

.PHONY: interface clean-interface help

# Default target
help:
	@echo "Available targets:"
	@echo "  make interface       - Copy src files to $(INTERFACE_DIR)"
	@echo "  make clean-interface - Remove copied src files from $(INTERFACE_DIR)"
	@echo "  make help            - Show this help message"

# Copy source files to interface directory
interface:
	@echo "Copying source files to $(INTERFACE_DIR)..."
	@mkdir -p $(INTERFACE_DIR)
	@for file in $(SRC_FILES); do \
		cp -v $(SRC_DIR)/$$file $(INTERFACE_DIR)/; \
	done
	@echo ""
	@echo "Done! Files copied to $(INTERFACE_DIR)"

# Clean copied source files from interface directory (keep LAMMPS-specific files)
clean-interface:
	@echo "Removing copied source files from $(INTERFACE_DIR)..."
	@for file in $(SRC_FILES); do \
		if [ -f "$(INTERFACE_DIR)/$$file" ]; then \
			rm -v "$(INTERFACE_DIR)/$$file"; \
		fi; \
	done
	@echo ""
	@echo "Done! Remaining files in $(INTERFACE_DIR)"
