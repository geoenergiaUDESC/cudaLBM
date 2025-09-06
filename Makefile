# Top-level Makefile
TOOL_SUBDIRS = applications/computeVersion
GPU_SUBDIRS = applications/momentBasedD3Q19 applications/fieldConvert applications/fieldCalculate
SUBDIRS = $(TOOL_SUBDIRS) $(GPU_SUBDIRS)

#BUILD_DIR = build
# BIN_DIR = $(BUILD_DIR)/bin
# INCLUDE_DIR = $(BUILD_DIR)/include

#BUILD_DIR = /home/gtchoaire/cudaLBM/build
#BIN_DIR = $(BUILD_DIR)/bin
#INCLUDE_DIR = $(BUILD_DIR)/include

BUILD_DIR = /home/gtchoaire/cudaLBM/build
BIN_DIR = $(BUILD_DIR)/bin
INCLUDE_DIR = $(BUILD_DIR)/include

.PHONY: all clean install uninstall $(SUBDIRS) directories

all: directories $(SUBDIRS)

# Create build directories
directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(INCLUDE_DIR)

# Generate hardware info in build directory
build/include/hardware.info:
	@echo "--- Detectando hardware CUDA ---"
	$(MAKE) -C computeVersion run

$(TOOL_SUBDIRS): directories
	$(MAKE) -C $@

$(GPU_SUBDIRS): build/include/hardware.info
	$(MAKE) -C $@

# Clean all projects
clean:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done
	@ rm -rf $(BUILD_DIR)

# Install all projects
install: directories
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir install; done

# Uninstall all projects
uninstall:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir uninstall; done
	@ rm -rf $(BIN_DIR) $(INCLUDE_DIR)