# Makefile for building UCX and OpenMPI dependencies
OPENMPI_VERSION = 5.0.7

# Check if required environment variables are set
ifeq ($(CUDALBM_BUILD_DIR),)
$(error CUDALBM_BUILD_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(CUDALBM_BIN_DIR),)
$(error CUDALBM_BIN_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(CUDALBM_INCLUDE_DIR),)
$(error CUDALBM_INCLUDE_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(CUDALBM_CUDA_DIR),)
$(error CUDALBM_CUDA_DIR is not set. Please run "source bashrc" in the project directory first)
endif

# Check UCX and MPI directory variables
ifeq ($(CUDALBM_UCX_DIR),)
$(error CUDALBM_UCX_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(CUDALBM_MPI_DIR),)
$(error CUDALBM_MPI_DIR is not set. Please run "source bashrc" in the project directory first)
endif

CFLAGS = -O3 -march=native -mtune=native
CXXFLAGS = -O3 -march=native -mtune=native
FCFLAGS = -O3 -march=native -mtune=native
FFLAGS = -O3 -march=native -mtune=native

UCX_SRC_DIR = ucx
UCX_INSTALL_DIR = $(CUDALBM_UCX_DIR)

OPENMPI_SRC_DIR = openmpi-$(OPENMPI_VERSION)
OPENMPI_TARBALL = openmpi-$(OPENMPI_VERSION).tar.gz
OPENMPI_INSTALL_DIR = $(CUDALBM_MPI_DIR)

# Detect if -jN was passed to make and preserve it for sub-makes
ifneq (,$(findstring j,$(MAKEFLAGS)))
    # Extract the job number from MAKEFLAGS
    JOBS := $(patsubst -j%,%,$(filter -j%,$(MAKEFLAGS)))
    ifeq ($(JOBS),)
        # If no number specified, use default
        JOBS :=
    else
        # Add the -j prefix
        JOBS := -j$(JOBS)
    endif
else
    # No -j flag specified
    JOBS :=
endif

.PHONY: all clean uninstall install ucx-build openmpi-build ucx-install openmpi-install

# Default target builds but doesn't install
all: ucx-build openmpi-build

# Build UCX without installing
ucx-build:
	rm -rf $(UCX_SRC_DIR) $(UCX_INSTALL_DIR)
	git clone https://github.com/openucx/ucx.git $(UCX_SRC_DIR)
	cd $(UCX_SRC_DIR) && \
	CFLAGS="$(CFLAGS)" CXXFLAGS="$(CXXFLAGS)" FCFLAGS="$(FCFLAGS)" FFLAGS="$(FFLAGS)" \
	./autogen.sh && \
	./contrib/configure-release --prefix=$(UCX_INSTALL_DIR) --with-cuda=$(CUDALBM_CUDA_DIR) && \
	$(MAKE) $(JOBS)

# Build OpenMPI without installing (depends on UCX being built)
openmpi-build: ucx-build
	rm -rf $(OPENMPI_SRC_DIR) $(OPENMPI_INSTALL_DIR)
	wget https://download.open-mpi.org/release/open-mpi/v5.0/$(OPENMPI_TARBALL)
	tar -xvf $(OPENMPI_TARBALL)
	cd $(OPENMPI_SRC_DIR) && \
	CFLAGS="$(CFLAGS)" CXXFLAGS="$(CXXFLAGS)" FCFLAGS="$(FCFLAGS)" FFLAGS="$(FFLAGS)" \
	./configure --prefix=$(OPENMPI_INSTALL_DIR) --with-cuda=$(CUDALBM_CUDA_DIR) --with-ucx=$(UCX_INSTALL_DIR) && \
	$(MAKE) $(JOBS)

# Install UCX (assumes it's already built)
ucx-install:
	cd $(UCX_SRC_DIR) && $(MAKE) $(JOBS) install

# Install OpenMPI (assumes it's already built)
openmpi-install:
	cd $(OPENMPI_SRC_DIR) && $(MAKE) $(JOBS) install

# Complete install process: clean, uninstall, build, and install
install: clean uninstall ucx-build ucx-install openmpi-build openmpi-install
	@echo "Installation completed successfully!"

# Clean source directories
clean:
	rm -rf $(UCX_SRC_DIR)
	rm -rf $(OPENMPI_SRC_DIR) 
	rm -rf $(OPENMPI_TARBALL)

# Uninstall (remove installed files)
uninstall:
	rm -rf $(UCX_INSTALL_DIR)
	rm -rf $(OPENMPI_INSTALL_DIR)

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build both UCX and OpenMPI without installing (default)"
	@echo "  ucx-build     - Build UCX without installing"
	@echo "  openmpi-build - Build OpenMPI without installing (requires UCX)"
	@echo "  ucx-install   - Install UCX (requires UCX to be built)"
	@echo "  openmpi-install - Install OpenMPI (requires OpenMPI to be built)"
	@echo "  install       - Complete process: clean, uninstall, build, and install"
	@echo "  clean         - Remove source directories"
	@echo "  uninstall     - Remove installed files"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Parallel builds:"
	@echo "  Use -jN flag with make to enable parallel builds (e.g., make install -j16)"