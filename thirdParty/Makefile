# Makefile for building UCX and OpenMPI dependencies
NPROC ?= 16
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

UCX_SRC_DIR = ucx
UCX_INSTALL_DIR = $(CUDALBM_UCX_DIR)

OPENMPI_SRC_DIR = openmpi-$(OPENMPI_VERSION)
OPENMPI_TARBALL = openmpi-$(OPENMPI_VERSION).tar.gz
OPENMPI_INSTALL_DIR = $(CUDALBM_MPI_DIR)

.PHONY: all clean ucx openmpi

all: ucx openmpi

ucx: $(UCX_INSTALL_DIR)/bin/ucx_info

openmpi: $(OPENMPI_INSTALL_DIR)/bin/mpirun

$(UCX_INSTALL_DIR)/bin/ucx_info:
	rm -rf $(UCX_SRC_DIR) $(UCX_INSTALL_DIR)
	git clone https://github.com/openucx/ucx.git $(UCX_SRC_DIR)
	cd $(UCX_SRC_DIR) && \
	./autogen.sh && \
	./contrib/configure-release --prefix=$(UCX_INSTALL_DIR) --with-cuda=$(CUDALBM_CUDA_DIR) && \
	make -j$(NPROC) && \
	make install

$(OPENMPI_INSTALL_DIR)/bin/mpirun: $(UCX_INSTALL_DIR)/bin/ucx_info
	rm -rf $(OPENMPI_SRC_DIR) $(OPENMPI_INSTALL_DIR)
	wget https://download.open-mpi.org/release/open-mpi/v5.0/$(OPENMPI_TARBALL)
	tar -xvf $(OPENMPI_TARBALL)
	cd $(OPENMPI_SRC_DIR) && \
	./configure --prefix=$(OPENMPI_INSTALL_DIR) --with-cuda=$(CUDALBM_CUDA_DIR) --with-ucx=$(UCX_INSTALL_DIR) && \
	make -j$(NPROC) && \
	make install

clean:
	rm -rf $(UCX_SRC_DIR) $(UCX_INSTALL_DIR)
	rm -rf $(OPENMPI_SRC_DIR) $(OPENMPI_TARBALL) $(OPENMPI_INSTALL_DIR)