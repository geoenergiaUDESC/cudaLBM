# Define the architecture type
export CUDALBM_ARCHITECTURE_DETECTION="Automatic"
export CUDALBM_ARCHITECTURE_VERSION="89"

# Define the third party install directory
export CUDALBM_PROJECT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Export the build directory
export CUDALBM_BUILD_DIR="$CUDALBM_PROJECT_DIR/build"
export CUDALBM_BIN_DIR="$CUDALBM_BUILD_DIR/bin"
export CUDALBM_INCLUDE_DIR="$CUDALBM_BUILD_DIR/include"

# Define the CUDA installation directory and load it
CUDALBM_CUDA_DIR=/usr/local/cuda
export PATH=$CUDALBM_CUDA_DIR/bin:$PATH
export LIBRARY_PATH=$CUDALBM_CUDA_DIR/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDALBM_CUDA_DIR/lib64:$LD_LIBRARY_PATH

# Export the path to the installed UCX
CUDALBM_UCX_DIR=$CUDALBM_PROJECT_DIR/opt/ucx
export PATH=$CUDALBM_UCX_DIR/bin:$PATH
export LIBRARY_PATH=$CUDALBM_UCX_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDALBM_UCX_DIR/lib:$LD_LIBRARY_PATH

# Export the path to the installed OpenMPI
CUDALBM_MPI_DIR=$CUDALBM_PROJECT_DIR/opt/OpenMPI
export PATH=$CUDALBM_MPI_DIR/bin:$PATH
export LIBRARY_PATH=$CUDALBM_MPI_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDALBM_MPI_DIR/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CUDALBM_MPI_DIR/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDALBM_MPI_DIR/include:$CPLUS_INCLUDE_PATH

# Export the path to the compiled executable
export PATH=$CUDALBM_PROJECT_DIR/build/bin:$PATH