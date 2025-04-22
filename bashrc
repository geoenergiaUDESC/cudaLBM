# Define the third party install directory
PROJECT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Define the CUDA installation directory and load it
CUDA_DIR=/usr/local/cuda
export PATH=$CUDA_DIR/bin:$PATH
export LIBRARY_PATH=$CUDA_DIR/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH

# Export the path to the installed UCX
UCX_DIR=$PROJECT_DIR/opt/ucx
export PATH=$UCX_DIR/bin:$PATH
export LIBRARY_PATH=$UCX_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH

# Export the path to the installed OpenMPI
MPI_DIR=$PROJECT_DIR/opt/OpenMPI
export PATH=$MPI_DIR/bin:$PATH
export LIBRARY_PATH=$MPI_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$MPI_DIR/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$MPI_DIR/include:$CPLUS_INCLUDE_PATH

# Export the path to the compiled executable
export PATH=$PROJECT_DIR/build/bin:$PATH
