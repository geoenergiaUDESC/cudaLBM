# Check that the environment variables are set before trying compilation
if [ -z "${CUDALBM_PROJECT_DIR}" ]; then
    echo ""
    echo "Third party library build failed"
    echo "The system variable CUDALBM_PROJECT_DIR has not been set"
    echo "Try going back to the main directory and running the bashrc file"
    return 0
fi

# Make sure the opt directory is clean
rm -rf $CUDALBM_UCX_DIR
rm -rf $CUDALBM_MPI_DIR

# Build UCX:
rm -rf ucx/
git clone https://github.com/openucx/ucx.git
cd ucx/
./autogen.sh
./contrib/configure-release --prefix=$CUDALBM_UCX_DIR --with-cuda=$CUDALBM_CUDA_DIR
make
make install
cd ../

# Build OpenMPI:
rm -rf openmpi-5.0.7/
rm -rf openmpi-5.0.7.tar.gz
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
tar -xvf openmpi-5.0.7.tar.gz
cd openmpi-5.0.7/
./configure --prefix=$CUDALBM_MPI_DIR --with-cuda=$CUDALBM_CUDA_DIR --with-ucx=$CUDALBM_UCX_DIR
make
make install
cd ../