# Check that the environment variables are set before trying compilation
if [ -z "${PROJECT_DIR}" ]; then
    echo ""
    echo "Third party library build failed"
    echo "The system variable PROJECT_DIR has not been set"
    echo "Try going back to the main directory and running the bashrc file"
    return 0
fi

# Make sure the opt directory is clean
rm -rf opt/

# Build UCX:
rm -rf $PROJECT_DIR/opt/ucx/
rm -rf ucx/
git clone https://github.com/openucx/ucx.git
cd ucx/
./autogen.sh
./contrib/configure-release --prefix=$PROJECT_DIR/opt/ucx --with-cuda=$CUDA_DIR
make -j16
make install
cd ../

# Build OpenMPI:
rm -rf $PROJECT_DIR/opt/OpenMPI/
rm -rf openmpi-5.0.7/
rm -rf openmpi-5.0.7.tar.gz
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
tar -xvf openmpi-5.0.7.tar.gz
cd openmpi-5.0.7/
./configure --prefix=$PROJECT_DIR/opt/OpenMPI --with-cuda=$CUDA_DIR --with-ucx=$PROJECT_DIR/opt/ucx
make -j16
make install
cd ../