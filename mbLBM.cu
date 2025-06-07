#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
// #include "latticeMesh/latticeMesh.cuh"
#include "array/array.cuh"
// #include "programControl.cuh"
// #include "inputControl.cuh"
// #include "latticeMesh/latticeMesh.cuh"
#include "moments/moments.cuh"
// #include "collision.cuh"
#include "original.cuh"
#include "postProcess.cuh"
#include "velocitySet/velocitySet.cuh"

using namespace mbLBM;

// int main(int argc, char *argv[])
int main(void)
{
    // set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // Perform device memory allocation
    device::array<scalar_t> d_fMom(NUMBER_LBM_NODES * NUMBER_MOMENTS);
    device::array<nodeType_t> dNodeType(NUMBER_LBM_NODES);
    device::halo ghostInterface(NX, NY, NZ);

    nodeType_t *const hNodeType = host::allocate<nodeType_t>(NUMBER_LBM_NODES);
    // scalar_t *const h_fMom = host::allocate<scalar_t>(NUMBER_LBM_NODES * NUMBER_MOMENTS);

    std::cout << "domain: " << NX << " " << NY << " " << NZ << std::endl;
    std::cout << "threadBlock: " << BLOCK_NX << " " << BLOCK_NY << " " << BLOCK_NZ << std::endl;
    std::cout << "gridBlock: " << NUM_BLOCK_X << " " << NUM_BLOCK_Y << " " << NUM_BLOCK_Z << std::endl;
    std::cout << "OMEGA = " << OMEGA << std::endl;
    std::cout << "VISC = " << VISC << std::endl;

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    initializeDomain(
        ghostInterface,
        d_fMom.ptr(),
        hNodeType, dNodeType.ptr(),
        gridBlock, threadBlock);

    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    for (label_t step = INI_STEP; step < N_STEPS; step++)
    {

        if ((step % 100) == 0)
        {
            std::cout << "Time: " << step << std::endl;
        }

        gpuMomCollisionStream<<<gridBlock, threadBlock, 0, 0>>>(
            d_fMom.ptr(),
            dNodeType.ptr(),
            ghostInterface);

        ghostInterface.swap();
    }

    std::cout << "Exited main loop" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    {
        scalar_t *const h_fMom = host::allocate<scalar_t>(NUMBER_LBM_NODES * NUMBER_MOMENTS);

        checkCudaErrors(cudaMemcpy(h_fMom, d_fMom.ptr(), sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

        writeTecplotHexahedralData(
            save<M_UY_INDEX>(h_fMom),
            "v_end.dat",
            NX, NY, NZ,
            "Title", {"v"});

        cudaFreeHost(h_fMom);
    }

    return 0;
}