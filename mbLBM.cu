#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
// #include "latticeMesh/latticeMesh.cuh"
#include "array/array.cuh"
// #include "programControl.cuh"
// #include "inputControl.cuh"
// #include "latticeMesh/latticeMesh.cuh"
#include "moments/moments.cuh"
#include "collision.cuh"
#include "postProcess.cuh"

using namespace mbLBM;

__host__ [[nodiscard]] const std::vector<scalar_t> hostInitialization_mom(
    const label_t nx,
    const label_t ny,
    const label_t nz)
{
    std::vector<scalar_t> fMom(nx * ny * nz * NUMBER_MOMENTS, 0);

    // Loop over all grid points
    for (label_t x = 0; x < nx; x++)
    {
        for (label_t y = 0; y < ny; y++)
        {
            for (label_t z = 0; z < nx; z++)
            {
                // Default: no-slip (zero velocity) on all boundaries
                constexpr const scalar_t ux = 0.0;
                constexpr const scalar_t uy = 0.0;
                constexpr const scalar_t uz = 0.0;

                // Zeroth moment (density fluctuation and velocity)
                fMom[idxMom<M_RHO_INDEX>(x, y, z, 0, 0, 0)] = 0; // rho - RHO_0 = 0
                // Override for the top wall (y = NY-1): ux = U_MAX
                if (y == NY - 1)
                {
                    fMom[idxMom<M_UX_INDEX>(x, y, z, 0, 0, 0)] = VelocitySet::velocitySet::F_M_I_SCALE() * U_MAX;
                }
                else
                {
                    fMom[idxMom<M_UX_INDEX>(x, y, z, 0, 0, 0)] = 0;
                }
                fMom[idxMom<M_UY_INDEX>(x, y, z, 0, 0, 0)] = 0;
                fMom[idxMom<M_UZ_INDEX>(x, y, z, 0, 0, 0)] = 0;

                // Second moments: compute equilibrium populations
                const std::array<scalar_t, Q> pop = VelocitySet::D3Q19::F_eq(ux, uy, uz);

                // Compute second-order moments (reused terms)
                const scalar_t pop_7_8 = pop[7] + pop[8];
                const scalar_t pop_9_10 = pop[9] + pop[10];
                const scalar_t pop_13_14 = pop[13] + pop[14];
                const scalar_t pop_15_16 = pop[15] + pop[16];
                const scalar_t pop_11_12 = pop[11] + pop[12];
                const scalar_t pop_17_18 = pop[17] + pop[18];

                const scalar_t pixx = (pop[1] + pop[2] + pop_7_8 + pop_9_10 + pop_13_14 + pop_15_16) - VelocitySet::velocitySet::cs2();
                const scalar_t pixy = (pop_7_8 - pop_13_14);
                const scalar_t pixz = (pop_9_10 - pop_15_16);
                const scalar_t piyy = (pop[3] + pop[4] + pop_7_8 + pop_11_12 + pop_13_14 + pop_17_18) - VelocitySet::velocitySet::cs2();
                const scalar_t piyz = (pop_11_12 - pop_17_18);
                const scalar_t pizz = (pop[5] + pop[6] + pop_9_10 + pop_11_12 + pop_15_16 + pop_17_18) - VelocitySet::velocitySet::cs2();

                // Store second-order moments
                fMom[idxMom<M_MXX_INDEX>(x, y, z, 0, 0, 0)] = VelocitySet::velocitySet::F_M_II_SCALE() * pixx;
                fMom[idxMom<M_MXY_INDEX>(x, y, z, 0, 0, 0)] = VelocitySet::velocitySet::F_M_IJ_SCALE() * pixy;
                fMom[idxMom<M_MXZ_INDEX>(x, y, z, 0, 0, 0)] = VelocitySet::velocitySet::F_M_IJ_SCALE() * pixz;
                fMom[idxMom<M_MYY_INDEX>(x, y, z, 0, 0, 0)] = VelocitySet::velocitySet::F_M_II_SCALE() * piyy;
                fMom[idxMom<M_MYZ_INDEX>(x, y, z, 0, 0, 0)] = VelocitySet::velocitySet::F_M_IJ_SCALE() * piyz;
                fMom[idxMom<M_MZZ_INDEX>(x, y, z, 0, 0, 0)] = VelocitySet::velocitySet::F_M_II_SCALE() * pizz;
            }
        }
    }

    return fMom;
}

__host__ [[nodiscard]] const std::vector<nodeType_t> hostInitialization_nodeType(
    const label_t nx,
    const label_t ny,
    const label_t nz) noexcept
{
    std::vector<nodeType_t> nodeTypes(nx * ny * nz, BULK);

    for (label_t x = 0; x < nx; x++)
    {
        for (label_t y = 0; y < ny; y++)
        {
            for (label_t z = 0; z < nz; z++)
            {
                nodeTypes[idxScalarBlock(x % block::nx(), y % block::ny(), z % block::nz(), x / block::nx(), y / block::ny(), z / block::nz())] = boundaryConditions::initialCondition(x, y, z, nx, ny, nz);
            }
        }
    }

    return nodeTypes;
}

int main(void)
{
    // set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // Perform device memory allocation
    device::array<scalar_t> d_fMom(hostInitialization_mom(NX, NY, NZ));
    const device::array<nodeType_t> dNodeType(hostInitialization_nodeType(NX, NY, NZ));
    device::halo ghostInterface(hostInitialization_mom(NX, NY, NZ), NX, NY, NZ);

    // Output some parameters
    std::cout << "domain: " << NX << " " << NY << " " << NZ << std::endl;
    std::cout << "threadBlock: " << block::nx() << " " << block::ny() << " " << block::nz() << std::endl;
    std::cout << "gridBlock: " << NUM_BLOCK_X << " " << NUM_BLOCK_Y << " " << NUM_BLOCK_Z << std::endl;
    std::cout << "OMEGA = " << OMEGA << std::endl;
    std::cout << "VISC = " << VISC << std::endl;

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaDeviceSynchronize());

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
        // Very inefficient, allocating the memory twice since we are creating a vector!
        scalar_t *const h_fMom_ = host::allocate<scalar_t>(NX * NY * NZ * NUMBER_MOMENTS);

        checkCudaErrors(cudaMemcpy(h_fMom_, d_fMom.ptr(), sizeof(scalar_t) * NX * NY * NZ * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

        writeTecplotHexahedralData(
            save<M_UY_INDEX>(h_fMom_, NX, NY, NZ),
            "v_1000.dat",
            NX, NY, NZ,
            "Title", {"v"});

        cudaFreeHost(h_fMom_);
    }

    return 0;
}