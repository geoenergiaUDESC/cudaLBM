#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
#include "array/array.cuh"
#include "programControl.cuh"
#include "latticeMesh/latticeMesh.cuh"
#include "moments/moments.cuh"
#include "collision.cuh"
#include "postProcess.cuh"
#include "momentBasedD3Q19.cuh"

using namespace LBM;

namespace LBM
{
    namespace host
    {
        __host__ [[nodiscard]] const std::vector<scalar_t> moments(const host::latticeMesh &mesh, const scalar_t u_inf)
        {
            const label_t nBlockx = mesh.nx() / block::nx();
            const label_t nBlocky = mesh.ny() / block::ny();
            std::vector<scalar_t> fMom(mesh.nx() * mesh.ny() * mesh.nz() * NUMBER_MOMENTS(), 0);

            // Loop over all grid points
            for (label_t x = 0; x < mesh.nx(); x++)
            {
                for (label_t y = 0; y < mesh.ny(); y++)
                {
                    for (label_t z = 0; z < mesh.nz(); z++)
                    {
                        // Default: no-slip (zero velocity) on all boundaries
                        constexpr const scalar_t ux = 0.0;
                        constexpr const scalar_t uy = 0.0;
                        constexpr const scalar_t uz = 0.0;

                        // Zeroth moment (density fluctuation and velocity)
                        fMom[idxMom<index::rho()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0; // rho - RHO_0 = 0
                        // Override for the top wall (y = mesh.ny()-1): ux = U_MAX
                        if (y == mesh.ny() - 1)
                        {
                            fMom[idxMom<index::u()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::F_M_I_SCALE() * u_inf;
                        }
                        else
                        {
                            fMom[idxMom<index::u()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;
                        }
                        fMom[idxMom<index::v()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;
                        fMom[idxMom<index::w()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;

                        // Second moments: compute equilibrium populations
                        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::F_eq(ux, uy, uz);

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
                        fMom[idxMom<index::xx()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::F_M_II_SCALE() * pixx;
                        fMom[idxMom<index::xy()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::F_M_IJ_SCALE() * pixy;
                        fMom[idxMom<index::xz()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::F_M_IJ_SCALE() * pixz;
                        fMom[idxMom<index::yy()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::F_M_II_SCALE() * piyy;
                        fMom[idxMom<index::yz()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::F_M_IJ_SCALE() * piyz;
                        fMom[idxMom<index::zz()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::F_M_II_SCALE() * pizz;
                    }
                }
            }
            return fMom;
        }

        __host__ [[nodiscard]] const std::vector<nodeType_t> nodeType(const latticeMesh &mesh) noexcept
        {
            std::vector<nodeType_t> nodeTypes(mesh.nx() * mesh.ny() * mesh.nz(), BULK);

            for (label_t x = 0; x < mesh.nx(); x++)
            {
                for (label_t y = 0; y < mesh.ny(); y++)
                {
                    for (label_t z = 0; z < mesh.nz(); z++)
                    {
                        nodeTypes[idxScalarBlock(x % block::nx(), y % block::ny(), z % block::nz(), x / block::nx(), y / block::ny(), z / block::nz(), mesh.nx(), mesh.ny())] = boundaryConditions::initialCondition(x, y, z, mesh.nx(), mesh.ny(), mesh.nz());
                    }
                }
            }

            return nodeTypes;
        }
    }
}

int main(void)
{
    const scalar_t u_infTemp = static_cast<scalar_t>(0.05);
    const host::latticeMesh mesh;

    // Some intialisation of constants, works for now
    {
        const scalar_t ReTemp = static_cast<scalar_t>(500);

        checkCudaErrors(cudaMemcpyToSymbol(d_Re, &ReTemp, sizeof(d_Re)));
        checkCudaErrors(cudaMemcpyToSymbol(d_u_inf, &u_infTemp, sizeof(u_infTemp)));
        const scalar_t VISC = u_infTemp * static_cast<scalar_t>(mesh.nx()) / ReTemp;
        const scalar_t TAU = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * VISC;
        const scalar_t omegaTemp = static_cast<scalar_t>(1.0) / TAU;

        checkCudaErrors(cudaMemcpyToSymbol(d_omega, &omegaTemp, sizeof(omegaTemp)));

        // Output some parameters
        // std::cout << "domain: " << mesh.nx() << " " << mesh.ny() << " " << mesh.nz() << std::endl;
        std::cout << "threadBlock: " << block::nx() << " " << block::ny() << " " << block::nz() << std::endl;
        std::cout << "gridBlock: " << mesh.nx() / block::nx() << " " << mesh.ny() / block::ny() << " " << mesh.nz() / block::nz() << std::endl;
        std::cout << "OMEGA = " << omegaTemp << std::endl;
        std::cout << "VISC = " << VISC << std::endl;
    }

    // set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // Perform device memory allocation
    device::array<scalar_t> d_fMom(host::moments(mesh, u_infTemp));
    const device::array<nodeType_t> dNodeType(host::nodeType(mesh));
    device::halo ghostInterface(host::moments(mesh, u_infTemp), mesh);

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

        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, 0>>>(
            d_fMom.ptr(),
            dNodeType.ptr(),
            ghostInterface);

        ghostInterface.swap();
    }

    std::cout << "Exited main loop" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    {
        // Very inefficient, allocating the memory twice since we are creating a vector!
        // Fix this!!!
        scalar_t *const h_fMom_ = host::allocate<scalar_t>(mesh.nx() * mesh.ny() * mesh.nz() * NUMBER_MOMENTS());

        checkCudaErrors(cudaMemcpy(h_fMom_, d_fMom.ptr(), sizeof(scalar_t) * mesh.nx() * mesh.ny() * mesh.nz() * NUMBER_MOMENTS(), cudaMemcpyDeviceToHost));

        writeTecplotHexahedralData(
            save<index::v()>(h_fMom_, mesh),
            "v_" + std::to_string(N_STEPS - 1) + ".dat",
            mesh,
            "Title", {"v"});

        cudaFreeHost(h_fMom_);
    }

    return 0;
}