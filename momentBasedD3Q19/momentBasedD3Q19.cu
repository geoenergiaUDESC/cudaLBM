#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "momentBasedD3Q19.cuh"

using namespace LBM;

namespace LBM
{
    struct GhostInterfacePtrs
    {
        scalar_t *ptrRestrict fGhost_x0_;
        scalar_t *ptrRestrict fGhost_x1_;
        scalar_t *ptrRestrict fGhost_y0_;
        scalar_t *ptrRestrict fGhost_y1_;
        scalar_t *ptrRestrict fGhost_z0_;
        scalar_t *ptrRestrict fGhost_z1_;

        scalar_t *ptrRestrict gGhost_x0_;
        scalar_t *ptrRestrict gGhost_x1_;
        scalar_t *ptrRestrict gGhost_y0_;
        scalar_t *ptrRestrict gGhost_y1_;
        scalar_t *ptrRestrict gGhost_z0_;
        scalar_t *ptrRestrict gGhost_z1_;

        __host__ inline void swap() noexcept
        {
            checkCudaErrorsInline(cudaDeviceSynchronize());
            std::swap(fGhost_x0_, gGhost_x0_);
            std::swap(fGhost_x1_, gGhost_x1_);
            std::swap(fGhost_y0_, gGhost_y0_);
            std::swap(fGhost_y1_, gGhost_y1_);
            std::swap(fGhost_z0_, gGhost_z0_);
            std::swap(fGhost_z1_, gGhost_z1_);
        }
    };

    template <const std::size_t faceIndex>
    [[nodiscard]] inline constexpr std::size_t nFaces(const host::latticeMesh &mesh) noexcept
    {
        if constexpr (faceIndex == device::haloFaces::x())
        {
            return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nx()) * VelocitySet::D3Q19::QF();
        }
        if constexpr (faceIndex == device::haloFaces::y())
        {
            return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::ny()) * VelocitySet::D3Q19::QF();
        }
        if constexpr (faceIndex == device::haloFaces::z())
        {
            return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nz()) * VelocitySet::D3Q19::QF();
        }

        return 0;
    }

    template <const std::size_t faceIndex, const std::size_t side>
    __host__ [[nodiscard]] const std::vector<scalar_t> initialise_pop_halo(
        const std::vector<momentArray> &fMom, // Changed to AoSoA type
        const host::latticeMesh &mesh) noexcept
    {
        std::vector<scalar_t> face(nFaces<faceIndex>(mesh), 0);
        const label_t nBlockx = mesh.nxBlocks();
        const label_t nBlocky = mesh.nyBlocks();
        const label_t nBlockz = mesh.nzBlocks();

        // Helper function to compute global index
        // auto compute_gid = [&](label_t tx, label_t ty, label_t tz,
        //                        label_t bx, label_t by, label_t bz)
        // {
        //     return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nBlockx * (by + nBlocky * bz))));
        // };

        for (label_t bz = 0; bz < nBlockz; ++bz)
        {
            for (label_t by = 0; by < nBlocky; ++by)
            {
                for (label_t bx = 0; bx < nBlockx; ++bx)
                {
                    for (label_t tz = 0; tz < block::nz(); ++tz)
                    {
                        for (label_t ty = 0; ty < block::ny(); ++ty)
                        {
                            for (label_t tx = 0; tx < block::nx(); ++tx)
                            {
                                // Skip out-of-bounds elements
                                if (tx >= mesh.nx() || ty >= mesh.ny() || tz >= mesh.nz())
                                {
                                    continue;
                                }

                                // Compute global index and AoSoA indices
                                // const size_t gid = compute_gid(tx, ty, tz, bx, by, bz);
                                const size_t gid = host::idx(tx, ty, tz, bx, by, bz, nBlockx, nBlocky);
                                const size_t chunk_idx = gid / VEC_SIZE();
                                const size_t lane = gid % VEC_SIZE();

                                // Read moments from AoSoA structure
                                const scalar_t rho_val = rho0() + fMom[chunk_idx].rho[lane];
                                const scalar_t u_val = fMom[chunk_idx].u[lane];
                                const scalar_t v_val = fMom[chunk_idx].v[lane];
                                const scalar_t w_val = fMom[chunk_idx].w[lane];
                                const scalar_t xx_val = fMom[chunk_idx].m_xx[lane];
                                const scalar_t xy_val = fMom[chunk_idx].m_xy[lane];
                                const scalar_t xz_val = fMom[chunk_idx].m_xz[lane];
                                const scalar_t yy_val = fMom[chunk_idx].m_yy[lane];
                                const scalar_t yz_val = fMom[chunk_idx].m_yz[lane];
                                const scalar_t zz_val = fMom[chunk_idx].m_zz[lane];

                                // Reconstruct populations
                                const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::reconstruct(
                                    {rho_val, u_val, v_val, w_val,
                                     xx_val, xy_val, xz_val,
                                     yy_val, yz_val, zz_val});

                                // Handle ghost cells (same as before)
                                if constexpr (faceIndex == device::haloFaces::x())
                                {
                                    if constexpr (side == 0)
                                    {
                                        if (tx == 0)
                                        {
                                            face[host::idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[2];
                                            face[host::idxPopX<1, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[8];
                                            face[host::idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[10];
                                            face[host::idxPopX<3, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[14];
                                            face[host::idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[16];
                                        }
                                    }
                                    if constexpr (side == 1)
                                    {
                                        if (tx == (block::nx() - 1))
                                        {
                                            face[host::idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[1];
                                            face[host::idxPopX<1, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[7];
                                            face[host::idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[9];
                                            face[host::idxPopX<3, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[13];
                                            face[host::idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[15];
                                        }
                                    }
                                }

                                if constexpr (faceIndex == device::haloFaces::y())
                                {
                                    if constexpr (side == 0)
                                    {
                                        if (ty == 0)
                                        {
                                            face[host::idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[4];
                                            face[host::idxPopY<1, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[8];
                                            face[host::idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[12];
                                            face[host::idxPopY<3, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[13];
                                            face[host::idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[18];
                                        }
                                    }
                                    if constexpr (side == 1)
                                    {
                                        if (ty == (block::ny() - 1))
                                        {
                                            face[host::idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[3];
                                            face[host::idxPopY<1, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[7];
                                            face[host::idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[11];
                                            face[host::idxPopY<3, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[14];
                                            face[host::idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[17];
                                        }
                                    }
                                }

                                if constexpr (faceIndex == device::haloFaces::z())
                                {
                                    if constexpr (side == 0)
                                    {
                                        if (tz == 0)
                                        {
                                            face[host::idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[6];
                                            face[host::idxPopZ<1, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[10];
                                            face[host::idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[12];
                                            face[host::idxPopZ<3, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[15];
                                            face[host::idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[17];
                                        }
                                    }
                                    if constexpr (side == 1)
                                    {
                                        if (tz == (block::nz() - 1))
                                        {
                                            face[host::idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[5];
                                            face[host::idxPopZ<1, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[9];
                                            face[host::idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[11];
                                            face[host::idxPopZ<3, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[16];
                                            face[host::idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[18];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return face;
    }
}

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh;

    VelocitySet::D3Q19::print();

    // const host::array<scalar_t, ctorType::NO_READ> hostMoments(
    //     programCtrl,
    //     {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
    //     mesh);

    const std::vector<momentArray> hostMoments = host::moments_aos(mesh, programCtrl.u_inf());

    std::cout << "Size of hostMoments: " << hostMoments.size() << std::endl;

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup Streams
    const std::array<cudaStream_t, 1> streamsLBM = createCudaStream();

    // Perform device memory allocation
    device::array<momentArray> deviceMoments(
        hostMoments,
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        mesh);

    GhostInterfacePtrs blockHalo;
    blockHalo.fGhost_x0_ = device::allocateArray(initialise_pop_halo<device::haloFaces::x(), 0>(hostMoments, mesh));
    blockHalo.fGhost_x1_ = device::allocateArray(initialise_pop_halo<device::haloFaces::x(), 1>(hostMoments, mesh));
    blockHalo.fGhost_y0_ = device::allocateArray(initialise_pop_halo<device::haloFaces::y(), 0>(hostMoments, mesh));
    blockHalo.fGhost_y1_ = device::allocateArray(initialise_pop_halo<device::haloFaces::y(), 1>(hostMoments, mesh));
    blockHalo.fGhost_z0_ = device::allocateArray(initialise_pop_halo<device::haloFaces::z(), 0>(hostMoments, mesh));
    blockHalo.fGhost_z1_ = device::allocateArray(initialise_pop_halo<device::haloFaces::z(), 1>(hostMoments, mesh));

    blockHalo.gGhost_x0_ = device::allocateArray(initialise_pop_halo<device::haloFaces::x(), 0>(hostMoments, mesh));
    blockHalo.gGhost_x1_ = device::allocateArray(initialise_pop_halo<device::haloFaces::x(), 1>(hostMoments, mesh));
    blockHalo.gGhost_y0_ = device::allocateArray(initialise_pop_halo<device::haloFaces::y(), 0>(hostMoments, mesh));
    blockHalo.gGhost_y1_ = device::allocateArray(initialise_pop_halo<device::haloFaces::y(), 1>(hostMoments, mesh));
    blockHalo.gGhost_z0_ = device::allocateArray(initialise_pop_halo<device::haloFaces::z(), 0>(hostMoments, mesh));
    blockHalo.gGhost_z1_ = device::allocateArray(initialise_pop_halo<device::haloFaces::z(), 1>(hostMoments, mesh));

    // device::halo blockHalo(hostMoments, mesh);

    // // // device::array<scalar_t> deviceMoments(hostMoments, mesh);
    // // // device::halo blockHalo(hostMoments.arr(), mesh);

    // Copy symbols to device
    mesh.copyDeviceSymbols();
    programCtrl.copyDeviceSymbols(mesh.nx());

    // checkCudaErrors(cudaFuncSetCacheConfig(momentBasedD3Q19, cudaFuncCachePreferShared));
    // checkCudaErrors(cudaFuncSetCacheConfig(momentBasedD3Q19_v2, cudaFuncCachePreferL1));

    std::cout << "Time loop start" << std::endl;
    std::cout << std::endl;

    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << "\n";
        }

        // momentBasedD3Q19_v2<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
        //     deviceMoments.ptr(),
        //     blockHalo);

        momentBasedD3Q19_aos<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
            deviceMoments.ptr(),
            blockHalo.fGhost_x0_,
            blockHalo.fGhost_x1_,
            blockHalo.fGhost_y0_,
            blockHalo.fGhost_y1_,
            blockHalo.fGhost_z0_,
            blockHalo.fGhost_z1_,
            blockHalo.gGhost_x0_,
            blockHalo.gGhost_x1_,
            blockHalo.gGhost_y0_,
            blockHalo.gGhost_y1_,
            blockHalo.gGhost_z0_,
            blockHalo.gGhost_z1_);

        // Perform the swaps here
        blockHalo.swap();

        // if (programCtrl.save(timeStep))
        // {
        //     deviceMoments.write(programCtrl.caseName(), timeStep);
        // }
    }

    cudaFree(blockHalo.fGhost_x0_);
    cudaFree(blockHalo.fGhost_x1_);
    cudaFree(blockHalo.fGhost_y0_);
    cudaFree(blockHalo.fGhost_y1_);
    cudaFree(blockHalo.fGhost_z0_);
    cudaFree(blockHalo.fGhost_z1_);
    cudaFree(blockHalo.gGhost_x0_);
    cudaFree(blockHalo.gGhost_x1_);
    cudaFree(blockHalo.gGhost_y0_);
    cudaFree(blockHalo.gGhost_y1_);
    cudaFree(blockHalo.gGhost_z0_);
    cudaFree(blockHalo.gGhost_z1_);

    // Get ending time point and output the elapsed time
    const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::cout << "Elapsed time: " << runTimeIO::duration(std::chrono::duration_cast<std::chrono::seconds>(end - start).count()) << std::endl;
    std::cout << std::endl;
    std::cout << "MLUPS: " << runTimeIO::MLUPS<double>(mesh, programCtrl, start, end) << std::endl;
    std::cout << "End" << std::endl;

    return 0;
}