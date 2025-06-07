#ifndef ORIGINAL_CUH
#define ORIGINAL_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "globalFunctions.cuh"
#include "boundaryConditions.cuh"

namespace mbLBM
{

    [[nodiscard]] inline consteval auto MAX_THREADS_PER_BLOCK() noexcept { return 1024; }
    [[nodiscard]] inline consteval auto MIN_BLOCKS_PER_MP() noexcept { return 8; }

#include "globalFunctions.cuh"

    __host__ void interfaceCudaMemcpy(
        device::haloFace &dst,
        const device::haloFace &src,
        const cudaMemcpyKind kind) noexcept
    {
        struct MemcpyPair
        {
            scalar_t *dst;
            const scalar_t *src;
            label_t size;
        };

        MemcpyPair memcpyPairs[] = {
            {dst.x0(), src.x0(), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF},
            {dst.x1(), src.x1(), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF},
            {dst.y0(), src.y0(), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF},
            {dst.y1(), src.y1(), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF},
            {dst.z0(), src.z0(), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF},
            {dst.z1(), src.z1(), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF}};

        checkCudaErrors(cudaDeviceSynchronize());
        for (const auto &pair : memcpyPairs)
        {
            checkCudaErrors(cudaMemcpy(pair.dst, pair.src, pair.size, kind));
        }
    }

    __host__ inline void foo(scalar_t *const fMom) noexcept
    {

        // first moments
        const scalar_t rho = RHO_0;
        const scalar_t ux = U_0_X;
        const scalar_t uy = U_0_Y;
        const scalar_t uz = U_0_Z;

        for (label_t bz = 0; bz < NUM_BLOCK_Z; bz++)
        {
            for (label_t by = 0; by < NUM_BLOCK_Y; by++)
            {
                for (label_t bx = 0; bx < NUM_BLOCK_X; bx++)
                {
                    for (label_t tz = 0; tz < BLOCK_NZ; tz++)
                    {
                        for (label_t ty = 0; ty < BLOCK_NY; ty++)
                        {
                            for (label_t tx = 0; tx < BLOCK_NX; tx++)
                            {
                                if (out_of_bounds(NX, NY, NZ, tx, ty, tz, bx, by, bz))
                                {
                                    return;
                                }

                                // Zeroth moment
                                fMom[idxMom<M_RHO_INDEX>(tx, ty, tz, bx, by, bz)] = rho - RHO_0;

                                // First moment
                                fMom[idxMom<M_UX_INDEX>(tx, ty, tz, bx, by, bz)] = F_M_I_SCALE * ux;
                                fMom[idxMom<M_UY_INDEX>(tx, ty, tz, bx, by, bz)] = F_M_I_SCALE * uy;
                                fMom[idxMom<M_UZ_INDEX>(tx, ty, tz, bx, by, bz)] = F_M_I_SCALE * uz;

                                // Second moments
                                // define equilibrium populations
                                scalar_t pop[Q];
                                for (label_t i = 0; i < Q; i++)
                                {
                                    pop[i] = gpu_f_eq(
                                        h_w[i] * RHO_0,
                                        static_cast<scalar_t>(3) * (ux * h_cx[i] + uy * h_cy[i] + uz * h_cz[i]),
                                        static_cast<scalar_t>(1) - static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz));
                                }

                                const scalar_t invRho = static_cast<scalar_t>(1.0) / rho;
                                const scalar_t pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
                                const scalar_t pixy = ((pop[7] + pop[8]) - (pop[13] + pop[14])) * invRho;
                                const scalar_t pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
                                const scalar_t piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
                                const scalar_t piyz = ((pop[11] + pop[12]) - (pop[17] + pop[18])) * invRho;
                                const scalar_t pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;

                                fMom[idxMom<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * pixx;
                                fMom[idxMom<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * pixy;
                                fMom[idxMom<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * pixz;
                                fMom[idxMom<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * piyy;
                                fMom[idxMom<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * piyz;
                                fMom[idxMom<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * pizz;
                            }
                        }
                    }
                }
            }
        }
    }

    __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuInitialization_mom(
        scalar_t *const fMom)
    {
        if (out_of_bounds(NX, NY, NZ))
        {
            return;
        }

        // first moments
        const scalar_t rho = RHO_0;
        const scalar_t ux = U_0_X;
        const scalar_t uy = U_0_Y;
        const scalar_t uz = U_0_Z;

        // zeroth moment
        fMom[idxMom<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = rho - RHO_0;
        fMom[idxMom<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE * ux;
        fMom[idxMom<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE * uy;
        fMom[idxMom<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE * uz;

        // second moments
        // define equilibrium populations
        scalar_t pop[Q];
        for (label_t i = 0; i < Q; i++)
        {
            pop[i] = gpu_f_eq(
                w[i] * RHO_0,
                3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                1 - 1.5 * (ux * ux + uy * uy + uz * uz));
        }

        const scalar_t invRho = 1.0 / rho;
        const scalar_t pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
        const scalar_t pixy = ((pop[7] + pop[8]) - (pop[13] + pop[14])) * invRho;
        const scalar_t pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
        const scalar_t piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
        const scalar_t piyz = ((pop[11] + pop[12]) - (pop[17] + pop[18])) * invRho;
        const scalar_t pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;

        fMom[idxMom<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * pixx;
        fMom[idxMom<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * pixy;
        fMom[idxMom<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * pixz;
        fMom[idxMom<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * piyy;
        fMom[idxMom<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * piyz;
        fMom[idxMom<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * pizz;
    }

    __device__ static inline void reconstruct(
        scalar_t pop[19],
        const scalar_t rhoVar,
        const scalar_t ux_t30,
        const scalar_t uy_t30,
        const scalar_t uz_t30,
        const scalar_t m_xx_t45,
        const scalar_t m_xy_t90,
        const scalar_t m_xz_t90,
        const scalar_t m_yy_t45,
        const scalar_t m_yz_t90,
        const scalar_t m_zz_t45) noexcept
    {
        const scalar_t pics2 = 1.0 - cs2 * (m_xx_t45 + m_yy_t45 + m_zz_t45);

        const scalar_t multiplyTerm_0 = rhoVar * W0;
        pop[0] = multiplyTerm_0 * pics2;

        const scalar_t multiplyTerm_1 = rhoVar * W1;
        pop[1] = multiplyTerm_1 * (pics2 + ux_t30 + m_xx_t45);
        pop[2] = multiplyTerm_1 * (pics2 - ux_t30 + m_xx_t45);
        pop[3] = multiplyTerm_1 * (pics2 + uy_t30 + m_yy_t45);
        pop[4] = multiplyTerm_1 * (pics2 - uy_t30 + m_yy_t45);
        pop[5] = multiplyTerm_1 * (pics2 + uz_t30 + m_zz_t45);
        pop[6] = multiplyTerm_1 * (pics2 - uz_t30 + m_zz_t45);

        const scalar_t multiplyTerm_2 = rhoVar * W2;
        pop[7] = multiplyTerm_2 * (pics2 + ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90);
        pop[8] = multiplyTerm_2 * (pics2 - ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90);
        pop[9] = multiplyTerm_2 * (pics2 + ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90);
        pop[10] = multiplyTerm_2 * (pics2 - ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90);
        pop[11] = multiplyTerm_2 * (pics2 + uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90);
        pop[12] = multiplyTerm_2 * (pics2 - uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90);
        pop[13] = multiplyTerm_2 * (pics2 + ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
        pop[14] = multiplyTerm_2 * (pics2 - ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
        pop[15] = multiplyTerm_2 * (pics2 + ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
        pop[16] = multiplyTerm_2 * (pics2 - ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
        pop[17] = multiplyTerm_2 * (pics2 + uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);
        pop[18] = multiplyTerm_2 * (pics2 - uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);
    }

    __device__ static inline void collide(
        scalar_t *const rhoVar,
        scalar_t *const ux_t30,
        scalar_t *const uy_t30,
        scalar_t *const uz_t30,
        scalar_t *const m_xx_t45,
        scalar_t *const m_xy_t90,
        scalar_t *const m_xz_t90,
        scalar_t *const m_yy_t45,
        scalar_t *const m_yz_t90,
        scalar_t *const m_zz_t45) noexcept
    {
        const scalar_t invRho = 1.0 / *rhoVar;
        const scalar_t omegaVar = OMEGA;
        const scalar_t t_omegaVar = 1 - omegaVar;
        const scalar_t tt_omegaVar = 1 - omegaVar / 2;
        const scalar_t omegaVar_d2 = omegaVar / 2;
        const scalar_t tt_omega_t3 = tt_omegaVar * 3;

        *ux_t30 = *ux_t30 + 3 * invRho * FX;
        *uy_t30 = *uy_t30 + 3 * invRho * FY;
        *uz_t30 = *uz_t30 + 3 * invRho * FZ;

        // equation 90
        const scalar_t invRho_mt15 = 3 * invRho / 2;
        *m_xx_t45 = (t_omegaVar * *m_xx_t45 + omegaVar_d2 * *ux_t30 * *ux_t30 + invRho_mt15 * tt_omegaVar * (FX * *ux_t30 + FX * *ux_t30));
        *m_yy_t45 = (t_omegaVar * *m_yy_t45 + omegaVar_d2 * *uy_t30 * *uy_t30 + invRho_mt15 * tt_omegaVar * (FY * *uy_t30 + FY * *uy_t30));
        *m_zz_t45 = (t_omegaVar * *m_zz_t45 + omegaVar_d2 * *uz_t30 * *uz_t30 + invRho_mt15 * tt_omegaVar * (FZ * *uz_t30 + FZ * *uz_t30));

        *m_xy_t90 = (t_omegaVar * *m_xy_t90 + omegaVar * *ux_t30 * *uy_t30 + tt_omega_t3 * invRho * (FX * *uy_t30 + FY * *ux_t30));
        *m_xz_t90 = (t_omegaVar * *m_xz_t90 + omegaVar * *ux_t30 * *uz_t30 + tt_omega_t3 * invRho * (FX * *uz_t30 + FZ * *ux_t30));
        *m_yz_t90 = (t_omegaVar * *m_yz_t90 + omegaVar * *uy_t30 * *uz_t30 + tt_omega_t3 * invRho * (FY * *uz_t30 + FZ * *uy_t30));
    }

    __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuInitialization_pop(
        const scalar_t *const fMom, device::halo ghostInterface)
    {
        if (out_of_bounds(NX, NY, NZ))
        {
            return;
        }

        // zeroth moment

        scalar_t rhoVar = RHO_0 + fMom[idxMom<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t ux_t30 = fMom[idxMom<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uy_t30 = fMom[idxMom<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uz_t30 = fMom[idxMom<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xx_t45 = fMom[idxMom<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xy_t90 = fMom[idxMom<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xz_t90 = fMom[idxMom<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yy_t45 = fMom[idxMom<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yz_t90 = fMom[idxMom<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_zz_t45 = fMom[idxMom<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        scalar_t pop[Q];
        reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

        // thread xyz
        const label_t tx = threadIdx.x;
        const label_t ty = threadIdx.y;
        const label_t tz = threadIdx.z;

        // block xyz
        const label_t bx = blockIdx.x;
        const label_t by = blockIdx.y;
        const label_t bz = blockIdx.z;

        if (threadIdx.x == 0)
        { // w
            ghostInterface.fGhost().x0()[idxPopX<0>(ty, tz, bx, by, bz)] = pop[2];
            ghostInterface.fGhost().x0()[idxPopX<1>(ty, tz, bx, by, bz)] = pop[8];
            ghostInterface.fGhost().x0()[idxPopX<2>(ty, tz, bx, by, bz)] = pop[10];
            ghostInterface.fGhost().x0()[idxPopX<3>(ty, tz, bx, by, bz)] = pop[14];
            ghostInterface.fGhost().x0()[idxPopX<4>(ty, tz, bx, by, bz)] = pop[16];
        }
        else if (threadIdx.x == (BLOCK_NX - 1))
        {
            ghostInterface.fGhost().x1()[idxPopX<0>(ty, tz, bx, by, bz)] = pop[1];
            ghostInterface.fGhost().x1()[idxPopX<1>(ty, tz, bx, by, bz)] = pop[7];
            ghostInterface.fGhost().x1()[idxPopX<2>(ty, tz, bx, by, bz)] = pop[9];
            ghostInterface.fGhost().x1()[idxPopX<3>(ty, tz, bx, by, bz)] = pop[13];
            ghostInterface.fGhost().x1()[idxPopX<4>(ty, tz, bx, by, bz)] = pop[15];
        }

        if (threadIdx.y == 0)
        { // s
            ghostInterface.fGhost().y0()[idxPopY<0>(tx, tz, bx, by, bz)] = pop[4];
            ghostInterface.fGhost().y0()[idxPopY<1>(tx, tz, bx, by, bz)] = pop[8];
            ghostInterface.fGhost().y0()[idxPopY<2>(tx, tz, bx, by, bz)] = pop[12];
            ghostInterface.fGhost().y0()[idxPopY<3>(tx, tz, bx, by, bz)] = pop[13];
            ghostInterface.fGhost().y0()[idxPopY<4>(tx, tz, bx, by, bz)] = pop[18];
        }
        else if (threadIdx.y == (BLOCK_NY - 1))
        {
            ghostInterface.fGhost().y1()[idxPopY<0>(tx, tz, bx, by, bz)] = pop[3];
            ghostInterface.fGhost().y1()[idxPopY<1>(tx, tz, bx, by, bz)] = pop[7];
            ghostInterface.fGhost().y1()[idxPopY<2>(tx, tz, bx, by, bz)] = pop[11];
            ghostInterface.fGhost().y1()[idxPopY<3>(tx, tz, bx, by, bz)] = pop[14];
            ghostInterface.fGhost().y1()[idxPopY<4>(tx, tz, bx, by, bz)] = pop[17];
        }

        if (threadIdx.z == 0)
        { // b
            ghostInterface.fGhost().z0()[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[6];
            ghostInterface.fGhost().z0()[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[10];
            ghostInterface.fGhost().z0()[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[12];
            ghostInterface.fGhost().z0()[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[15];
            ghostInterface.fGhost().z0()[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[17];
        }
        else if (threadIdx.z == (BLOCK_NZ - 1))
        {
            ghostInterface.fGhost().z1()[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[5];
            ghostInterface.fGhost().z1()[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[9];
            ghostInterface.fGhost().z1()[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[11];
            ghostInterface.fGhost().z1()[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[16];
            ghostInterface.fGhost().z1()[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[18];
        }
    }

    __host__ void hostInitialization_nodeType(nodeType_t *const hNodeType) noexcept
    {
        for (label_t x = 0; x < NX; x++)
        {
            for (label_t y = 0; y < NY; y++)
            {
                for (label_t z = 0; z < NZ_TOTAL; z++)
                {
                    hNodeType[idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)] = boundaryConditions::initialCondition(x, y, z);
                }
            }
        }

        std::cout << "boundary condition done" << std::endl;
    }

    __host__ void initializeDomain(
        device::halo &ghostInterface,
        scalar_t *const &d_fMom,
        // scalar_t *const &h_fMom,
        nodeType_t *const &hNodeType,
        nodeType_t *const &dNodeType,
        const dim3 gBlock,
        const dim3 tBlock) noexcept
    {

        gpuInitialization_mom<<<gBlock, tBlock, 0, 0>>>(d_fMom);

        gpuInitialization_pop<<<gBlock, tBlock, 0, 0>>>(d_fMom, ghostInterface);

        // Node type initialization
        // checkCudaErrors(cudaMallocHost(&hNodeType, sizeof(nodeType_t) * NUMBER_LBM_NODES));

        hostInitialization_nodeType(hNodeType);
        checkCudaErrors(cudaMemcpy(dNodeType, hNodeType, sizeof(nodeType_t) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());

        // Interface population initialization
        interfaceCudaMemcpy(
            ghostInterface.gGhost(),
            ghostInterface.fGhost(),
            cudaMemcpyDeviceToDevice);

        // Synchronize after all initializations
        checkCudaErrors(cudaDeviceSynchronize());

        // Synchronize and transfer data back to host if needed
        {
            // checkCudaErrors(cudaDeviceSynchronize());
            // checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            // checkCudaErrors(cudaDeviceSynchronize());
            // std::cout << "Mom copy to host" << std::endl;
            // if constexpr (console_flush)
            // {
            //     fflush(stdout);
            // }
        }
    }

    __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuMomCollisionStream(
        scalar_t *const fMom,
        const nodeType_t *const dNodeType,
        device::halo ghostInterface)
    {
        const nodeType_t nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        if (out_of_bounds(NX, NY, NZ) || bad_node_type(nodeType))
        {
            return;
        }

        scalar_t pop[Q];
        __shared__ scalar_t s_pop[BLOCK_LBM_SIZE * (Q - 1)];

        scalar_t rhoVar = RHO_0 + fMom[idxMom<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t ux_t30 = fMom[idxMom<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uy_t30 = fMom[idxMom<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uz_t30 = fMom[idxMom<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xx_t45 = fMom[idxMom<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xy_t90 = fMom[idxMom<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xz_t90 = fMom[idxMom<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yy_t45 = fMom[idxMom<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yz_t90 = fMom[idxMom<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_zz_t45 = fMom[idxMom<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

        const label_t xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
        const label_t xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

        const label_t yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
        const label_t ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

        const label_t zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
        const label_t zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;

        // save populations in shared memory
        s_pop[idxPopBlock<0>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[1];
        s_pop[idxPopBlock<1>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[2];
        s_pop[idxPopBlock<2>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[3];
        s_pop[idxPopBlock<3>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[4];
        s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[5];
        s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[6];
        s_pop[idxPopBlock<6>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[7];
        s_pop[idxPopBlock<7>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[8];
        s_pop[idxPopBlock<8>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[9];
        s_pop[idxPopBlock<9>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[10];
        s_pop[idxPopBlock<10>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[11];
        s_pop[idxPopBlock<11>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[12];
        s_pop[idxPopBlock<12>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[13];
        s_pop[idxPopBlock<13>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[14];
        s_pop[idxPopBlock<14>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[15];
        s_pop[idxPopBlock<15>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[16];
        s_pop[idxPopBlock<16>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[17];
        s_pop[idxPopBlock<17>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[18];

        // sync threads of the block so all populations are saved
        __syncthreads();

        /* pull */

        pop[1] = s_pop[idxPopBlock<0>(xm1, threadIdx.y, threadIdx.z)];
        pop[2] = s_pop[idxPopBlock<1>(xp1, threadIdx.y, threadIdx.z)];
        pop[3] = s_pop[idxPopBlock<2>(threadIdx.x, ym1, threadIdx.z)];
        pop[4] = s_pop[idxPopBlock<3>(threadIdx.x, yp1, threadIdx.z)];
        pop[5] = s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, zm1)];
        pop[6] = s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, zp1)];
        pop[7] = s_pop[idxPopBlock<6>(xm1, ym1, threadIdx.z)];
        pop[8] = s_pop[idxPopBlock<7>(xp1, yp1, threadIdx.z)];
        pop[9] = s_pop[idxPopBlock<8>(xm1, threadIdx.y, zm1)];
        pop[10] = s_pop[idxPopBlock<9>(xp1, threadIdx.y, zp1)];
        pop[11] = s_pop[idxPopBlock<10>(threadIdx.x, ym1, zm1)];
        pop[12] = s_pop[idxPopBlock<11>(threadIdx.x, yp1, zp1)];
        pop[13] = s_pop[idxPopBlock<12>(xm1, yp1, threadIdx.z)];
        pop[14] = s_pop[idxPopBlock<13>(xp1, ym1, threadIdx.z)];
        pop[15] = s_pop[idxPopBlock<14>(xm1, threadIdx.y, zp1)];
        pop[16] = s_pop[idxPopBlock<15>(xp1, threadIdx.y, zm1)];
        pop[17] = s_pop[idxPopBlock<16>(threadIdx.x, ym1, zp1)];
        pop[18] = s_pop[idxPopBlock<17>(threadIdx.x, yp1, zm1)];

        /* load pop from global in cover nodes */
        ghostInterface.popLoad(pop);

        scalar_t invRho;
        if (nodeType != BULK)
        {
            boundaryConditions::calculateMoments(
                pop,
                &rhoVar,
                &ux_t30,
                &uy_t30,
                &uz_t30,
                &m_xx_t45,
                &m_xy_t90,
                &m_xz_t90,
                &m_yy_t45,
                &m_yz_t90,
                &m_zz_t45,
                nodeType);

            invRho = 1.0 / rhoVar;
        }
        else
        {
            // Calculate streaming moments

            // Equation 3
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
            invRho = 1 / rhoVar;

            // Equation 4 + force correction
            ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16])) * invRho;
            uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18])) * invRho;
            uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17])) * invRho;

            // Equation 5
            m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
            m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14]) * invRho;
            m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16]) * invRho;
            m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
            m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18]) * invRho;
            m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;
        }

        // Multiply moments by as2 -- as4*0.5 -- as4 - add correction to m_alpha_beta
        ux_t30 = F_M_I_SCALE * ux_t30;
        uy_t30 = F_M_I_SCALE * uy_t30;
        uz_t30 = F_M_I_SCALE * uz_t30;

        m_xx_t45 = F_M_II_SCALE * (m_xx_t45);
        m_xy_t90 = F_M_IJ_SCALE * (m_xy_t90);
        m_xz_t90 = F_M_IJ_SCALE * (m_xz_t90);
        m_yy_t45 = F_M_II_SCALE * (m_yy_t45);
        m_yz_t90 = F_M_IJ_SCALE * (m_yz_t90);
        m_zz_t45 = F_M_II_SCALE * (m_zz_t45);

        // Collide
        collide(&rhoVar, &ux_t30, &uy_t30, &uz_t30, &m_xx_t45, &m_xy_t90, &m_xz_t90, &m_yy_t45, &m_yz_t90, &m_zz_t45);

        // Calculate post collision populations
        reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

        /* write to global mom */

        fMom[idxMom<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar - RHO_0;
        fMom[idxMom<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30;
        fMom[idxMom<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30;
        fMom[idxMom<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30;
        fMom[idxMom<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xx_t45;
        fMom[idxMom<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xy_t90;
        fMom[idxMom<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xz_t90;
        fMom[idxMom<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yy_t45;
        fMom[idxMom<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yz_t90;
        fMom[idxMom<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_zz_t45;

        ghostInterface.populationSave(pop);
    }

}

#endif