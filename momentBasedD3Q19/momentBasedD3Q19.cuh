/**
Filename: momentBasedD3Q19.cuh
Contents: Main kernel for the moment representation with the D3Q19 velocity set
**/

#ifndef __MBLBM_MOMENTBASEDD319_CUH
#define __MBLBM_MOMENTBASEDD319_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"
#include "../collision.cuh"
#include "../moments/moments.cuh"
#include "../fileIO/fileIO.cuh"
#include "../runTimeIO/runTimeIO.cuh"
#include "../postProcess.cuh"

namespace LBM
{
    /**
     * @brief Enumerated type of cache eviction policies
     **/
    typedef enum GPUCacheEvictionPolicyEnum : label_t
    {
        evictFirst = 0,
        evictLast = 1
    } GPUCacheEvictionPolicy;

    /**
     * @brief Enumerated type of cache levels
     **/
    typedef enum GPUCacheLevelEnum : label_t
    {
        L1 = 0,
        L2 = 1
    } GPUCacheLevel;

    /**
     * @brief Perform a prefetch to a particular level of cache
     * @tparam cacheLevel The cache level to prefetch to
     * @tparam evictionPolicy The cache eviction policy
     * @tparam prefetchDistance The number of cycles ahead to prefetch
     * @param fMom Pointer to the interleaved moment variables on the GPU
     **/
    template <const GPUCacheLevel cacheLevel, const GPUCacheEvictionPolicy evictionPolicy, const label_t prefetchDistance>
    __device__ inline void prefetch(const scalar_t *const ptrRestrict fMom) noexcept
    {
        static_assert((cacheLevel == L1) | (cacheLevel == L2), "Prefetch cache level must be 1 or 2");
        static_assert((evictionPolicy == evictFirst) | (evictionPolicy == evictLast), "Cache eviction policy must be evictFirst or evictLast");
        // static_assert((__CUDA_ARCH__ >= 350), "CUDA architecture must be >= 350");

        const label_t total_blocks = d_NUM_BLOCK_X * d_NUM_BLOCK_Y * d_NUM_BLOCK_Z;
        const label_t current_block_index = blockIdx.z * (d_NUM_BLOCK_X * d_NUM_BLOCK_Y) + blockIdx.y * d_NUM_BLOCK_X + blockIdx.x;

        // Prefetch multiple blocks ahead
        constexpr_for<1, prefetchDistance>(
            [&](const auto lookahead)
            {
                const label_t target_block_index = current_block_index + lookahead;

                if (target_block_index < total_blocks)
                {
                    // Calculate target block coordinates
                    const label_t target_bz = target_block_index / (d_NUM_BLOCK_X * d_NUM_BLOCK_Y);
                    const label_t target_by = (target_block_index % (d_NUM_BLOCK_X * d_NUM_BLOCK_Y)) / d_NUM_BLOCK_X;
                    const label_t target_bx = target_block_index % d_NUM_BLOCK_X;

                    // Calculate base index for target block
                    const label_t target_base_idx = NUMBER_MOMENTS() * (threadIdx.x + block::nx() * (threadIdx.y + block::ny() * threadIdx.z) + block::size() * (target_bx + d_NUM_BLOCK_X * (target_by + d_NUM_BLOCK_Y * target_bz)));

                    // Prefetch the moments
                    if constexpr (cacheLevel == L1)
                    {
                        asm volatile("prefetch.global.L1 [%0];" : : "l"(&fMom[target_base_idx]));

                        if constexpr (evictionPolicy == evictLast)
                        {
                            asm volatile("prefetch.global.L1::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                        }

                        if constexpr (evictionPolicy == evictFirst)
                        {
                            asm volatile("prefetch.global.L1::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                        }

                        // For 64-bit precision, prefetch the next cache line
                        if constexpr (sizeof(scalar_t) == 8)
                        {
                            asm volatile("prefetch.global.L1 [%0];" : : "l"(&fMom[target_base_idx + 8]));

                            if constexpr (evictionPolicy == evictLast)
                            {
                                asm volatile("prefetch.global.L1::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                            }

                            if constexpr (evictionPolicy == evictFirst)
                            {
                                asm volatile("prefetch.global.L1::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                            }
                        }
                    }

                    if constexpr (cacheLevel == L2)
                    {
                        asm volatile("prefetch.global.L2 [%0];" : : "l"(&fMom[target_base_idx]));

                        if constexpr (evictionPolicy == evictLast)
                        {
                            asm volatile("prefetch.global.L2::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                        }

                        if constexpr (evictionPolicy == evictFirst)
                        {
                            asm volatile("prefetch.global.L2::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                        }

                        // For 64-bit precision, prefetch the next cache line
                        if constexpr (sizeof(scalar_t) == 8)
                        {
                            asm volatile("prefetch.global.L2 [%0];" : : "l"(&fMom[target_base_idx + 8]));

                            if constexpr (evictionPolicy == evictLast)
                            {
                                asm volatile("prefetch.global.L2::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                            }

                            if constexpr (evictionPolicy == evictFirst)
                            {
                                asm volatile("prefetch.global.L2::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                            }
                        }
                    }
                }
            });
    }

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param fMom Pointer to the interleaved moment variables on the GPU
     * @param nodeType Pointer to the mesh node types on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBounds __global__ void momentBasedD3Q19(
        scalar_t *const ptrRestrict fMom,
        device::halo blockHalo)
    {
        prefetch<L1, evictFirst, 1>(fMom);

        if (device::out_of_bounds())
        {
            return;
        }

        scalar_t pop[VelocitySet::D3Q19::Q()];
        __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];

        momentArray_t moments = {
            rho0() + fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::u()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::v()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::w()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)]};

        // Reconstruct the population from the moments
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // Save populations in shared memory
        sharedMemory::save<VelocitySet::D3Q19>(pop, s_pop);

        // Pull from shared memory
        sharedMemory::pull<VelocitySet::D3Q19>(pop, s_pop);

        // Load pop from global memory in cover nodes
        blockHalo.popLoad<VelocitySet::D3Q19>(pop);

        // Calculate the moments either at the boundary or interior
        const normalVector b_n;
        if (b_n.isBoundary())
        {
            boundaryConditions::calculateMoments<VelocitySet::D3Q19>(pop, moments, b_n);
        }
        else
        {
            VelocitySet::D3Q19::calculateMoments(pop, moments);
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments);

        // Collide
        collide(moments);

        // Calculate post collision populations
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // Write to global memory
        fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)] = moments[0] - rho0();
        fMom[device::idxMom<index::u()>(threadIdx, blockIdx)] = moments[1];
        fMom[device::idxMom<index::v()>(threadIdx, blockIdx)] = moments[2];
        fMom[device::idxMom<index::w()>(threadIdx, blockIdx)] = moments[3];
        fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)] = moments[4];
        fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)] = moments[5];
        fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)] = moments[6];
        fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)] = moments[7];
        fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)] = moments[8];
        fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)] = moments[9];

        // Save the populations to the block halo
        blockHalo.popSave<VelocitySet::D3Q19>(pop);
    }

    template <class VSet>
    __device__ inline void popLoad_v2(
        const scalar_t *ptrRestrict const fGhost_x0_,
        const scalar_t *ptrRestrict const fGhost_x1_,
        const scalar_t *ptrRestrict const fGhost_y0_,
        const scalar_t *ptrRestrict const fGhost_y1_,
        const scalar_t *ptrRestrict const fGhost_z0_,
        const scalar_t *ptrRestrict const fGhost_z1_,
        scalar_t (&ptrRestrict pop)[VSet::Q()]) noexcept
    {
        const label_t tx = threadIdx.x;
        const label_t ty = threadIdx.y;
        const label_t tz = threadIdx.z;

        const label_t bx = blockIdx.x;
        const label_t by = blockIdx.y;
        const label_t bz = blockIdx.z;

        const label_t txp1 = (tx + 1 + block::nx()) % block::nx();
        const label_t txm1 = (tx - 1 + block::nx()) % block::nx();

        const label_t typ1 = (ty + 1 + block::ny()) % block::ny();
        const label_t tym1 = (ty - 1 + block::ny()) % block::ny();

        const label_t tzp1 = (tz + 1 + block::nz()) % block::nz();
        const label_t tzm1 = (tz - 1 + block::nz()) % block::nz();

        const label_t bxm1 = (bx - 1 + d_NUM_BLOCK_X) % d_NUM_BLOCK_X;
        const label_t bxp1 = (bx + 1 + d_NUM_BLOCK_X) % d_NUM_BLOCK_X;

        const label_t bym1 = (by - 1 + d_NUM_BLOCK_Y) % d_NUM_BLOCK_Y;
        const label_t byp1 = (by + 1 + d_NUM_BLOCK_Y) % d_NUM_BLOCK_Y;

        const label_t bzm1 = (bz - 1 + d_NUM_BLOCK_Z) % d_NUM_BLOCK_Z;
        const label_t bzp1 = (bz + 1 + d_NUM_BLOCK_Z) % d_NUM_BLOCK_Z;

        if (tx == 0)
        { // w
            pop[1] = fGhost_x1_[device::idxPopX<0, VSet::QF()>(ty, tz, bxm1, by, bz)];
            pop[7] = fGhost_x1_[device::idxPopX<1, VSet::QF()>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)];
            pop[9] = fGhost_x1_[device::idxPopX<2, VSet::QF()>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))];
            pop[13] = fGhost_x1_[device::idxPopX<3, VSet::QF()>(typ1, tz, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)];
            pop[15] = fGhost_x1_[device::idxPopX<4, VSet::QF()>(ty, tzp1, bxm1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
        }
        else if (tx == (block::nx() - 1))
        { // e
            pop[2] = fGhost_x0_[device::idxPopX<0, VSet::QF()>(ty, tz, bxp1, by, bz)];
            pop[8] = fGhost_x0_[device::idxPopX<1, VSet::QF()>(typ1, tz, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)];
            pop[10] = fGhost_x0_[device::idxPopX<2, VSet::QF()>(ty, tzp1, bxp1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
            pop[14] = fGhost_x0_[device::idxPopX<3, VSet::QF()>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)];
            pop[16] = fGhost_x0_[device::idxPopX<4, VSet::QF()>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))];
        }

        if (ty == 0)
        { // s
            pop[3] = fGhost_y1_[device::idxPopY<0, VSet::QF()>(tx, tz, bx, bym1, bz)];
            pop[7] = fGhost_y1_[device::idxPopY<1, VSet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)];
            pop[11] = fGhost_y1_[device::idxPopY<2, VSet::QF()>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))];
            pop[14] = fGhost_y1_[device::idxPopY<3, VSet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, bz)];
            pop[17] = fGhost_y1_[device::idxPopY<4, VSet::QF()>(tx, tzp1, bx, bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
        }
        else if (ty == (block::ny() - 1))
        { // n
            pop[4] = fGhost_y0_[device::idxPopY<0, VSet::QF()>(tx, tz, bx, byp1, bz)];
            pop[8] = fGhost_y0_[device::idxPopY<1, VSet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, bz)];
            pop[12] = fGhost_y0_[device::idxPopY<2, VSet::QF()>(tx, tzp1, bx, byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
            pop[13] = fGhost_y0_[device::idxPopY<3, VSet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)];
            pop[18] = fGhost_y0_[device::idxPopY<4, VSet::QF()>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))];
        }

        if (tz == 0)
        { // b
            pop[5] = fGhost_z1_[device::idxPopZ<0, VSet::QF()>(tx, ty, bx, by, bzm1)];
            pop[9] = fGhost_z1_[device::idxPopZ<1, VSet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)];
            pop[11] = fGhost_z1_[device::idxPopZ<2, VSet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)];
            pop[16] = fGhost_z1_[device::idxPopZ<3, VSet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzm1)];
            pop[18] = fGhost_z1_[device::idxPopZ<4, VSet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)];
        }
        else if (tz == (block::nz() - 1))
        { // f
            pop[6] = fGhost_z0_[device::idxPopZ<0, VSet::QF()>(tx, ty, bx, by, bzp1)];
            pop[10] = fGhost_z0_[device::idxPopZ<1, VSet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzp1)];
            pop[12] = fGhost_z0_[device::idxPopZ<2, VSet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)];
            pop[15] = fGhost_z0_[device::idxPopZ<3, VSet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)];
            pop[17] = fGhost_z0_[device::idxPopZ<4, VSet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)];
        }
    }

    __device__ [[nodiscard]] inline bool West(const label_t x) noexcept
    {
        return (threadIdx.x == 0 && x != 0);
    }
    __device__ [[nodiscard]] inline bool East(const label_t x) noexcept
    {
        return (threadIdx.x == (block::nx() - 1) && x != (d_nx - 1));
    }
    __device__ [[nodiscard]] inline bool South(const label_t y) noexcept
    {
        return (threadIdx.y == 0 && y != 0);
    }
    __device__ [[nodiscard]] inline bool North(const label_t y) noexcept
    {
        return (threadIdx.y == (block::ny() - 1) && y != (d_ny - 1));
    }
    __device__ [[nodiscard]] inline bool Back(const label_t z) noexcept
    {
        return (threadIdx.z == 0 && z != 0);
    }
    __device__ [[nodiscard]] inline bool Front(const label_t z) noexcept
    {
        return (threadIdx.z == (block::nz() - 1) && z != (d_nz - 1));
    }

    template <class VSet>
    __device__ inline void popSave_v2(
        scalar_t *ptrRestrict const gGhost_x0_,
        scalar_t *ptrRestrict const gGhost_x1_,
        scalar_t *ptrRestrict const gGhost_y0_,
        scalar_t *ptrRestrict const gGhost_y1_,
        scalar_t *ptrRestrict const gGhost_z0_,
        scalar_t *ptrRestrict const gGhost_z1_,
        const scalar_t (&ptrRestrict pop)[VSet::Q()]) noexcept
    {
        const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
        const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
        const label_t z = threadIdx.z + blockDim.z * blockIdx.z;

        const label_t tx = threadIdx.x;
        const label_t ty = threadIdx.y;
        const label_t tz = threadIdx.z;

        // const label_t bx = blockIdx.x;
        // const label_t by = blockIdx.y;
        // const label_t bz = blockIdx.z;

        /* write to global pop **/
        if (West(x))
        { // w
            gGhost_x0_[device::idxPopX<0, VSet::QF()>(ty, tz, blockIdx)] = pop[2];
            gGhost_x0_[device::idxPopX<1, VSet::QF()>(ty, tz, blockIdx)] = pop[8];
            gGhost_x0_[device::idxPopX<2, VSet::QF()>(ty, tz, blockIdx)] = pop[10];
            gGhost_x0_[device::idxPopX<3, VSet::QF()>(ty, tz, blockIdx)] = pop[14];
            gGhost_x0_[device::idxPopX<4, VSet::QF()>(ty, tz, blockIdx)] = pop[16];
        }
        if (East(x))
        { // e
            gGhost_x1_[device::idxPopX<0, VSet::QF()>(ty, tz, blockIdx)] = pop[1];
            gGhost_x1_[device::idxPopX<1, VSet::QF()>(ty, tz, blockIdx)] = pop[7];
            gGhost_x1_[device::idxPopX<2, VSet::QF()>(ty, tz, blockIdx)] = pop[9];
            gGhost_x1_[device::idxPopX<3, VSet::QF()>(ty, tz, blockIdx)] = pop[13];
            gGhost_x1_[device::idxPopX<4, VSet::QF()>(ty, tz, blockIdx)] = pop[15];
        }

        if (South(y))
        { // s
            gGhost_y0_[device::idxPopY<0, VSet::QF()>(tx, tz, blockIdx)] = pop[4];
            gGhost_y0_[device::idxPopY<1, VSet::QF()>(tx, tz, blockIdx)] = pop[8];
            gGhost_y0_[device::idxPopY<2, VSet::QF()>(tx, tz, blockIdx)] = pop[12];
            gGhost_y0_[device::idxPopY<3, VSet::QF()>(tx, tz, blockIdx)] = pop[13];
            gGhost_y0_[device::idxPopY<4, VSet::QF()>(tx, tz, blockIdx)] = pop[18];
        }
        if (North(y))
        { // n
            gGhost_y1_[device::idxPopY<0, VSet::QF()>(tx, tz, blockIdx)] = pop[3];
            gGhost_y1_[device::idxPopY<1, VSet::QF()>(tx, tz, blockIdx)] = pop[7];
            gGhost_y1_[device::idxPopY<2, VSet::QF()>(tx, tz, blockIdx)] = pop[11];
            gGhost_y1_[device::idxPopY<3, VSet::QF()>(tx, tz, blockIdx)] = pop[14];
            gGhost_y1_[device::idxPopY<4, VSet::QF()>(tx, tz, blockIdx)] = pop[17];
        }

        if (Back(z))
        { // b
            gGhost_z0_[device::idxPopZ<0, VSet::QF()>(tx, ty, blockIdx)] = pop[6];
            gGhost_z0_[device::idxPopZ<1, VSet::QF()>(tx, ty, blockIdx)] = pop[10];
            gGhost_z0_[device::idxPopZ<2, VSet::QF()>(tx, ty, blockIdx)] = pop[12];
            gGhost_z0_[device::idxPopZ<3, VSet::QF()>(tx, ty, blockIdx)] = pop[15];
            gGhost_z0_[device::idxPopZ<4, VSet::QF()>(tx, ty, blockIdx)] = pop[17];
        }
        if (Front(z))
        {
            gGhost_z1_[device::idxPopZ<0, VSet::QF()>(tx, ty, blockIdx)] = pop[5];
            gGhost_z1_[device::idxPopZ<1, VSet::QF()>(tx, ty, blockIdx)] = pop[9];
            gGhost_z1_[device::idxPopZ<2, VSet::QF()>(tx, ty, blockIdx)] = pop[11];
            gGhost_z1_[device::idxPopZ<3, VSet::QF()>(tx, ty, blockIdx)] = pop[16];
            gGhost_z1_[device::idxPopZ<4, VSet::QF()>(tx, ty, blockIdx)] = pop[18];
        }
    }

    launchBounds __global__ void momentBasedD3Q19_aos(
        momentArray *const ptrRestrict fMom,
        const scalar_t *ptrRestrict const fGhost_x0_,
        const scalar_t *ptrRestrict const fGhost_x1_,
        const scalar_t *ptrRestrict const fGhost_y0_,
        const scalar_t *ptrRestrict const fGhost_y1_,
        const scalar_t *ptrRestrict const fGhost_z0_,
        const scalar_t *ptrRestrict const fGhost_z1_,
        scalar_t *ptrRestrict const gGhost_x0_,
        scalar_t *ptrRestrict const gGhost_x1_,
        scalar_t *ptrRestrict const gGhost_y0_,
        scalar_t *ptrRestrict const gGhost_y1_,
        scalar_t *ptrRestrict const gGhost_z0_,
        scalar_t *ptrRestrict const gGhost_z1_)
    {
        // prefetch<L1, evictFirst, 1>(fMom);

        if (device::out_of_bounds())
        {
            return;
        }

        const label_t gid = device::idx(threadIdx, blockIdx);

        const label_t chunk_idx = gid / VEC_SIZE();
        const label_t lane = gid % VEC_SIZE();

        scalar_t pop[VelocitySet::D3Q19::Q()];
        __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];

        // Access all 10 values via the chunk
        momentArray_t moments = {
            fMom[chunk_idx].rho[lane] + rho0(),
            fMom[chunk_idx].u[lane],
            fMom[chunk_idx].v[lane],
            fMom[chunk_idx].w[lane],
            fMom[chunk_idx].m_xx[lane],
            fMom[chunk_idx].m_xy[lane],
            fMom[chunk_idx].m_xz[lane],
            fMom[chunk_idx].m_yy[lane],
            fMom[chunk_idx].m_yz[lane],
            fMom[chunk_idx].m_zz[lane]};

        // momentArray_t moments = {
        //     rho0() + fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::u()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::v()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::w()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)],
        //     fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)]};

        // Reconstruct the population from the moments
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // Save populations in shared memory
        sharedMemory::save<VelocitySet::D3Q19>(pop, s_pop);

        // Pull from shared memory
        sharedMemory::pull<VelocitySet::D3Q19>(pop, s_pop);

        // Load pop from global memory in cover nodes
        popLoad_v2<VelocitySet::D3Q19>(
            fGhost_x0_,
            fGhost_x1_,
            fGhost_y0_,
            fGhost_y1_,
            fGhost_z0_,
            fGhost_z1_,
            pop);
        // blockHalo.popLoad<VelocitySet::D3Q19>(pop);

        // Calculate the moments either at the boundary or interior
        const normalVector b_n;
        if (b_n.isBoundary())
        {
            boundaryConditions::calculateMoments<VelocitySet::D3Q19>(pop, moments, b_n);
        }
        else
        {
            VelocitySet::D3Q19::calculateMoments(pop, moments);
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments);

        // Collide
        collide(moments);

        // Calculate post collision populations
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // // Write to global memory
        // fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)] = moments[0] - rho0();
        // fMom[device::idxMom<index::u()>(threadIdx, blockIdx)] = moments[1];
        // fMom[device::idxMom<index::v()>(threadIdx, blockIdx)] = moments[2];
        // fMom[device::idxMom<index::w()>(threadIdx, blockIdx)] = moments[3];
        // fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)] = moments[4];
        // fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)] = moments[5];
        // fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)] = moments[6];
        // fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)] = moments[7];
        // fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)] = moments[8];
        // fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)] = moments[9];

        // Save the populations to the block halo
        popSave_v2<VelocitySet::D3Q19>(
            gGhost_x0_,
            gGhost_x1_,
            gGhost_y0_,
            gGhost_y1_,
            gGhost_z0_,
            gGhost_z1_,
            pop);
        // blockHalo.popSave<VelocitySet::D3Q19>(pop);
    }

    // launchBounds __global__ void momentBasedD3Q19_v2(
    //     momentArray *const ptrRestrict fMom,
    //     device::halo blockHalo)
    // {
    //     // prefetch<L1, evictFirst, 1>(fMom);

    //     if (device::out_of_bounds())
    //     {
    //         return;
    //     }

    //     scalar_t pop[VelocitySet::D3Q19::Q()];
    //     __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];

    //     const label_t index = device::idx(threadIdx, blockIdx);

    //     momentArray moments = fMom[index];

    //     // momentArray_t moments = {
    //     //     rho0() + fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::u()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::v()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::w()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)],
    //     //     fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)]};

    //     // Reconstruct the population from the moments
    //     VelocitySet::D3Q19::reconstruct_v2(pop, moments);

    //     // Save populations in shared memory
    //     sharedMemory::save<VelocitySet::D3Q19>(pop, s_pop);

    //     // Pull from shared memory
    //     sharedMemory::pull<VelocitySet::D3Q19>(pop, s_pop);

    //     // Load pop from global memory in cover nodes
    //     blockHalo.popLoad<VelocitySet::D3Q19>(pop);

    //     // Calculate the moments either at the boundary or interior
    //     const normalVector b_n;
    //     if (b_n.isBoundary())
    //     {
    //         boundaryConditions::calculateMoments_v2<VelocitySet::D3Q19>(pop, moments, b_n);
    //     }
    //     else
    //     {
    //         VelocitySet::D3Q19::calculateMoments_v2(pop, moments);
    //     }

    //     // Scale the moments correctly
    //     VelocitySet::velocitySet::scale_v2(moments);

    //     // Collide
    //     collide_v2(moments);

    //     // Calculate post collision populations
    //     VelocitySet::D3Q19::reconstruct_v2(pop, moments);

    //     // Write to global memory
    //     fMom[index] = moments;
    //     // fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)] = moments[0] - rho0();
    //     // fMom[device::idxMom<index::u()>(threadIdx, blockIdx)] = moments[1];
    //     // fMom[device::idxMom<index::v()>(threadIdx, blockIdx)] = moments[2];
    //     // fMom[device::idxMom<index::w()>(threadIdx, blockIdx)] = moments[3];
    //     // fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)] = moments[4];
    //     // fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)] = moments[5];
    //     // fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)] = moments[6];
    //     // fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)] = moments[7];
    //     // fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)] = moments[8];
    //     // fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)] = moments[9];

    //     // Save the populations to the block halo
    //     blockHalo.popSave<VelocitySet::D3Q19>(pop);
    // }

    [[nodiscard]] const std::array<cudaStream_t, 1> createCudaStream() noexcept
    {
        std::array<cudaStream_t, 1> streamsLBM;

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
        checkCudaErrors(cudaDeviceSynchronize());

        return streamsLBM;
    }
}

#endif