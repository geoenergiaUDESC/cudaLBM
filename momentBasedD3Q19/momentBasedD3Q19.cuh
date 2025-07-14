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