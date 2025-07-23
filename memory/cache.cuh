/**
Filename: cache.cuh
Contents: Handles the use of the cache on the GPU
**/

#ifndef __MBLBM_CACHE_CUH
#define __MBLBM_CACHE_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace device
    {
        /**
         * @brief Enumerated type of cache eviction policies
         **/
        namespace evictionPolicy
        {
            typedef enum Enum : label_t
            {
                first = 0,
                last = 1
            } Enum;
        }

        /**
         * @brief Enumerated type of cache levels
         **/
        namespace cacheLevel
        {
            typedef enum Enum : label_t
            {
                L1 = 0,
                L2 = 1
            } Enum;
        }

        /**
         * @brief Perform a prefetch to a particular level of cache
         * @tparam cacheLevel The cache level to prefetch to
         * @tparam evictionPolicy The cache eviction policy
         * @tparam prefetchDistance The number of cycles ahead to prefetch
         * @param fMom Pointer to the interleaved moment variables on the GPU
         **/
        template <const cacheLevel::Enum level, const evictionPolicy::Enum policy, const label_t prefetchDistance>
        __device__ inline void prefetch(const scalar_t *const ptrRestrict fMom) noexcept
        {
            static_assert((level == cacheLevel::L1) | (level == cacheLevel::L2), "Prefetch cache level must be 1 or 2");
            static_assert((policy == evictionPolicy::first) | (policy == evictionPolicy::last), "Cache eviction policy must be evictFirst or evictLast");
            // static_assert((__CUDA_ARCH__ >= 350), "CUDA architecture must be >= 350");

            const label_t total_blocks = device::NUM_BLOCK_X * device::NUM_BLOCK_Y * device::NUM_BLOCK_Z;
            const label_t current_block_index = blockIdx.z * (device::NUM_BLOCK_X * device::NUM_BLOCK_Y) + blockIdx.y * device::NUM_BLOCK_X + blockIdx.x;

            // Prefetch multiple blocks ahead
            constexpr_for<1, prefetchDistance>(
                [&](const auto lookahead)
                {
                    const label_t target_block_index = current_block_index + lookahead;

                    if (target_block_index < total_blocks)
                    {
                        // Calculate target block coordinates
                        const label_t target_bz = target_block_index / (device::NUM_BLOCK_X * device::NUM_BLOCK_Y);
                        const label_t target_by = (target_block_index % (device::NUM_BLOCK_X * device::NUM_BLOCK_Y)) / device::NUM_BLOCK_X;
                        const label_t target_bx = target_block_index % device::NUM_BLOCK_X;

                        // Calculate base index for target block
                        const label_t target_base_idx = NUMBER_MOMENTS() * (threadIdx.x + block::nx() * (threadIdx.y + block::ny() * threadIdx.z) + block::size() * (target_bx + device::NUM_BLOCK_X * (target_by + device::NUM_BLOCK_Y * target_bz)));

                        // Prefetch the moments
                        if constexpr (level == cacheLevel::L1)
                        {
                            asm volatile("prefetch.global.L1 [%0];" : : "l"(&fMom[target_base_idx]));

                            if constexpr (policy == evictionPolicy::last)
                            {
                                asm volatile("prefetch.global.L1::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                            }

                            if constexpr (policy == evictionPolicy::first)
                            {
                                asm volatile("prefetch.global.L1::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                            }

                            // For 64-bit precision, prefetch the next cache line
                            if constexpr (sizeof(scalar_t) == 8)
                            {
                                asm volatile("prefetch.global.L1 [%0];" : : "l"(&fMom[target_base_idx + 8]));

                                if constexpr (policy == evictionPolicy::last)
                                {
                                    asm volatile("prefetch.global.L1::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                                }

                                if constexpr (policy == evictionPolicy::first)
                                {
                                    asm volatile("prefetch.global.L1::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                                }
                            }
                        }

                        if constexpr (level == cacheLevel::L2)
                        {
                            asm volatile("prefetch.global.L2 [%0];" : : "l"(&fMom[target_base_idx]));

                            if constexpr (policy == evictionPolicy::last)
                            {
                                asm volatile("prefetch.global.L2::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                            }

                            if constexpr (policy == evictionPolicy::first)
                            {
                                asm volatile("prefetch.global.L2::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                            }

                            // For 64-bit precision, prefetch the next cache line
                            if constexpr (sizeof(scalar_t) == 8)
                            {
                                asm volatile("prefetch.global.L2 [%0];" : : "l"(&fMom[target_base_idx + 8]));

                                if constexpr (policy == evictionPolicy::last)
                                {
                                    asm volatile("prefetch.global.L2::evict_last [%0];" : : "l"(&fMom[target_base_idx]));
                                }

                                if constexpr (policy == evictionPolicy::first)
                                {
                                    asm volatile("prefetch.global.L2::evict_first [%0];" : : "l"(&fMom[target_base_idx]));
                                }
                            }
                        }
                    }
                });
        }
    }
}

#endif