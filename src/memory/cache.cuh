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
    namespace cache
    {
        /**
         * @brief Enumerated type of cache eviction policies
         **/
        namespace Policy
        {
            typedef enum Enum : label_t
            {
                evict_first = 0,
                evict_last = 1
            } Enum;
        }

        /**
         * @brief Enumerated type of cache levels
         **/
        namespace Level
        {
            typedef enum Enum : label_t
            {
                L1 = 0,
                L2 = 1
            } Enum;
        }

        /**
         * @brief Perform a prefetch to a particular level of cache
         * @tparam level The cache level to prefetch to
         * @tparam policy The cache eviction policy
         * @tparam T The type of pointer
         * @param ptr Pointer to the interleaved moment variables on the GPU
         **/
        template <const Level::Enum level, const Policy::Enum policy, typename T>
        __device__ inline void prefetch(const T *const ptrRestrict ptr) noexcept
        {
            // Check that the CUDA architecture is valid
#if defined(__CUDA_ARCH__)
            static_assert((__CUDA_ARCH__ >= 350), "CUDA architecture must be >= 350");
#endif

            // Check that the cache level is valid
            static_assert((level == Level::L1) | (level == Level::L2), "Prefetch cache level must be 1 or 2");

            // Check that the eviction policy is valid
            static_assert((policy == Policy::evict_first) | (policy == Policy::evict_last), "Cache eviction policy must be evict_first or evict_last");

            if constexpr (level == Level::L1)
            {
                if constexpr (policy == Policy::evict_first)
                {
                    asm volatile("prefetch.global.L1::evict_first [%0];" ::"l"(ptr));
                }
                else if constexpr (policy == Policy::evict_last)
                {
                    asm volatile("prefetch.global.L1::evict_last [%0];" ::"l"(ptr));
                }
            }
            else if constexpr (level == Level::L2)
            {
                if constexpr (policy == Policy::evict_first)
                {
                    asm volatile("prefetch.global.L2::evict_first [%0];" ::"l"(ptr));
                }
                else if constexpr (policy == Policy::evict_last)
                {
                    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(ptr));
                }
            }
        }
    }
}

#endif