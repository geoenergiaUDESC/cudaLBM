/**
 * @file hardwareConfig.cuh
 * @brief Compile-time constants for the GPU
 **/

#ifndef __MBLBM_HARDWARECONFIG_CUH
#define __MBLBM_HARDWARECONFIG_CUH

namespace LBM
{
    /**
     * @brief CUDA block dimension configuration
     * @details Compile-time constants defining thread block dimensions
     **/
    namespace block
    {
        /**
         * @brief Threads per block in x-dimension (compile-time constant)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t nx() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        /**
         * @brief Threads per block in y-dimension (compile-time constant)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t ny() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        /**
         * @brief Threads per block in z-dimension (compile-time constant)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t nz() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        /**
         * @brief Total threads per block (nx * ny * nz)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t size() noexcept
        {
            return nx() * ny() * nz();
        }

        /**
         * @brief Padding for the shared memory
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t padding() noexcept
        {
            return 33;
        }

        /**
         * @brief Stride for the shared memory
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t stride() noexcept
        {
            return size() + padding();
        }

        /**
         * @brief Total size of the shared memory
         **/
        template <class VSet, const label_t nVars>
        __device__ __host__ [[nodiscard]] inline consteval label_t sharedMemoryBufferSize() noexcept
        {
            return (VSet::Q() - 1 > nVars ? VSet::Q() - 1 : nVars) * (size() + padding());
        }

    }

    /**
     * @brief Launch bounds information
     * @note These variables are device specific - enable modification later
     **/
    [[nodiscard]] inline consteval label_t MAX_THREADS_PER_BLOCK() noexcept { return block::nx() * block::ny() * block::nz(); }
    [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }
#define launchBounds __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP())
}

#endif