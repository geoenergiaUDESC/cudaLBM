/**
Filename: sharedMemory.cuh
Contents: Handles the use of shared memory on the GPU
**/

#ifndef __MBLBM_SHAREDMEMORY_CUH
#define __MBLBM_SHAREDMEMORY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    class sharedMemory
    {
    public:
        /**
         * @brief Default constructor for the sharedMemory class
         **/
        [[nodiscard]] inline consteval sharedMemory() {};

        /**
         * @brief Saves the thread population density to the block shared memory
         * @param pop The population density at the current thread
         * @param s_pop The population density in shared memory
         **/
        template <class VSet>
        __device__ static inline void save(
            const scalar_t (&ptrRestrict pop)[VSet::Q()],
            scalar_t *s_pop, // Flattened shared buffer
            const label_t tid) noexcept
        {
            device::constexpr_for<0, (VSet::Q() - 1)>(
                [&](const auto q_)
                {
                    // const label_t idx = q_ * block::stride() + tid;
                    s_pop[q_ * block::stride() + tid] = pop[q_ + 1];
                });

            // __syncthreads();
        }

        /**
         * @brief Pulls the population density from shared memory
         * @param pop The population density at the current thread
         * @param s_pop The population density in shared memory
         **/
        template <class VSet>
        __device__ static inline void pull(
            scalar_t (&ptrRestrict pop)[VSet::Q()],
            const scalar_t *s_pop) noexcept
        {
            // Check for the size of the velocity set
            static_assert((VSet::Q() == 19) | (VSet::Q() == 27), "Incorrectly sized velocity set in sharedMemory::pull");

            const label_t xm1 = periodic_index<-1, block::nx()>(threadIdx.x);
            const label_t xp1 = periodic_index<1, block::nx()>(threadIdx.x);
            const label_t ym1 = periodic_index<-1, block::ny()>(threadIdx.y);
            const label_t yp1 = periodic_index<1, block::ny()>(threadIdx.y);
            const label_t zm1 = periodic_index<-1, block::nz()>(threadIdx.z);
            const label_t zp1 = periodic_index<1, block::nz()>(threadIdx.z);

            pop[1] = s_pop[0 * block::stride() + blockIndex(xm1, threadIdx.y, threadIdx.z)];
            pop[2] = s_pop[1 * block::stride() + blockIndex(xp1, threadIdx.y, threadIdx.z)];
            pop[3] = s_pop[2 * block::stride() + blockIndex(threadIdx.x, ym1, threadIdx.z)];
            pop[4] = s_pop[3 * block::stride() + blockIndex(threadIdx.x, yp1, threadIdx.z)];
            pop[5] = s_pop[4 * block::stride() + blockIndex(threadIdx.x, threadIdx.y, zm1)];
            pop[6] = s_pop[5 * block::stride() + blockIndex(threadIdx.x, threadIdx.y, zp1)];
            pop[7] = s_pop[6 * block::stride() + blockIndex(xm1, ym1, threadIdx.z)];
            pop[8] = s_pop[7 * block::stride() + blockIndex(xp1, yp1, threadIdx.z)];
            pop[9] = s_pop[8 * block::stride() + blockIndex(xm1, threadIdx.y, zm1)];
            pop[10] = s_pop[9 * block::stride() + blockIndex(xp1, threadIdx.y, zp1)];
            pop[11] = s_pop[10 * block::stride() + blockIndex(threadIdx.x, ym1, zm1)];
            pop[12] = s_pop[11 * block::stride() + blockIndex(threadIdx.x, yp1, zp1)];
            pop[13] = s_pop[12 * block::stride() + blockIndex(xm1, yp1, threadIdx.z)];
            pop[14] = s_pop[13 * block::stride() + blockIndex(xp1, ym1, threadIdx.z)];
            pop[15] = s_pop[14 * block::stride() + blockIndex(xm1, threadIdx.y, zp1)];
            pop[16] = s_pop[15 * block::stride() + blockIndex(xp1, threadIdx.y, zm1)];
            pop[17] = s_pop[16 * block::stride() + blockIndex(threadIdx.x, ym1, zp1)];
            pop[18] = s_pop[17 * block::stride() + blockIndex(threadIdx.x, yp1, zm1)];
        }

    private:
        /**
         * @brief Population index within a single block (thread-local storage)
         * @tparam pop Population index
         * @param tx Thread-local x-coordinate
         * @param ty Thread-local y-coordinate
         * @param tz Thread-local z-coordinate
         * @return Linearized index: tx + block::nx() * (ty + block::ny() * (tz + block::nz() * pop))
         *
         * @note Assumes populations are stored in thread-local order within a block
         * @note Memory layout: [pop][tz][ty][tx] (pop slowest varying, tx fastest)
         **/
        template <const label_t pop>
        __device__ [[nodiscard]] static inline label_t idxPopBlock(const label_t tx, const label_t ty, const label_t tz) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (pop)));
        }

        /**
         * @overload
         * @brief Population index within a single block (dim3 version)
         * @param tx Thread coordinates (dim3 struct)
         **/
        template <const label_t pop>
        __device__ [[nodiscard]] static inline label_t idxPopBlock(const dim3 &tx) noexcept
        {
            return idxPopBlock<pop>(tx.x, tx.y, tx.z);
        }

        __device__ [[nodiscard]] static inline label_t blockIndex(const label_t tx, const label_t ty, const label_t tz) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * tz);
        }

        /**
         * @brief Compute periodic boundary index with optimized power-of-two handling
         * @tparam Shift Direction shift (-1 for backward, +1 for forward)
         * @tparam Dim Dimension size (periodic length)
         * @tparam label_t Integer type for indexing
         * @param idx Current index position
         * @return Shifted index with periodic wrapping
         *
         * @note Uses bitwise AND optimization when Dim is power-of-two
         * @warning Shift must be either -1 or 1 (statically enforced)
         *
         * Example usage:
         * @code
         * // For 256-element periodic dimension (power-of-two)
         * periodic_index<-1, 256>(5);  // Returns 4
         * periodic_index<1, 256>(255); // Returns 0
         *
         * // For non-power-of-two dimension (e.g., 100)
         * periodic_index<-1, 100>(0);   // Returns 99
         * periodic_index<1, 100>(99);   // Returns 0
         * @endcode
         **/
        template <const int Shift, const int Dim>
        __device__ [[nodiscard]] static inline label_t periodic_index(const label_t idx) noexcept
        {
            static_assert((Shift == -1) || (Shift == 1), "Shift must be -1 or 1");

            if constexpr (Dim > 0 && (Dim & (Dim - 1)) == 0)
            {
                // Power-of-two: use bitwise AND
                if constexpr (Shift == -1)
                {
                    return (idx - 1) & (Dim - 1);
                }
                else
                {
                    return (idx + 1) & (Dim - 1);
                }
            }
            else
            {
                // General case: adjust by adding Dim to ensure nonnegative modulo
                if constexpr (Shift == -1)
                {
                    return (idx - 1 + Dim) % Dim;
                }
                else
                {
                    return (idx + 1) % Dim;
                }
            }
        }
    };

}

#endif