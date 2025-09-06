/**
Filename: streaming.cuh
Contents: Handles the streaming process on the GPU
**/

#ifndef __MBLBM_STREAMING_CUH
#define __MBLBM_STREAMING_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"

namespace LBM
{
    class streaming
    {
    public:
        /**
         * @brief Default constructor for the sharedMemory class
         **/
        [[nodiscard]] inline consteval streaming() {};

        /**
         * @brief Saves the thread population density to the block shared memory
         * @param pop The population density at the current thread
         * @param s_pop The population density in shared memory
         **/
        template <class VSet, const label_t N>
        __device__ static inline void save(
            const thread::array<scalar_t, VSet::Q()> &pop,
            thread::array<scalar_t, N> &s_pop,
            const label_t tid) noexcept
        {
            device::constexpr_for<0, (VSet::Q() - 1)>(
                [&](const auto q_)
                {
                    // const label_t idx = q_ * block::stride() + tid;
                    s_pop[label_constant<q_ * block::stride()>() + tid] = pop(label_constant<q_ + 1>());
                });
        }

        /**
         * @brief Pulls the population density from shared memory
         * @param pop The population density at the current thread
         * @param s_pop The population density in shared memory
         **/
        template <class VSet, const label_t N>
        __device__ static inline void pull(
            thread::array<scalar_t, VSet::Q()> &pop,
            const thread::array<scalar_t, N> &s_pop) noexcept
        {
            const label_t xm1 = periodic_index<-1, block::nx()>(threadIdx.x);
            const label_t xp1 = periodic_index<1, block::nx()>(threadIdx.x);
            const label_t ym1 = periodic_index<-1, block::ny()>(threadIdx.y);
            const label_t yp1 = periodic_index<1, block::ny()>(threadIdx.y);
            const label_t zm1 = periodic_index<-1, block::nz()>(threadIdx.z);
            const label_t zp1 = periodic_index<1, block::nz()>(threadIdx.z);

            pop(label_constant<1>()) = s_pop[label_constant<0 * block::stride()>() + device::idxBlock(xm1, threadIdx.y, threadIdx.z)];
            pop(label_constant<2>()) = s_pop[label_constant<1 * block::stride()>() + device::idxBlock(xp1, threadIdx.y, threadIdx.z)];
            pop(label_constant<3>()) = s_pop[label_constant<2 * block::stride()>() + device::idxBlock(threadIdx.x, ym1, threadIdx.z)];
            pop(label_constant<4>()) = s_pop[label_constant<3 * block::stride()>() + device::idxBlock(threadIdx.x, yp1, threadIdx.z)];
            pop(label_constant<5>()) = s_pop[label_constant<4 * block::stride()>() + device::idxBlock(threadIdx.x, threadIdx.y, zm1)];
            pop(label_constant<6>()) = s_pop[label_constant<5 * block::stride()>() + device::idxBlock(threadIdx.x, threadIdx.y, zp1)];
            pop(label_constant<7>()) = s_pop[label_constant<6 * block::stride()>() + device::idxBlock(xm1, ym1, threadIdx.z)];
            pop(label_constant<8>()) = s_pop[label_constant<7 * block::stride()>() + device::idxBlock(xp1, yp1, threadIdx.z)];
            pop(label_constant<9>()) = s_pop[label_constant<8 * block::stride()>() + device::idxBlock(xm1, threadIdx.y, zm1)];
            pop(label_constant<10>()) = s_pop[label_constant<9 * block::stride()>() + device::idxBlock(xp1, threadIdx.y, zp1)];
            pop(label_constant<11>()) = s_pop[label_constant<10 * block::stride()>() + device::idxBlock(threadIdx.x, ym1, zm1)];
            pop(label_constant<12>()) = s_pop[label_constant<11 * block::stride()>() + device::idxBlock(threadIdx.x, yp1, zp1)];
            pop(label_constant<13>()) = s_pop[label_constant<12 * block::stride()>() + device::idxBlock(xm1, yp1, threadIdx.z)];
            pop(label_constant<14>()) = s_pop[label_constant<13 * block::stride()>() + device::idxBlock(xp1, ym1, threadIdx.z)];
            pop(label_constant<15>()) = s_pop[label_constant<14 * block::stride()>() + device::idxBlock(xm1, threadIdx.y, zp1)];
            pop(label_constant<16>()) = s_pop[label_constant<15 * block::stride()>() + device::idxBlock(xp1, threadIdx.y, zm1)];
            pop(label_constant<17>()) = s_pop[label_constant<16 * block::stride()>() + device::idxBlock(threadIdx.x, ym1, zp1)];
            pop(label_constant<18>()) = s_pop[label_constant<17 * block::stride()>() + device::idxBlock(threadIdx.x, yp1, zm1)];
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

        // __device__ [[nodiscard]] static inline label_t blockIndex(const label_t tx, const label_t ty, const label_t tz) noexcept
        // {
        //     return tx + block::nx() * (ty + block::ny() * tz);
        // }

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