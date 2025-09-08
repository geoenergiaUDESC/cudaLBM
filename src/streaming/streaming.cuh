/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Class handling the streaming step

Namespace
    LBM

SourceFiles
    streaming.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_STREAMING_CUH
#define __MBLBM_STREAMING_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"

namespace LBM
{
    /**
     * @class streaming
     * @brief Handles the streaming step in Lattice Boltzmann Method simulations
     *
     * This class manages the streaming (propagation) step of the LBM algorithm,
     * where particle distributions move to neighboring lattice sites. It provides
     * efficient shared memory operations for storing and retrieving population
     * data with optimized periodic boundary handling.
     */
    class streaming
    {
    public:
        /**
         * @brief Default constructor
         */
        [[nodiscard]] inline consteval streaming() {};

        /**
         * @brief Saves thread population density to shared memory
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         * @tparam N Size of shared memory array
         * @param[in] pop Population density array for current thread
         * @param[out] s_pop Shared memory array for population storage
         * @param[in] tid Thread ID within block
         *
         * This method stores population data from individual threads into
         * shared memory for efficient access during the streaming step.
         * It uses compile-time loop unrolling for optimal performance.
         */
        template <class VelocitySet, const label_t N>
        __device__ static inline void save(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, N> &s_pop,
            const label_t tid) noexcept
        {
            device::constexpr_for<0, (VelocitySet::Q() - 1)>(
                [&](const auto q_)
                {
                    // const label_t idx = q_ * block::stride() + tid;
                    s_pop[label_constant<q_ * block::stride()>() + tid] = pop(label_constant<q_ + 1>());
                });
        }

        /**
         * @brief Pulls population density from shared memory with periodic boundaries
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         * @tparam N Size of shared memory array
         * @param[out] pop Population density array to be populated
         * @param[in] s_pop Shared memory array containing population data
         *
         * This method retrieves population data from shared memory, applying
         * periodic boundary conditions to handle data exchange between threads
         * at block boundaries. It implements the D3Q19 streaming pattern.
         */
        template <class VelocitySet, const label_t N>
        __device__ static inline void pull(
            thread::array<scalar_t, VelocitySet::Q()> &pop,
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
         * @brief Computes linear index for population data within a block
         * @tparam pop Population component index
         * @param[in] tx Thread x-coordinate within block
         * @param[in] ty Thread y-coordinate within block
         * @param[in] tz Thread z-coordinate within block
         * @return Linearized index in shared memory
         *
         * Memory layout: [pop][tz][ty][tx] (pop slowest varying, tx fastest)
         */
        template <const label_t pop>
        __device__ [[nodiscard]] static inline label_t idxPopBlock(const label_t tx, const label_t ty, const label_t tz) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (pop)));
        }

        /**
         * @brief Computes linear index for population data using dim3 coordinates
         * @tparam pop Population component index
         * @param[in] tx Thread coordinates as dim3 structure
         * @return Linearized index in shared memory
         */
        template <const label_t pop>
        __device__ [[nodiscard]] static inline label_t idxPopBlock(const dim3 &tx) noexcept
        {
            return idxPopBlock<pop>(tx.x, tx.y, tx.z);
        }

        /**
         * @brief Computes periodic boundary index with optimization for power-of-two dimensions
         * @tparam Shift Direction shift (-1 for backward, +1 for forward)
         * @tparam Dim Dimension size (periodic length)
         * @param[in] idx Current index position
         * @return Shifted index with periodic wrapping
         *
         * This function uses bitwise AND optimization when Dim is power-of-two
         * for improved performance, falling back to modulo arithmetic otherwise.
         */
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