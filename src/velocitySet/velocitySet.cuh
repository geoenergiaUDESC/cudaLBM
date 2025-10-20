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
    Top-level header file for the velocity set classes

Namespace
    LBM

SourceFiles
    D3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VELOCITYSET_CUH
#define __MBLBM_VELOCITYSET_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"

namespace LBM
{
    /**
     * @class velocitySet
     * @brief Base class for LBM velocity sets providing common constants and scaling operations
     *
     * This class serves as a base for specific velocity set implementations (e.g., D3Q19, D3Q27)
     * and provides common constants, scaling factors, and utility functions used across
     * different velocity set configurations in the Lattice Boltzmann Method.
     **/
    class velocitySet
    {
    public:
        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval velocitySet() noexcept {};

        /**
         * @brief Get the a^2 constant (3.0)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T as2() noexcept
        {
            return static_cast<T>(3.0);
        }

        /**
         * @brief Get the speed of sound squared (c^2 = 1 / 3)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T cs2() noexcept
        {
            return static_cast<T>(static_cast<double>(1.0) / static_cast<double>(3.0));
        }

        /**
         * @brief Get scaling factor for first-order moments
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_i() noexcept
        {
            return static_cast<T>(3.0);
        }

        /**
         * @brief Get scaling factor for diagonal second-order moments
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_ii() noexcept
        {
            return static_cast<T>(4.5);
        }

        /**
         * @brief Get scaling factor for off-diagonal second-order moments
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T scale_ij() noexcept
        {
            return static_cast<T>(9.0);
        }

        /**
         * @brief Apply velocity set scaling factors to moment array
         * @param[in,out] moments Array of 10 moment variables to be scaled
         *
         * This method applies the appropriate scaling factors to each moment component:
         * - First-order moments (velocity components): scaled by scale_i()
         * - Diagonal second-order moments: scaled by scale_ii()
         * - Off-diagonal second-order moments: scaled by scale_ij()
         **/
        __device__ static inline void scale(thread::array<scalar_t, 10> &moments) noexcept
        {
            // Scale the moments correctly
            moments(m_i<1>()) = scale_i<scalar_t>() * (moments(m_i<1>()));
            moments(m_i<2>()) = scale_i<scalar_t>() * (moments(m_i<2>()));
            moments(m_i<3>()) = scale_i<scalar_t>() * (moments(m_i<3>()));
            moments(m_i<4>()) = scale_ii<scalar_t>() * (moments(m_i<4>()));
            moments(m_i<5>()) = scale_ij<scalar_t>() * (moments(m_i<5>()));
            moments(m_i<6>()) = scale_ij<scalar_t>() * (moments(m_i<6>()));
            moments(m_i<7>()) = scale_ii<scalar_t>() * (moments(m_i<7>()));
            moments(m_i<8>()) = scale_ij<scalar_t>() * (moments(m_i<8>()));
            moments(m_i<9>()) = scale_ii<scalar_t>() * (moments(m_i<9>()));
        }

    private:
    };
}

#include "D3Q19.cuh"
#include "D3Q27.cuh"

#endif