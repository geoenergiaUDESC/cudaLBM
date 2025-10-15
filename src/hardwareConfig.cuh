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
    Compile-time constants for the GPU

Namespace
    LBM, LBM::block

SourceFiles
    hardwareConfig.cuh

\*---------------------------------------------------------------------------*/

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
        template <class VelocitySet, const label_t nVars>
        __device__ __host__ [[nodiscard]] inline consteval label_t sharedMemoryBufferSize(const label_t size = 1) noexcept
        {
            constexpr const label_t A = (VelocitySet::Q() - 1) * block::stride();
            constexpr const label_t B = block::size() * (nVars + 1);
            return (A > B ? A : B) * size;
        }

        /**
         * @brief Launch bounds information
         * @note These variables are device specific - enable modification later
         **/
        __host__ [[nodiscard]] inline consteval label_t maxThreads() noexcept
        {
            return block::nx() * block::ny() * block::nz();
        }
    }
}

#endif