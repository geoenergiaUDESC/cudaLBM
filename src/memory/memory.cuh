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
    Memory management routines for the LBM code

Namespace
    LBM::host, LBM::device

SourceFiles
    memory.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MEMORY_CUH
#define __MBLBM_MEMORY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"

namespace LBM
{
    namespace host
    {
        /**
         * @brief Copies data from device memory to host memory
         * @tparam T Data type of the elements
         * @param[in] devPtr Pointer to device memory to copy from
         * @param[in] nPoints Number of elements to copy
         * @return std::vector<T> containing the copied data
         * @throws std::runtime_error if CUDA memory copy fails
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> toHost(const T *const ptrRestrict devPtr, const std::size_t nPoints)
        {
            std::vector<T> hostFields(nPoints, 0);

            if (devPtr == nullptr)
            {
                std::cout << "Null pointer!" << std::endl;
            }

            const cudaError_t err = cudaMemcpy(hostFields.data(), devPtr, nPoints * sizeof(T), cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyDeviceToHost failed: " + std::string(cudaGetErrorString(err)));
            }

            return hostFields;
        }

        /**
         * @brief Copies multiple device arrays to host and interleaves them
         * @tparam M Mesh type
         * @tparam T Data type of the elements
         * @tparam nVars Number of variables (arrays) to copy
         * @param[in] devPtrs Collection of device pointers to copy from
         * @param[in] mesh Mesh object providing dimension information
         * @return std::vector<T> containing interleaved data from all arrays
         *
         * This function copies multiple device arrays to host memory and
         * interleaves them in AoSoA (Array of Structures of Array) format
         **/
        template <class M, typename T, const label_t nVars>
        __host__ [[nodiscard]] const std::vector<T> toHost(
            const device::ptrCollection<nVars, T> &devPtrs,
            const M &mesh)
        {
            // Allocate size and all to 0
            std::vector<T> arr(mesh.nPoints() * nVars, 0);

            // Create a run-time indexable array of pointers
            const std::array<T *, nVars> ptrs = devPtrs.to_array();

            // Now do the copy
            for (std::size_t var = 0; var < nVars; var++)
            {
                const std::vector<T> f_temp = host::toHost(ptrs[var], mesh.nPoints());

                for (label_t bz = 0; bz < mesh.nzBlocks(); bz++)
                {
                    for (label_t by = 0; by < mesh.nyBlocks(); by++)
                    {
                        for (label_t bx = 0; bx < mesh.nxBlocks(); bx++)
                        {
                            for (label_t tz = 0; tz < block::nz(); tz++)
                            {
                                for (label_t ty = 0; ty < block::ny(); ty++)
                                {
                                    for (label_t tx = 0; tx < block::nx(); tx++)
                                    {
                                        arr[host::idx(tx, ty, tz, bx, by, bz, mesh) + (var * mesh.nPoints())] = f_temp[host::idx(tx, ty, tz, bx, by, bz, mesh)];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return arr;
        }
    }

    namespace device
    {
        /**
         * @brief Allocates memory on the device
         * @tparam T Data type to allocate
         * @param[out] ptr Pointer to be allocated
         * @param[in] nPoints Number of elements to allocate
         * @throws std::runtime_error if CUDA allocation fails
         **/
        template <typename T>
        __host__ void allocateMemory(T **ptr, const std::size_t nPoints)
        {
            const cudaError_t err = cudaMalloc(ptr, sizeof(T) * nPoints);

            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
            }
        }

        /**
         * @brief Allocates and returns a pointer to device memory
         * @tparam T Data type to allocate
         * @param[in] nPoints Number of elements to allocate
         * @return Pointer to allocated device memory
         * @throws std::runtime_error if CUDA allocation fails
         * @note Verbose mode prints allocation details
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const std::size_t nPoints) noexcept
        {
            T *ptr;

            allocateMemory(&ptr, nPoints);

#ifdef VERBOSE
            std::cout << "Allocated " << sizeof(T) * nPoints << " bytes of memory in cudaMalloc to address " << ptr << std::endl;
#endif

            return ptr;
        }

        /**
         * @brief Copies data from host to device memory
         * @tparam T Data type of the elements
         * @param[out] ptr Destination device pointer
         * @param[in] f Source host vector
         * @throws std::runtime_error if CUDA memory copy fails
         * @note Verbose mode prints copy details
         **/
        template <typename T>
        __host__ void copy(T *const ptr, const std::vector<T> &f)
        {
            const cudaError_t err = cudaMemcpy(ptr, f.data(), f.size() * sizeof(T), cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyHostToDevice failed: " + std::string(cudaGetErrorString(err)));
            }
            else
            {
#ifdef VERBOSE
                std::cout << "Copied " << sizeof(T) * f.size() << " bytes of memory in cudaMemcpy to address " << ptr << std::endl;
#endif
            }
        }

        /**
         * @brief Allocates device memory and copies host data to it
         * @tparam T Data type of the elements
         * @param[in] f Host vector to copy to device
         * @return Pointer to allocated device memory containing copied data
         * @throws std::runtime_error if CUDA operations fail
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const std::vector<T> &f) noexcept
        {
            T *ptr = allocate<T>(f.size());

            copy(ptr, f);

            return ptr;
        }

        /**
         * @brief Allocates device memory and initializes it with a value
         * @tparam T Data type of the elements
         * @param[in] nPoints Number of elements to allocate
         * @param[in] val Value to initialize all elements with
         * @return Pointer to allocated and initialized device memory
         * @throws std::runtime_error if CUDA operations fail
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const label_t nPoints, const T val) noexcept
        {
            T *ptr = allocate<T>(nPoints);

            copy(ptr, std::vector<T>(nPoints, val));

            return ptr;
        }
    }
}

#include "cache.cuh"

#endif