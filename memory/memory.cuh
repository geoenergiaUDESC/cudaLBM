/**
Filename: memory.cuh
Contents: Memory management routines for the LBM code
**/

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
         * @brief Copies a device pointer of type T into an std::vector of type T on the host
         * @param devPtr Pointer to the array on the device
         * @param nPoints The number of elements contained within devPtr
         * @return An std::vector of type T copied from the device
         * @note This is currently somewhat redundant but will be taken care of later
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> toHost(const T *const ptrRestrict devPtr, const std::size_t nPoints)
        {
            std::vector<T> hostFields(nPoints, 0);

            const cudaError_t err = cudaMemcpy(hostFields.data(), devPtr, nPoints * sizeof(T), cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyDeviceToHost failed: " + std::string(cudaGetErrorString(err)));
            }

            return hostFields;
        }

        /**
         * @brief Copies a device variable to the host
         * @param devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param mesh The mesh
         * @return An std::vector of type T, interlaced into fMom format
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
                                        arr[host::idxMom(tx, ty, tz, bx, by, bz, var, mesh)] = f_temp[host::idx(tx, ty, tz, bx, by, bz, mesh)];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return arr;
        }

        /**
         * @brief Prepares a host array for allocation on the device
         * @param hostMoments The device variable array
         * @param mesh The mesh
         * @return An std::vector of type T, de-interlaced from fMom
         **/
        template <const label_t mom, class M>
        __host__ [[nodiscard]] const std::vector<scalar_t> toDevice(const std::vector<scalar_t> hostMoments, const M &mesh)
        {
            // Allocate size and all to 0
            std::vector<scalar_t> arr(mesh.nPoints(), 0);

            // Copy idxMom to idx
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
                                    arr[host::idx(tx, ty, tz, bx, by, bz, mesh)] = hostMoments[host::idxMom<mom>(tx, ty, tz, bx, by, bz, mesh)];
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
         * @brief Allocates nPoints worth of T to ptr
         * @param ptr The pointer to which the memory is to be allocated
         * @param size The number of elements to be allocated to T
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
         * @brief Allocates a block of memory on the device and returns its pointer
         * @return A raw pointer to a block of memory
         * @param size The amount of memory to be allocated
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
         * @brief Copies a vector of type T to a device pointer of type T
         * @param ptr The pointer to which the vector is to be copied
         * @param f The vector which is to be copied to ptr
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
         * @brief Allocates a scalar array on the device
         * @return A T * object pointing to a block of memory on the GPU
         * @param f The pre-existing array on the host to be copied to the GPU
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const std::vector<T> &f) noexcept
        {
            T *ptr = allocate<T>(f.size());

            copy(ptr, f);

            return ptr;
        }

        /**
         * @brief Allocates a scalar array on the device
         * @return A pointer of type T pointing to a block of memory on the GPU
         * @param nPoints The number of scalar points to be allocated to the block of memory
         * @param val The value set
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

#include "sharedMemory.cuh"
#include "cache.cuh"

#endif