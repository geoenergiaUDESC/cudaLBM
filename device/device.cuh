/**
Filename: device.cuh
Contents: Common memory allocation routines for allocating memory on the GPU
**/

#ifndef __MBLBM_DEVICE_CUH
#define __MBLBM_DEVICE_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace mbLBM
{
    namespace device
    {
        /**
         * @brief Allocates a block of memory on the device and returns its pointer
         * @return A raw pointer to a block of memory
         * @param size The amount of memory to be allocated
         **/
        template <typename T>
        [[nodiscard]] inline T *allocate(const std::size_t size) noexcept
        {
            T *ptr;
            const cudaError_t i = cudaMalloc(static_cast<T **>(&ptr), size);

            if (i != cudaSuccess)
            {
                exceptions::program_exit(i, "Unable to allocate array");
            }
#ifdef VERBOSE
            std::cout << "Allocated " << size << " bytes of memory in cudaMalloc to address " << ptr << std::endl;
#endif

            return ptr;
        }

        /**
         * @brief Allocates a scalar array on the device
         * @return A T * object pointing to a block of memory on the GPU
         * @param f The pre-existing array on the host to be copied to the GPU
         **/
        template <typename T, class F>
        [[nodiscard]] const T *allocateArray(const F &f) noexcept
        {
            T *ptr = allocate<T>(f.nPoints() * sizeof(T));

            const cudaError_t i = cudaMemcpy(ptr, &(f.arrRef()[0]), f.nPoints() * sizeof(T), cudaMemcpyHostToDevice);

            if (i != cudaSuccess)
            {
                exceptions::program_exit(i, "Unable to copy array");
            }

            return ptr;
        }

        /**
         * @brief Allocates a scalar array on the device
         * @return A T * object pointing to a block of memory on the GPU
         * @param f The pre-existing array on the host to be copied to the GPU
         **/
        template <typename T>
        [[nodiscard]] const T *allocateArray(const std::vector<T> &f) noexcept
        {
            T *ptr = allocate<T>(f.size() * sizeof(T));

            const cudaError_t i = cudaMemcpy(ptr, &(f[0]), f.size() * sizeof(T), cudaMemcpyHostToDevice);

            std::cout << "Allocated " << f.size() << " elements of size " << sizeof(T) << std::endl;

            if (i != cudaSuccess)
            {
                exceptions::program_exit(i, "Unable to copy array");
            }

            return ptr;
        }

        /**
         * @brief Allocates a scalar array on the device
         * @return A scalar_t * object pointing to a block of memory on the GPU
         * @param nPoints The number of scalar points to be allocated to the block of memory
         * @param val The value set
         **/
        template <typename T>
        [[nodiscard]] const T *allocateArray(const std::size_t nPoints, const T val) noexcept
        {
            T *ptr = allocate<T>(nPoints * sizeof(T));

            const scalarArray_t f = scalarArray_t(nPoints, val);

            const cudaError_t i = cudaMemcpy(ptr, &(f[0]), nPoints * sizeof(T), cudaMemcpyHostToDevice);

            if (i != cudaSuccess)
            {
                exceptions::program_exit(i, "Unable to set array");
            }

            return ptr;
        }
    }
}

#endif
