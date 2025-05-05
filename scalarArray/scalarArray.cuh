/**
Filename: scalarArray.cuh
Contents: A class representing a scalar variable
The host version of this class is principally used to
partition the mesh prior to distribution to the devices
**/

#ifndef __MBLBM_SCALARARRAYS_CUH
#define __MBLBM_SCALARARRAYS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "latticeMesh.cuh"
#include "../device/device.cuh"
#include "../array/array.cuh"

namespace mbLBM
{
    namespace host
    {
        typedef array<scalar_t> scalarArray;
    }

    namespace device
    {
        class scalarArray
        {
        public:
            /**
             * @brief Constructs a scalar solution variable on the device from a copy of its host version
             * @return A scalarArray object copied from the host to the device
             * @param f The host copy of the scalar solution variable
             **/
            [[nodiscard]] scalarArray(const host::scalarArray &f) noexcept
                : ptr_(device::allocateArray<scalar_t, host::scalarArray>(f)) {};

            /**
             * @brief Constructs a scalar solution variable from an arbitrary number of points and a value
             * @return A scalarArray object of arbitrary number of points and value
             * @param nPoints The number of lattice points to assign
             * @param value The value to assign to all points of the array
             **/
            [[nodiscard]] scalarArray(const std::size_t nPoints, const scalar_t value) noexcept
                : ptr_(device::allocateArray(nPoints, value)) {};

            /**
             * @brief Destructor
             **/
            ~scalarArray() noexcept
            {
#ifdef VERBOSE
                std::cout << "Freeing scalar array" << std::endl;
#endif
                cudaFree((void *)ptr_);
            };

            /**
             * @brief Returns immutable access to the underlying pointer
             * @return A const-qualified pointer
             **/
            __device__ [[nodiscard]] inline constexpr const scalar_t *ptr() const noexcept
            {
                return ptr_;
            }

        private:
            /**
             * @brief Pointer to the underlying variable
             **/
            const scalar_t *ptr_;

            /**
             * @brief Allocates a block of memory on the device and returns its pointer
             * @return A raw pointer to a block of memory
             * @param size The amount of memory to be allocated
             **/
            template <typename T>
            [[nodiscard]] inline T *deviceMalloc(const std::size_t size) const noexcept
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
        };
    }
}

#include "ghostInterface.cuh"

#endif
