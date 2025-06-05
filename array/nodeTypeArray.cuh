/**
Filename: nodeTypeArray.cuh
Contents: A class representing the types of each individual mesh node
This is a temporary fix before the boundary condition function pointers are implemented
**/

#ifndef __MBLBM_NODETYPEARRAY_CUH
#define __MBLBM_NODETYPEARRAY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
// #include "latticeMesh.cuh"

namespace mbLBM
{
    namespace host
    {
        // typedef array<nodeType::type> nodeTypeArray;
    }

    namespace device
    {
        class nodeTypeArray
        {
        public:
            /**
             * @brief Constructs a scalar solution variable on the device from a copy of its host version
             * @return A scalarArray object copied from the host to the device
             * @param f The host copy of the scalar solution variable
             **/
            [[nodiscard]] nodeTypeArray(const nodeTypeArray_t &nodeTypes) noexcept
                : ptr_(device::allocateArray<nodeType::type>(nodeTypes))
            {
#ifdef VERBOSE
                std::cout << "Allocated device node types for " << nodeTypes.size() << " lattice points" << std::endl;
#endif
            };

            /**
             * @brief Destructor
             **/
            ~nodeTypeArray() noexcept
            {
#ifdef VERBOSE
                std::cout << "Freeing device node types" << std::endl;
#endif
                cudaFree((void *)ptr_);
                // cudaFree(static_cast<void *>(ptr_));
            };

            /**
             * @brief Returns immutable access to the underlying pointer
             * @return A const-qualified pointer
             **/
            __host__ __device__ [[nodiscard]] inline const nodeType::type *ptr() const noexcept
            {
                return ptr_;
            }

        private:
            /**
             * @brief Pointer to the underlying array
             **/
            const nodeType::type *const ptrRestrict ptr_;
        };
    }
}

#endif
