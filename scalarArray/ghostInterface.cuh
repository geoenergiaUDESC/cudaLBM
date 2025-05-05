/**
Filename: ghostInterface.cuh
Contents: A handling the ghost interface on the device
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_GHOSTINTERFACE_CUH
#define __MBLBM_GHOSTINTERFACE_CUH

#include "ghostPtrs.cuh"

namespace mbLBM
{
    namespace device
    {
        class ghostInterface
        {
        public:
            /**
             * @brief Constructs 3 individual ghostPtr objects
             * @return A ghostInterface object constructed from the mesh
             * @param mesh The mesh used to define the amount of memory to allocate to the pointers
             **/
            [[nodiscard]] ghostInterface(const host::latticeMesh &mesh) noexcept
                : fGhost_(ghostPtrs(mesh)),
                  gGhost_(ghostPtrs(mesh)),
                  h_fGhost_(ghostPtrs(mesh))
            {
#ifdef VERBOSE
                std::cout << "Allocated ghostInterface object:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << mesh.nx() << ";" << std::endl;
                std::cout << "    ny = " << mesh.ny() << ";" << std::endl;
                std::cout << "    nz = " << mesh.nz() << ";" << std::endl;
                std::cout << "}" << std::endl;
#endif
            };

            /**
             * @brief Destructor for the ghostInterface class
             **/
            ~ghostInterface() noexcept
            {
#ifdef VERBOSE
                std::cout << "Freeing ghostInterface object" << std::endl;
#endif
            };

            /**
             * @brief Returns access to the ghost pointer objects
             * @return An immutable reference to the underlying ghostPtrs objects
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const ghostPtrs &fGhost() const noexcept
            {
                return fGhost_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const ghostPtrs &gGhost() const noexcept
            {
                return gGhost_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const ghostPtrs &h_fGhost() const noexcept
            {
                return h_fGhost_;
            }

        private:
            /**
             * @brief Lists of 6 unique pointers on each face of a CUDA block
             **/
            const ghostPtrs fGhost_;
            const ghostPtrs gGhost_;
            const ghostPtrs h_fGhost_;
        };
    }
}

#endif