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
        template <class VelSet>
        class ghostInterface
        {
        public:
            /**
             * @brief Constructs 3 individual ghostPtr objects
             * @return A ghostInterface object constructed from the mesh
             * @param mesh The mesh used to define the amount of memory to allocate to the pointers
             **/
            [[nodiscard]] ghostInterface(const latticeMesh &mesh) noexcept
                : fGhost_(ghostPtrs<VelSet>(mesh)),
                  gGhost_(ghostPtrs<VelSet>(mesh)),
                  h_fGhost_(ghostPtrs<VelSet>(mesh)) {};

            ~ghostInterface() noexcept {};

            [[nodiscard]] inline constexpr ghostPtrs<VelSet> &fGhost() const noexcept
            {
                return fGhost_;
            }
            [[nodiscard]] inline constexpr ghostPtrs<VelSet> &gGhost() const noexcept
            {
                return fGhost_;
            }
            [[nodiscard]] inline constexpr ghostPtrs<VelSet> &h_fGhost() const noexcept
            {
                return h_fGhost_;
            }

        private:
            const ghostPtrs<VelSet> fGhost_;
            const ghostPtrs<VelSet> gGhost_;
            const ghostPtrs<VelSet> h_fGhost_;
        };
    }
}

#endif