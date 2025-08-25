/**
Filename: halo.cuh
Contents: A class handling the device halo
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_BLOCKHALO_CUH
#define __MBLBM_BLOCKHALO_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"
#include "../velocitySet/velocitySet.cuh"
#include "../latticeMesh/latticeMesh.cuh"

namespace LBM
{
    namespace device
    {
        namespace haloFaces
        {
            /**
             * @brief Consteval functions used to distinguish between halo face normal directions
             * @return An unsigned integer corresponding to the correct direction
             **/
            [[nodiscard]] static inline consteval label_t x() noexcept { return 0; }
            [[nodiscard]] static inline consteval label_t y() noexcept { return 1; }
            [[nodiscard]] static inline consteval label_t z() noexcept { return 2; }
        }
    }
}

#include "haloFace.cuh"
#include "halo.cuh"

#endif