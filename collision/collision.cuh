/**
Filename: collision.cuh
Contents: Definition of the collision GPU kernel
**/

#ifndef __MBLBM_COLLISION_CUH
#define __MBLBM_COLLISION_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../velocitySet/velocitySet.cuh"
#include "../globalFunctions.cuh"

namespace LBM
{
    class collision
    {
    public:
        /**
         * @brief Constructor for the collision class
         * @return A collision object
         * @note This constructor is consteval
         **/
        [[nodiscard]] inline consteval collision() noexcept {};

    private:
    };
}

#include "secondOrder.cuh"

#endif