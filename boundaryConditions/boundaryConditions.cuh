/**
Filename: boundaryConditions.cuh
Contents: A class applying boundary conditions to the lid driven cavity case
**/

#ifndef __MBLBM_BOUNDARYCONDITIONS_CUH
#define __MBLBM_BOUNDARYCONDITIONS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

#include "normalVector.cuh"

#include "D3Q19BoundaryConditions.cuh"
#include "D3Q27BoundaryConditions.cuh"

namespace LBM
{
    class boundaryConditions
    {
    public:
        [[nodiscard]] inline consteval boundaryConditions() {};

        /**
         * @brief Calculate the moment variables at the boundary
         * @param pop The population density at the current lattice node
         * @param moments The moment variables at the current lattice node
         * @param b_n The boundary normal vector at the current lattice node
         **/
        template <class VSet>
        __device__ static inline constexpr void calculateMoments(
            const scalar_t (&ptrRestrict pop)[VSet::Q()],
            scalar_t (&ptrRestrict moments)[10],
            const normalVector &b_n) noexcept
        {
            static_assert((VSet::Q() == VelocitySet::D3Q19::Q()) | (VSet::Q() == VelocitySet::D3Q27::Q()), "Must be either a D3Q19 or D3Q27 velocity set!");

            if constexpr (VSet::Q() == 19)
            {
                apply_boundaries_D3Q19(pop, moments, b_n);
            }

            if constexpr (VSet::Q() == 27)
            {
                apply_boundaries_D3Q27(pop, moments, b_n);
            }
        }

    private:
    };
}

#endif