/**
Filename: collision.cuh
Contents: Definition of the collision GPU kernel
**/

#ifndef __MBLBM_COLLISION_CUH
#define __MBLBM_COLLISION_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "velocitySet/velocitySet.cuh"
#include "boundaryConditions.cuh"
#include "globalFunctions.cuh"
#include "boundaryConditions.cuh"

namespace LBM
{

    [[nodiscard]] inline consteval auto MAX_THREADS_PER_BLOCK() noexcept { return 1024; }
    [[nodiscard]] inline consteval auto MIN_BLOCKS_PER_MP() noexcept { return 8; }

    __device__ static inline void collide(
        scalar_t *const ptrRestrict rhoVar,
        scalar_t *const ptrRestrict ux_t30,
        scalar_t *const ptrRestrict uy_t30,
        scalar_t *const ptrRestrict uz_t30,
        scalar_t *const ptrRestrict m_xx_t45,
        scalar_t *const ptrRestrict m_xy_t90,
        scalar_t *const ptrRestrict m_xz_t90,
        scalar_t *const ptrRestrict m_yy_t45,
        scalar_t *const ptrRestrict m_yz_t90,
        scalar_t *const ptrRestrict m_zz_t45) noexcept
    {
        // const scalar_t omegaVar = d_omega;
        const scalar_t t_omegaVar = 1 - d_omega;
        const scalar_t omegaVar_d2 = d_omega / 2;

        // Velocity updates are removed since force terms are zero
        // Diagonal moment updates (remove force terms)
        *m_xx_t45 = t_omegaVar * *m_xx_t45 + omegaVar_d2 * (*ux_t30) * (*ux_t30);
        *m_yy_t45 = t_omegaVar * *m_yy_t45 + omegaVar_d2 * (*uy_t30) * (*uy_t30);
        *m_zz_t45 = t_omegaVar * *m_zz_t45 + omegaVar_d2 * (*uz_t30) * (*uz_t30);

        // Off-diagonal moment updates (remove force terms)
        *m_xy_t90 = t_omegaVar * *m_xy_t90 + d_omega * (*ux_t30) * (*uy_t30);
        *m_xz_t90 = t_omegaVar * *m_xz_t90 + d_omega * (*ux_t30) * (*uz_t30);
        *m_yz_t90 = t_omegaVar * *m_yz_t90 + d_omega * (*uy_t30) * (*uz_t30);
    }

    __device__ static inline void collide(
        scalar_t moments[10]) noexcept
    {
        // const scalar_t omegaVar = d_omega;
        const scalar_t t_omegaVar = 1 - d_omega;
        const scalar_t omegaVar_d2 = d_omega / 2;

        // Velocity updates are removed since force terms are zero
        // Diagonal moment updates (remove force terms)
        moments[4] = t_omegaVar * moments[4] + omegaVar_d2 * (moments[1]) * (moments[1]);
        moments[7] = t_omegaVar * moments[7] + omegaVar_d2 * (moments[2]) * (moments[2]);
        moments[9] = t_omegaVar * moments[9] + omegaVar_d2 * (moments[3]) * (moments[3]);

        // Off-diagonal moment updates (remove force terms)
        moments[5] = t_omegaVar * moments[5] + d_omega * (moments[1]) * (moments[2]);
        moments[6] = t_omegaVar * moments[6] + d_omega * (moments[1]) * (moments[3]);
        moments[8] = t_omegaVar * moments[8] + d_omega * (moments[2]) * (moments[3]);
    }
}

#endif