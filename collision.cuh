/**
Filename: collision.cuh
Contents: Definition of the collision GPU kernel
**/

#ifndef __MBLBM_COLLISION_CUH
#define __MBLBM_COLLISION_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "velocitySet/velocitySet.cuh"
#include "globalFunctions.cuh"

namespace LBM
{
    /**
     * @brief Performs the collision operation
     * @param moments The 10 solution moments
     **/
    __device__ static inline void collide(scalar_t (&ptrRestrict moments)[10]) noexcept
    {
        // const scalar_t omegaVar = d_omega;
        const scalar_t t_omegaVar = static_cast<scalar_t>(1) - d_omega;
        const scalar_t omegaVar_d2 = d_omega * static_cast<scalar_t>(0.5);

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

    // __device__ static inline void collide_v2(momentArray &ptrRestrict moments) noexcept
    // {
    //     // const scalar_t omegaVar = d_omega;
    //     const scalar_t t_omegaVar = static_cast<scalar_t>(1) - d_omega;
    //     const scalar_t omegaVar_d2 = d_omega * static_cast<scalar_t>(0.5);

    //     // Velocity updates are removed since force terms are zero
    //     // Diagonal moment updates (remove force terms)
    //     moments.m_xx = t_omegaVar * moments.m_xx + omegaVar_d2 * (moments.u) * (moments.u);
    //     moments.m_yy = t_omegaVar * moments.m_yy + omegaVar_d2 * (moments.v) * (moments.v);
    //     moments.m_zz = t_omegaVar * moments.m_zz + omegaVar_d2 * (moments.w) * (moments.w);

    //     // Off-diagonal moment updates (remove force terms)
    //     moments.m_xy = t_omegaVar * moments.m_xy + d_omega * (moments.u) * (moments.v);
    //     moments.m_xz = t_omegaVar * moments.m_xz + d_omega * (moments.u) * (moments.w);
    //     moments.m_yz = t_omegaVar * moments.m_yz + d_omega * (moments.v) * (moments.w);
    // }
}

#endif