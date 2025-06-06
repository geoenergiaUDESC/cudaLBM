/**
Filename: boundaryConditions.cuh
Contents: A class applying boundary conditions to the lid driven cavity case
**/

#ifndef __MBLBM_BOUNDARYCONDITIONS_CUH
#define __MBLBM_BOUNDARYCONDITIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    class boundaryConditions
    {
    public:
        [[nodiscard]] inline consteval boundaryConditions() {};

        // This is way too long but seems to work for now
        __device__ static inline void calculateMoments(
            const scalar_t pop[19],
            scalar_t *const rhoVar,
            scalar_t *const ux_t30,
            scalar_t *const uy_t30,
            scalar_t *const uz_t30,
            scalar_t *const m_xx_t45,
            scalar_t *const m_xy_t90,
            scalar_t *const m_xz_t90,
            scalar_t *const m_yy_t45,
            scalar_t *const m_yz_t90,
            scalar_t *const m_zz_t45,
            const nodeType_t nodeType) noexcept
        {
            scalar_t rho_I;
            scalar_t inv_rho_I;

            scalar_t m_xx_I;
            scalar_t m_xy_I;
            scalar_t m_xz_I;
            scalar_t m_yy_I;
            scalar_t m_yz_I;
            scalar_t m_zz_I;

            scalar_t rho;
            // scalar_t inv_rho;

            constexpr scalar_t omegaVar = OMEGA;

            switch (nodeType)
            {
            case SOUTH_WEST_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12]);
                m_xy_I = inv_rho_I * (pop[8]);
                m_xz_I = inv_rho_I * (pop[10]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12]);
                m_yz_I = inv_rho_I * (pop[12]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = -(14 * m_xy_I - 14 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 2 * m_yz_I - 2 * m_zz_I - 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = -(14 * m_xx_I - 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = -(14 * m_xy_I - 2 * m_xx_I + 2 * m_xz_I - 14 * m_yy_I + 14 * m_yz_I - 2 * m_zz_I + 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I - 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = -(2 * m_xx_I - 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = -(2 * m_xy_I - 2 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 14 * m_yz_I - 14 * m_zz_I + 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 21 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case SOUTH_WEST_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8]);
                m_xz_I = inv_rho_I * (-pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (-pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = (14 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = (14 * m_xx_I - 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = (2 * m_xx_I - 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = (2 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = (2 * m_xx_I - 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I + 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case NORTH_WEST_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (-pop[14]);
                m_xz_I = inv_rho_I * (pop[10]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (-pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = -(14 * m_xx_I + 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I - 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = (2 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case NORTH_WEST_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);
                m_xy_I = inv_rho_I * (-pop[14]);
                m_xz_I = inv_rho_I * (-pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16]);
                m_yz_I = inv_rho_I * (pop[11]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = (14 * m_xx_I + 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = -(2 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case SOUTH_EAST_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);
                m_xy_I = inv_rho_I * (-pop[13]);
                m_xz_I = inv_rho_I * (-pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15]);
                m_yz_I = inv_rho_I * (pop[12]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = (14 * m_xx_I + 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = -(2 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case SOUTH_EAST_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (-pop[13]);
                m_xz_I = inv_rho_I * (pop[9]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (-pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = -(14 * m_xx_I + 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I - 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = (2 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case NORTH_EAST_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7]);
                m_xz_I = inv_rho_I * (-pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (-pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = (14 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = (14 * m_xx_I - 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = (2 * m_xx_I - 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = (2 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = (2 * m_xx_I - 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I + 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case NORTH_EAST_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11]);
                m_xy_I = inv_rho_I * (pop[7]);
                m_xz_I = inv_rho_I * (pop[9]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11]);
                m_yz_I = inv_rho_I * (pop[11]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

                *m_xx_t45 = -(14 * m_xy_I - 14 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 2 * m_yz_I - 2 * m_zz_I - 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_xz_t90 = -(14 * m_xx_I - 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yy_t45 = -(14 * m_xy_I - 2 * m_xx_I + 2 * m_xz_I - 14 * m_yy_I + 14 * m_yz_I - 2 * m_zz_I + 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I - 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_yz_t90 = -(2 * m_xx_I - 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
                *m_zz_t45 = -(2 * m_xy_I - 2 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 14 * m_yz_I - 14 * m_zz_I + 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 21 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

                *rhoVar = rho;

                break;
            case SOUTH_WEST:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8]);
                m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I - 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (2412 * m_xy_I - 504 * m_xx_I - 504 * m_yy_I + 216 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I + 79 * omegaVar * m_yy_I + 34 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (216 * m_xx_I - 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = -(72 * m_xx_I - 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case NORTH_WEST:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (-pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I + 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (504 * m_xx_I + 2412 * m_xy_I + 504 * m_yy_I - 216 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I - 79 * omegaVar * m_yy_I - 34 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (216 * m_xx_I + 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = -(72 * m_xx_I + 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case SOUTH_EAST:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (-pop[13]);
                m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I + 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (504 * m_xx_I + 2412 * m_xy_I + 504 * m_yy_I - 216 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I - 79 * omegaVar * m_yy_I - 34 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (216 * m_xx_I + 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = -(72 * m_xx_I + 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case NORTH_EAST:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7]);
                m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I - 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (2412 * m_xy_I - 504 * m_xx_I - 504 * m_yy_I + 216 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I + 79 * omegaVar * m_yy_I + 34 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (216 * m_xx_I - 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = -(72 * m_xx_I - 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case WEST_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (2412 * m_xz_I - 504 * m_xx_I + 216 * m_yy_I - 504 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I + 34 * omegaVar * m_yy_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = -(72 * m_xx_I - 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case WEST_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                m_xz_I = inv_rho_I * (-pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (504 * m_xx_I + 2412 * m_xz_I - 216 * m_yy_I + 504 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I - 34 * omegaVar * m_yy_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = -(72 * m_xx_I + 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case EAST_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                m_xz_I = inv_rho_I * (-pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (504 * m_xx_I + 2412 * m_xz_I - 216 * m_yy_I + 504 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I - 34 * omegaVar * m_yy_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = -(72 * m_xx_I + 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case EAST_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = (936 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (2412 * m_xz_I - 504 * m_xx_I + 216 * m_yy_I - 504 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I + 34 * omegaVar * m_yy_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = -(72 * m_xx_I - 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case SOUTH_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);
                m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15]);
                m_yz_I = inv_rho_I * (pop[12]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I - 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I - 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I - 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (216 * m_xx_I - 504 * m_yy_I + 2412 * m_yz_I - 504 * m_zz_I + 34 * omegaVar * m_xx_I + 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case SOUTH_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (-pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I + 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I - 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I + 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I + 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (504 * m_yy_I - 216 * m_xx_I + 2412 * m_yz_I + 504 * m_zz_I - 34 * omegaVar * m_xx_I - 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case NORTH_BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (-pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I + 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I - 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I + 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I + 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (504 * m_yy_I - 216 * m_xx_I + 2412 * m_yz_I + 504 * m_zz_I - 34 * omegaVar * m_xx_I - 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case NORTH_FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);
                m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16]);
                m_yz_I = inv_rho_I * (pop[11]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I - 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

                *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I - 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I - 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_yz_t90 = (216 * m_xx_I - 504 * m_yy_I + 2412 * m_yz_I - 504 * m_zz_I + 34 * omegaVar * m_xx_I + 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
                *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

                *rhoVar = rho;

                break;
            case WEST:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4)) / (omegaVar + 9);

                *m_xx_t45 = (15 * m_xx_I + 2) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_xy_t90 = (2 * m_xy_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_yy_t45 = (4 * (omegaVar + 9) * (10 * m_yy_I - m_zz_I)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_yz_t90 = (m_yz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_zz_t45 = -(4 * (m_yy_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));

                *rhoVar = rho;

                break;
            case EAST:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4)) / (omegaVar + 9);

                *m_xx_t45 = (15 * m_xx_I + 2) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_xy_t90 = (2 * m_xy_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_yy_t45 = (4 * (omegaVar + 9) * (10 * m_yy_I - m_zz_I)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_yz_t90 = (m_yz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
                *m_zz_t45 = -(4 * (m_yy_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));

                *rhoVar = rho;

                break;
            case SOUTH:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4)) / (omegaVar + 9);

                *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_zz_I)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_xy_t90 = (2 * m_xy_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_xz_t90 = (m_xz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_yy_t45 = (15 * m_yy_I + 2) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_zz_t45 = -(4 * (m_xx_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));

                *rhoVar = rho;

                break;
            case NORTH:
                *ux_t30 = U_MAX;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);

                rho = (3 * rho_I * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4)) / (omegaVar + 9);

                *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_zz_I)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_xy_t90 = (18 * m_xy_I - 4 * U_MAX + 2 * omegaVar * m_xy_I - 3 * U_MAX * m_yy_I + 3 * omegaVar * U_MAX * m_yy_I) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_xz_t90 = (m_xz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_yy_t45 = (15 * m_yy_I + 2) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
                *m_zz_t45 = -(4 * (m_xx_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));

                *rhoVar = rho;

                break;
            case BACK:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (3 * rho_I * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4)) / (omegaVar + 9);

                *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_yy_I)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_xy_t90 = (m_xy_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_yy_t45 = -(4 * (m_xx_I - 10 * m_yy_I) * (omegaVar + 9)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_zz_t45 = (15 * m_zz_I + 2) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));

                *rhoVar = rho;

                break;
            case FRONT:
                *ux_t30 = 0;
                *uy_t30 = 0;
                *uz_t30 = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
                m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4)) / (omegaVar + 9);

                *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_yy_I)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_xy_t90 = (m_xy_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_yy_t45 = -(4 * (m_xx_I - 10 * m_yy_I) * (omegaVar + 9)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
                *m_zz_t45 = (15 * m_zz_I + 2) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));

                *rhoVar = rho;

                break;
            }
        }

        __host__ [[nodiscard]] inline static nodeType_t initialCondition(
            const label_t x,
            const label_t y,
            const label_t z)
        {
            if (y == 0 && x == 0 && z == 0) // SWB
            {
                return SOUTH_WEST_BACK;
            }
            else if (y == 0 && x == 0 && z == (NZ_TOTAL - 1)) // SWF
            {
                return SOUTH_WEST_FRONT;
            }
            else if (y == 0 && x == (NX - 1) && z == 0) // SEB
            {
                return SOUTH_EAST_BACK;
            }
            else if (y == 0 && x == (NX - 1) && z == (NZ_TOTAL - 1)) // SEF
            {
                return SOUTH_EAST_FRONT;
            }
            else if (y == (NY - 1) && x == 0 && z == 0) // NWB
            {
                return NORTH;
            }
            else if (y == (NY - 1) && x == 0 && z == (NZ_TOTAL - 1)) // NWF
            {
                return NORTH;
            }
            else if (y == (NY - 1) && x == (NX - 1) && z == 0) // NEB
            {
                return NORTH;
            }
            else if (y == (NY - 1) && x == (NX - 1) && z == (NZ_TOTAL - 1)) // NEF
            {
                return NORTH;
            }
            else if (y == 0 && x == 0) // SW
            {
                return SOUTH_WEST;
            }
            else if (y == 0 && x == (NX - 1)) // SE
            {
                return SOUTH_EAST;
            }
            else if (y == (NY - 1) && x == 0) // NW
            {
                return NORTH;
            }
            else if (y == (NY - 1) && x == (NX - 1)) // NE
            {
                return NORTH;
            }
            else if (y == 0 && z == 0) // SB
            {
                return SOUTH_BACK;
            }
            else if (y == 0 && z == (NZ_TOTAL - 1)) // SF
            {
                return SOUTH_FRONT;
            }
            else if (y == (NY - 1) && z == 0) // NB
            {
                return NORTH;
            }
            else if (y == (NY - 1) && z == (NZ_TOTAL - 1)) // NF
            {
                return NORTH;
            }
            else if (x == 0 && z == 0) // WB
            {
                return WEST_BACK;
            }
            else if (x == 0 && z == (NZ_TOTAL - 1)) // WF
            {
                return WEST_FRONT;
            }
            else if (x == (NX - 1) && z == 0) // EB
            {
                return EAST_BACK;
            }
            else if (x == (NX - 1) && z == (NZ_TOTAL - 1)) // EF
            {
                return EAST_FRONT;
            }
            else if (y == 0) // S
            {
                return SOUTH;
            }
            else if (y == (NY - 1)) // N
            {
                return NORTH;
            }
            else if (x == 0) // W
            {
                return WEST;
            }
            else if (x == (NX - 1)) // E
            {
                return EAST;
            }
            else if (z == 0) // B
            {
                return BACK;
            }
            else if (z == (NZ_TOTAL - 1)) // F
            {
                return FRONT;
            }
            else
            {
                return BULK;
            }
        }

    private:
    };
}

#endif