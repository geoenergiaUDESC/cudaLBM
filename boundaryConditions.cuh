/**
Filename: boundaryConditions.cuh
Contents: A class applying boundary conditions to the lid driven cavity case
**/

#ifndef __MBLBM_BOUNDARYCONDITIONS_CUH
#define __MBLBM_BOUNDARYCONDITIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    class boundaryConditions
    {
    public:
        [[nodiscard]] inline consteval boundaryConditions() {};

        template <class VSet>
        __device__ static inline void calculateMoments(
            const scalar_t pop[VSet::Q()],
            scalar_t (&ptrRestrict moments)[10],
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

            switch (nodeType)
            {
            case SOUTH_WEST_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12]);
                m_xy_I = inv_rho_I * (pop[8]);
                m_xz_I = inv_rho_I * (pop[10]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12]);
                m_yz_I = inv_rho_I * (pop[12]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I + 2 * d_omega * m_xy_I * rho_I + 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I + 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = -(14 * m_xy_I - 14 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 2 * m_yz_I - 2 * m_zz_I - 21 * d_omega * m_xx_I + 7 * d_omega * m_xy_I + 7 * d_omega * m_xz_I + 9 * d_omega * m_yy_I - 23 * d_omega * m_yz_I + 9 * d_omega * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = -(14 * m_xx_I - 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I - 69 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = -(14 * m_xx_I - 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I + 21 * d_omega * m_xy_I - 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I + 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = -(14 * m_xy_I - 2 * m_xx_I + 2 * m_xz_I - 14 * m_yy_I + 14 * m_yz_I - 2 * m_zz_I + 9 * d_omega * m_xx_I + 7 * d_omega * m_xy_I - 23 * d_omega * m_xz_I - 21 * d_omega * m_yy_I + 7 * d_omega * m_yz_I + 9 * d_omega * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = -(2 * m_xx_I - 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I + 21 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = -(2 * m_xy_I - 2 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 14 * m_yz_I - 14 * m_zz_I + 9 * d_omega * m_xx_I - 23 * d_omega * m_xy_I + 7 * d_omega * m_xz_I + 9 * d_omega * m_yy_I + 7 * d_omega * m_yz_I - 21 * d_omega * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case SOUTH_WEST_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8]);
                m_xz_I = inv_rho_I * (-pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (-pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I + 2 * d_omega * m_xy_I * rho_I - 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I - 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = (14 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * d_omega * m_xx_I - 7 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I - 23 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = -(14 * m_xx_I - 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I - 69 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = (14 * m_xx_I - 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I + 21 * d_omega * m_xy_I + 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I - 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = (2 * m_xx_I - 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * d_omega * m_xx_I - 7 * d_omega * m_xy_I - 23 * d_omega * m_xz_I + 21 * d_omega * m_yy_I + 7 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = (2 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I + 21 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = (2 * m_xx_I - 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * d_omega * m_xx_I + 23 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I + 7 * d_omega * m_yz_I + 21 * d_omega * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case NORTH_WEST_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (-pop[14]);
                m_xz_I = inv_rho_I * (pop[10]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (-pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I - 2 * d_omega * m_xy_I * rho_I + 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I - 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = (14 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * d_omega * m_xx_I + 7 * d_omega * m_xy_I - 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I - 23 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = (14 * m_xx_I + 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I + 69 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = -(14 * m_xx_I + 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I - 21 * d_omega * m_xy_I - 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I - 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = (2 * m_xx_I + 14 * m_xy_I - 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * d_omega * m_xx_I + 7 * d_omega * m_xy_I + 23 * d_omega * m_xz_I + 21 * d_omega * m_yy_I + 7 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = (2 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I - 21 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = (2 * m_xx_I + 2 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * d_omega * m_xx_I - 23 * d_omega * m_xy_I - 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I + 7 * d_omega * m_yz_I + 21 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case NORTH_WEST_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);
                m_xy_I = inv_rho_I * (-pop[14]);
                m_xz_I = inv_rho_I * (-pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16]);
                m_yz_I = inv_rho_I * (pop[11]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I - 2 * d_omega * m_xy_I * rho_I - 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I + 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = (14 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 2 * m_yz_I + 2 * m_zz_I + 21 * d_omega * m_xx_I + 7 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I + 23 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = (14 * m_xx_I + 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I + 69 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = (14 * m_xx_I + 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I - 21 * d_omega * m_xy_I + 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I + 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = (2 * m_xx_I + 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I - 9 * d_omega * m_xx_I + 7 * d_omega * m_xy_I - 23 * d_omega * m_xz_I + 21 * d_omega * m_yy_I - 7 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = -(2 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I - 21 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = (2 * m_xx_I + 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I - 9 * d_omega * m_xx_I - 23 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I - 7 * d_omega * m_yz_I + 21 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case SOUTH_EAST_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);
                m_xy_I = inv_rho_I * (-pop[13]);
                m_xz_I = inv_rho_I * (-pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15]);
                m_yz_I = inv_rho_I * (pop[12]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I - 2 * d_omega * m_xy_I * rho_I - 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I + 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = (14 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 2 * m_yz_I + 2 * m_zz_I + 21 * d_omega * m_xx_I + 7 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I + 23 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = (14 * m_xx_I + 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I + 69 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = (14 * m_xx_I + 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I - 21 * d_omega * m_xy_I + 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I + 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = (2 * m_xx_I + 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I - 9 * d_omega * m_xx_I + 7 * d_omega * m_xy_I - 23 * d_omega * m_xz_I + 21 * d_omega * m_yy_I - 7 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = -(2 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I - 21 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = (2 * m_xx_I + 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I - 9 * d_omega * m_xx_I - 23 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I - 7 * d_omega * m_yz_I + 21 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case SOUTH_EAST_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (-pop[13]);
                m_xz_I = inv_rho_I * (pop[9]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (-pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);

                rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I - 2 * d_omega * m_xy_I * rho_I + 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I - 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = (14 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * d_omega * m_xx_I + 7 * d_omega * m_xy_I - 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I - 23 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = (14 * m_xx_I + 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I + 69 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = -(14 * m_xx_I + 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I - 21 * d_omega * m_xy_I - 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I - 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = (2 * m_xx_I + 14 * m_xy_I - 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * d_omega * m_xx_I + 7 * d_omega * m_xy_I + 23 * d_omega * m_xz_I + 21 * d_omega * m_yy_I + 7 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = (2 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I - 21 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = (2 * m_xx_I + 2 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * d_omega * m_xx_I - 23 * d_omega * m_xy_I - 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I + 7 * d_omega * m_yz_I + 21 * d_omega * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I - 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case NORTH_EAST_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7]);
                m_xz_I = inv_rho_I * (-pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (-pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I + 2 * d_omega * m_xy_I * rho_I - 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I - 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = (14 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * d_omega * m_xx_I - 7 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I - 23 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = -(14 * m_xx_I - 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I - 69 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = (14 * m_xx_I - 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I + 21 * d_omega * m_xy_I + 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I - 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = (2 * m_xx_I - 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * d_omega * m_xx_I - 7 * d_omega * m_xy_I - 23 * d_omega * m_xz_I + 21 * d_omega * m_yy_I + 7 * d_omega * m_yz_I - 9 * d_omega * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = (2 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I + 21 * d_omega * m_xy_I - 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = (2 * m_xx_I - 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * d_omega * m_xx_I + 23 * d_omega * m_xy_I + 7 * d_omega * m_xz_I - 9 * d_omega * m_yy_I + 7 * d_omega * m_yz_I + 21 * d_omega * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I - 2 * d_omega * m_xz_I - d_omega * m_yy_I - 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case NORTH_EAST_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11]);
                m_xy_I = inv_rho_I * (pop[7]);
                m_xz_I = inv_rho_I * (pop[9]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11]);
                m_yz_I = inv_rho_I * (pop[11]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11]);

                rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - d_omega * m_xx_I * rho_I + 2 * d_omega * m_xy_I * rho_I + 2 * d_omega * m_xz_I * rho_I - d_omega * m_yy_I * rho_I + 2 * d_omega * m_yz_I * rho_I - d_omega * m_zz_I * rho_I)) / (5 * d_omega + 2);

                moments[4] = -(14 * m_xy_I - 14 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 2 * m_yz_I - 2 * m_zz_I - 21 * d_omega * m_xx_I + 7 * d_omega * m_xy_I + 7 * d_omega * m_xz_I + 9 * d_omega * m_yy_I - 23 * d_omega * m_yz_I + 9 * d_omega * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[5] = -(14 * m_xx_I - 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * d_omega * m_xx_I - 69 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I + 21 * d_omega * m_yz_I - 23 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[6] = -(14 * m_xx_I - 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * d_omega * m_xx_I + 21 * d_omega * m_xy_I - 69 * d_omega * m_xz_I - 23 * d_omega * m_yy_I + 21 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[7] = -(14 * m_xy_I - 2 * m_xx_I + 2 * m_xz_I - 14 * m_yy_I + 14 * m_yz_I - 2 * m_zz_I + 9 * d_omega * m_xx_I + 7 * d_omega * m_xy_I - 23 * d_omega * m_xz_I - 21 * d_omega * m_yy_I + 7 * d_omega * m_yz_I + 9 * d_omega * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[8] = -(2 * m_xx_I - 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * d_omega * m_xx_I + 21 * d_omega * m_xy_I + 21 * d_omega * m_xz_I + 7 * d_omega * m_yy_I - 69 * d_omega * m_yz_I + 7 * d_omega * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));
                moments[9] = -(2 * m_xy_I - 2 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 14 * m_yz_I - 14 * m_zz_I + 9 * d_omega * m_xx_I - 23 * d_omega * m_xy_I + 7 * d_omega * m_xz_I + 9 * d_omega * m_yy_I + 7 * d_omega * m_yz_I - 21 * d_omega * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - d_omega * m_xx_I + 2 * d_omega * m_xy_I + 2 * d_omega * m_xz_I - d_omega * m_yy_I + 2 * d_omega * m_yz_I - d_omega * m_zz_I + 1));

                moments[0] = rho;

                break;
            case SOUTH_WEST:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8]);
                m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I + 57 * d_omega * m_xy_I * rho_I - 24 * d_omega * m_yy_I * rho_I + 6 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I - 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * d_omega * m_xx_I + 158 * d_omega * m_xy_I - 191 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[5] = (2412 * m_xy_I - 504 * m_xx_I - 504 * m_yy_I + 216 * m_zz_I + 79 * d_omega * m_xx_I + 538 * d_omega * m_xy_I + 79 * d_omega * m_yy_I + 34 * d_omega * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[7] = (216 * m_xx_I - 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * d_omega * m_xx_I + 158 * d_omega * m_xy_I + 239 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[9] = -(72 * m_xx_I - 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * d_omega * m_xx_I - 34 * d_omega * m_xy_I + 3 * d_omega * m_yy_I - 162 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case NORTH_WEST:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (-pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I - 57 * d_omega * m_xy_I * rho_I - 24 * d_omega * m_yy_I * rho_I + 6 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I + 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * d_omega * m_xx_I - 158 * d_omega * m_xy_I - 191 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[5] = (504 * m_xx_I + 2412 * m_xy_I + 504 * m_yy_I - 216 * m_zz_I - 79 * d_omega * m_xx_I + 538 * d_omega * m_xy_I - 79 * d_omega * m_yy_I - 34 * d_omega * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[7] = (216 * m_xx_I + 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * d_omega * m_xx_I - 158 * d_omega * m_xy_I + 239 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[9] = -(72 * m_xx_I + 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * d_omega * m_xx_I + 34 * d_omega * m_xy_I + 3 * d_omega * m_yy_I - 162 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case SOUTH_EAST:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (-pop[13]);
                m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I - 57 * d_omega * m_xy_I * rho_I - 24 * d_omega * m_yy_I * rho_I + 6 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I + 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * d_omega * m_xx_I - 158 * d_omega * m_xy_I - 191 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[5] = (504 * m_xx_I + 2412 * m_xy_I + 504 * m_yy_I - 216 * m_zz_I - 79 * d_omega * m_xx_I + 538 * d_omega * m_xy_I - 79 * d_omega * m_yy_I - 34 * d_omega * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[7] = (216 * m_xx_I + 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * d_omega * m_xx_I - 158 * d_omega * m_xy_I + 239 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[9] = -(72 * m_xx_I + 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * d_omega * m_xx_I + 34 * d_omega * m_xy_I + 3 * d_omega * m_yy_I - 162 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case NORTH_EAST:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7]);
                m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I + 57 * d_omega * m_xy_I * rho_I - 24 * d_omega * m_yy_I * rho_I + 6 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I - 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * d_omega * m_xx_I + 158 * d_omega * m_xy_I - 191 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[5] = (2412 * m_xy_I - 504 * m_xx_I - 504 * m_yy_I + 216 * m_zz_I + 79 * d_omega * m_xx_I + 538 * d_omega * m_xy_I + 79 * d_omega * m_yy_I + 34 * d_omega * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[7] = (216 * m_xx_I - 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * d_omega * m_xx_I + 158 * d_omega * m_xy_I + 239 * d_omega * m_yy_I - 6 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));
                moments[9] = -(72 * m_xx_I - 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * d_omega * m_xx_I - 34 * d_omega * m_xy_I + 3 * d_omega * m_yy_I - 162 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xy_I - 24 * d_omega * m_yy_I + 6 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case WEST_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I + 57 * d_omega * m_xz_I * rho_I + 6 * d_omega * m_yy_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * d_omega * m_xx_I + 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (2412 * m_xz_I - 504 * m_xx_I + 216 * m_yy_I - 504 * m_zz_I + 79 * d_omega * m_xx_I + 538 * d_omega * m_xz_I + 34 * d_omega * m_yy_I + 79 * d_omega * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = -(72 * m_xx_I - 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * d_omega * m_xx_I - 34 * d_omega * m_xz_I - 162 * d_omega * m_yy_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * d_omega * m_xx_I + 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case WEST_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                m_xz_I = inv_rho_I * (-pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I - 57 * d_omega * m_xz_I * rho_I + 6 * d_omega * m_yy_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * d_omega * m_xx_I - 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (504 * m_xx_I + 2412 * m_xz_I - 216 * m_yy_I + 504 * m_zz_I - 79 * d_omega * m_xx_I + 538 * d_omega * m_xz_I - 34 * d_omega * m_yy_I - 79 * d_omega * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = -(72 * m_xx_I + 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * d_omega * m_xx_I + 34 * d_omega * m_xz_I - 162 * d_omega * m_yy_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * d_omega * m_xx_I - 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case EAST_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                m_xz_I = inv_rho_I * (-pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I - 57 * d_omega * m_xz_I * rho_I + 6 * d_omega * m_yy_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * d_omega * m_xx_I - 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (504 * m_xx_I + 2412 * m_xz_I - 216 * m_yy_I + 504 * m_zz_I - 79 * d_omega * m_xx_I + 538 * d_omega * m_xz_I - 34 * d_omega * m_yy_I - 79 * d_omega * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = -(72 * m_xx_I + 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * d_omega * m_xx_I + 34 * d_omega * m_xz_I - 162 * d_omega * m_yy_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * d_omega * m_xx_I - 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I - 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case EAST_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * d_omega * m_xx_I * rho_I + 57 * d_omega * m_xz_I * rho_I + 6 * d_omega * m_yy_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = (936 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * d_omega * m_xx_I + 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (2412 * m_xz_I - 504 * m_xx_I + 216 * m_yy_I - 504 * m_zz_I + 79 * d_omega * m_xx_I + 538 * d_omega * m_xz_I + 34 * d_omega * m_yy_I + 79 * d_omega * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = -(72 * m_xx_I - 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * d_omega * m_xx_I - 34 * d_omega * m_xz_I - 162 * d_omega * m_yy_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (5 * m_yz_I * (43 * d_omega + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * d_omega * m_xx_I + 158 * d_omega * m_xz_I - 6 * d_omega * m_yy_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * d_omega * m_xx_I + 57 * d_omega * m_xz_I + 6 * d_omega * m_yy_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case SOUTH_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);
                m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15]);
                m_yz_I = inv_rho_I * (pop[12]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I - 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * d_omega * m_xx_I * rho_I - 24 * d_omega * m_yy_I * rho_I + 57 * d_omega * m_yz_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = -(72 * m_yy_I - 288 * m_xx_I - 216 * m_yz_I + 72 * m_zz_I - 162 * d_omega * m_xx_I + 3 * d_omega * m_yy_I - 34 * d_omega * m_yz_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = (936 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 216 * m_zz_I - 6 * d_omega * m_xx_I + 239 * d_omega * m_yy_I + 158 * d_omega * m_yz_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (216 * m_xx_I - 504 * m_yy_I + 2412 * m_yz_I - 504 * m_zz_I + 34 * d_omega * m_xx_I + 79 * d_omega * m_yy_I + 538 * d_omega * m_yz_I + 79 * d_omega * m_zz_I - 228) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 936 * m_zz_I - 6 * d_omega * m_xx_I - 191 * d_omega * m_yy_I + 158 * d_omega * m_yz_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case SOUTH_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (-pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I + 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * d_omega * m_xx_I * rho_I - 24 * d_omega * m_yy_I * rho_I - 57 * d_omega * m_yz_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = -(72 * m_yy_I - 288 * m_xx_I + 216 * m_yz_I + 72 * m_zz_I - 162 * d_omega * m_xx_I + 3 * d_omega * m_yy_I + 34 * d_omega * m_yz_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = (936 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 216 * m_zz_I - 6 * d_omega * m_xx_I + 239 * d_omega * m_yy_I - 158 * d_omega * m_yz_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (504 * m_yy_I - 216 * m_xx_I + 2412 * m_yz_I + 504 * m_zz_I - 34 * d_omega * m_xx_I - 79 * d_omega * m_yy_I + 538 * d_omega * m_yz_I - 79 * d_omega * m_zz_I + 228) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 936 * m_zz_I - 6 * d_omega * m_xx_I - 191 * d_omega * m_yy_I - 158 * d_omega * m_yz_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case NORTH_BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (-pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I + 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * d_omega * m_xx_I * rho_I - 24 * d_omega * m_yy_I * rho_I - 57 * d_omega * m_yz_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = -(72 * m_yy_I - 288 * m_xx_I + 216 * m_yz_I + 72 * m_zz_I - 162 * d_omega * m_xx_I + 3 * d_omega * m_yy_I + 34 * d_omega * m_yz_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = (936 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 216 * m_zz_I - 6 * d_omega * m_xx_I + 239 * d_omega * m_yy_I - 158 * d_omega * m_yz_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (504 * m_yy_I - 216 * m_xx_I + 2412 * m_yz_I + 504 * m_zz_I - 34 * d_omega * m_xx_I - 79 * d_omega * m_yy_I + 538 * d_omega * m_yz_I - 79 * d_omega * m_zz_I + 228) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 936 * m_zz_I - 6 * d_omega * m_xx_I - 191 * d_omega * m_yy_I - 158 * d_omega * m_yz_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I - 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case NORTH_FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);
                m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16]);
                m_yz_I = inv_rho_I * (pop[11]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);

                rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I - 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * d_omega * m_xx_I * rho_I - 24 * d_omega * m_yy_I * rho_I + 57 * d_omega * m_yz_I * rho_I - 24 * d_omega * m_zz_I * rho_I)) / (5 * (43 * d_omega + 72));

                moments[4] = -(72 * m_yy_I - 288 * m_xx_I - 216 * m_yz_I + 72 * m_zz_I - 162 * d_omega * m_xx_I + 3 * d_omega * m_yy_I - 34 * d_omega * m_yz_I + 3 * d_omega * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[5] = (5 * m_xy_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[6] = (5 * m_xz_I * (43 * d_omega + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[7] = (936 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 216 * m_zz_I - 6 * d_omega * m_xx_I + 239 * d_omega * m_yy_I + 158 * d_omega * m_yz_I - 191 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[8] = (216 * m_xx_I - 504 * m_yy_I + 2412 * m_yz_I - 504 * m_zz_I + 34 * d_omega * m_xx_I + 79 * d_omega * m_yy_I + 538 * d_omega * m_yz_I + 79 * d_omega * m_zz_I - 228) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));
                moments[9] = (216 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 936 * m_zz_I - 6 * d_omega * m_xx_I - 191 * d_omega * m_yy_I + 158 * d_omega * m_yz_I + 239 * d_omega * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * d_omega * m_xx_I - 24 * d_omega * m_yy_I + 57 * d_omega * m_yz_I - 24 * d_omega * m_zz_I + 23));

                moments[0] = rho;

                break;
            case WEST:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4)) / (d_omega + 9);

                moments[4] = (15 * m_xx_I + 2) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[5] = (2 * m_xy_I * (d_omega + 9)) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[6] = (2 * m_xz_I * (d_omega + 9)) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[7] = (4 * (d_omega + 9) * (10 * m_yy_I - m_zz_I)) / (99 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[8] = (m_yz_I * (d_omega + 9)) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[9] = -(4 * (m_yy_I - 10 * m_zz_I) * (d_omega + 9)) / (99 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));

                moments[0] = rho;

                break;
            case EAST:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4)) / (d_omega + 9);

                moments[4] = (15 * m_xx_I + 2) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[5] = (2 * m_xy_I * (d_omega + 9)) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[6] = (2 * m_xz_I * (d_omega + 9)) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[7] = (4 * (d_omega + 9) * (10 * m_yy_I - m_zz_I)) / (99 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[8] = (m_yz_I * (d_omega + 9)) / (3 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));
                moments[9] = -(4 * (m_yy_I - 10 * m_zz_I) * (d_omega + 9)) / (99 * (3 * m_xx_I - 3 * d_omega * m_xx_I + 4));

                moments[0] = rho;

                break;
            case SOUTH:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4)) / (d_omega + 9);

                moments[4] = (4 * (d_omega + 9) * (10 * m_xx_I - m_zz_I)) / (99 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[5] = (2 * m_xy_I * (d_omega + 9)) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[6] = (m_xz_I * (d_omega + 9)) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[7] = (15 * m_yy_I + 2) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[8] = (2 * m_yz_I * (d_omega + 9)) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[9] = -(4 * (m_xx_I - 10 * m_zz_I) * (d_omega + 9)) / (99 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));

                moments[0] = rho;

                break;
            case NORTH:
                moments[1] = d_u_inf;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);

                rho = (3 * rho_I * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4)) / (d_omega + 9);

                moments[4] = (4 * (d_omega + 9) * (10 * m_xx_I - m_zz_I)) / (99 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[5] = (18 * m_xy_I - 4 * d_u_inf + 2 * d_omega * m_xy_I - 3 * d_u_inf * m_yy_I + 3 * d_omega * d_u_inf * m_yy_I) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[6] = (m_xz_I * (d_omega + 9)) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[7] = (15 * m_yy_I + 2) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[8] = (2 * m_yz_I * (d_omega + 9)) / (3 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));
                moments[9] = -(4 * (m_xx_I - 10 * m_zz_I) * (d_omega + 9)) / (99 * (3 * m_yy_I - 3 * d_omega * m_yy_I + 4));

                moments[0] = rho;

                break;
            case BACK:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
                m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
                m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
                m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

                rho = (3 * rho_I * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4)) / (d_omega + 9);

                moments[4] = (4 * (d_omega + 9) * (10 * m_xx_I - m_yy_I)) / (99 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[5] = (m_xy_I * (d_omega + 9)) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[6] = (2 * m_xz_I * (d_omega + 9)) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[7] = -(4 * (m_xx_I - 10 * m_yy_I) * (d_omega + 9)) / (99 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[8] = (2 * m_yz_I * (d_omega + 9)) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[9] = (15 * m_zz_I + 2) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));

                moments[0] = rho;

                break;
            case FRONT:
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
                inv_rho_I = 1.0 / rho_I;
                m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
                m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
                m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
                m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

                rho = (3 * rho_I * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4)) / (d_omega + 9);

                moments[4] = (4 * (d_omega + 9) * (10 * m_xx_I - m_yy_I)) / (99 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[5] = (m_xy_I * (d_omega + 9)) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[6] = (2 * m_xz_I * (d_omega + 9)) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[7] = -(4 * (m_xx_I - 10 * m_yy_I) * (d_omega + 9)) / (99 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[8] = (2 * m_yz_I * (d_omega + 9)) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));
                moments[9] = (15 * m_zz_I + 2) / (3 * (3 * m_zz_I - 3 * d_omega * m_zz_I + 4));

                moments[0] = rho;

                break;
            }
        }

        __host__ [[nodiscard]] inline static nodeType_t initialCondition(
            const label_t x,
            const label_t y,
            const label_t z,
            const label_t nx,
            const label_t ny,
            const label_t nz)
        {
            if (y == 0 && x == 0 && z == 0) // SWB
            {
                return SOUTH_WEST_BACK;
            }
            else if (y == 0 && x == 0 && z == (nz - 1)) // SWF
            {
                return SOUTH_WEST_FRONT;
            }
            else if (y == 0 && x == (nx - 1) && z == 0) // SEB
            {
                return SOUTH_EAST_BACK;
            }
            else if (y == 0 && x == (nx - 1) && z == (nz - 1)) // SEF
            {
                return SOUTH_EAST_FRONT;
            }
            else if (y == (ny - 1) && x == 0 && z == 0) // NWB
            {
#ifdef NORTH_BCS_READY
                return NORTH_WEST_BACK;
#else
                return NORTH;
#endif
            }
            else if (y == (ny - 1) && x == 0 && z == (nz - 1)) // NWF
            {
#ifdef NORTH_BCS_READY
                return NORTH_WEST_FRONT;
#else
                return NORTH;
#endif
            }
            else if (y == (ny - 1) && x == (nx - 1) && z == 0) // NEB
            {
#ifdef NORTH_BCS_READY
                return NORTH_EAST_BACK;
#else
                return NORTH;
#endif
            }
            else if (y == (ny - 1) && x == (nx - 1) && z == (nz - 1)) // NEF
            {
#ifdef NORTH_BCS_READY
                return NORTH_EAST_FRONT;
#else
                return NORTH;
#endif
            }
            else if (y == 0 && x == 0) // SW
            {
                return SOUTH_WEST;
            }
            else if (y == 0 && x == (nx - 1)) // SE
            {
                return SOUTH_EAST;
            }
            else if (y == (ny - 1) && x == 0) // NW
            {
#ifdef NORTH_BCS_READY
                return NORTH_WEST;
#else
                return NORTH;
#endif
            }
            else if (y == (ny - 1) && x == (nx - 1)) // NE
            {
#ifdef NORTH_BCS_READY
                return NORTH_EAST;
#else
                return NORTH;
#endif
            }
            else if (y == 0 && z == 0) // SB
            {
                return SOUTH_BACK;
            }
            else if (y == 0 && z == (nz - 1)) // SF
            {
                return SOUTH_FRONT;
            }
            else if (y == (ny - 1) && z == 0) // NB
            {
#ifdef NORTH_BCS_READY
                return NORTH_BACK;
#else
                return NORTH;
#endif
            }
            else if (y == (ny - 1) && z == (nz - 1)) // NF
            {
#ifdef NORTH_BCS_READY
                return NORTH_FRONT;
#else
                return NORTH;
#endif
            }
            else if (x == 0 && z == 0) // WB
            {
                return WEST_BACK;
            }
            else if (x == 0 && z == (nz - 1)) // WF
            {
                return WEST_FRONT;
            }
            else if (x == (nx - 1) && z == 0) // EB
            {
                return EAST_BACK;
            }
            else if (x == (nx - 1) && z == (nz - 1)) // EF
            {
                return EAST_FRONT;
            }
            else if (y == 0) // S
            {
                return SOUTH;
            }
            else if (y == (ny - 1)) // N
            {
                return NORTH;
            }
            else if (x == 0) // W
            {
                return WEST;
            }
            else if (x == (nx - 1)) // E
            {
                return EAST;
            }
            else if (z == 0) // B
            {
                return BACK;
            }
            else if (z == (nz - 1)) // F
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