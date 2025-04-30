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
        __device__ static inline void stream(
            scalar_t (&moments)[10],
            const scalar_t (&pop)[19],
            const nodeType::type nodeType,
            const scalar_t Re,
            const scalar_t u_inf,
            const label_t nx) noexcept
        {
            switch (nodeType)
            {
            case nodeType::SOUTHWESTBACK:
            {
                // printf("Doing SOUTHWESTBACK\n");
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];

                const scalar_t rho = (12.0 * rho_I) / 7.0;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTHWESTFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];

                const scalar_t rho = (12 * rho_I) / 7;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHWESTBACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];

                const scalar_t rho = (12 * rho_I) / 7;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHWESTFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];

                const scalar_t rho = (12 * rho_I) / 7;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTHEASTBACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];

                const scalar_t rho = (12 * rho_I) / 7;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTHEASTFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];

                const scalar_t rho = (12 * rho_I) / 7;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHEASTBACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];

                const scalar_t rho = (12 * rho_I) / 7;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHEASTFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];

                const scalar_t rho = (12 * rho_I) / 7;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTHWEST:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[8]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (OMEGA * m_xy_I - m_xy_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = (25 * m_xy_I - 1) / (9 * (OMEGA * m_xy_I - m_xy_I + 1));
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHWEST:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (-pop[14]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (m_xy_I - OMEGA * m_xy_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = (25 * m_xy_I + 1) / (9 * (m_xy_I - OMEGA * m_xy_I + 1));
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTHEAST:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (-pop[13]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (m_xy_I - OMEGA * m_xy_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = (25 * m_xy_I + 1) / (9 * (m_xy_I - OMEGA * m_xy_I + 1));
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHEAST:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (OMEGA * m_xy_I - m_xy_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = (25 * m_xy_I - 1) / (9 * (OMEGA * m_xy_I - m_xy_I + 1));
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::WESTBACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[10]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (OMEGA * m_xz_I - m_xz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = (25 * m_xz_I - 1) / (9 * (OMEGA * m_xz_I - m_xz_I + 1));
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::WESTFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (-pop[16]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (m_xz_I - OMEGA * m_xz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = (25 * m_xz_I + 1) / (9 * (m_xz_I - OMEGA * m_xz_I + 1));
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::EASTBACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (-pop[15]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (m_xz_I - OMEGA * m_xz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = (25 * m_xz_I + 1) / (9 * (m_xz_I - OMEGA * m_xz_I + 1));
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::EASTFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[9]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (OMEGA * m_xz_I - m_xz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = (25 * m_xz_I - 1) / (9 * (OMEGA * m_xz_I - m_xz_I + 1));
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTHBACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (pop[12]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (OMEGA * m_yz_I - m_yz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = (25 * m_yz_I - 1) / (9 * (OMEGA * m_yz_I - m_yz_I + 1));
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTHFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (-pop[18]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (m_yz_I - OMEGA * m_yz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = (25 * m_yz_I + 1) / (9 * (m_yz_I - OMEGA * m_yz_I + 1));
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHBACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (-pop[17]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (m_yz_I - OMEGA * m_yz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = (25 * m_yz_I + 1) / (9 * (m_yz_I - OMEGA * m_yz_I + 1));
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTHFRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (pop[11]);

                const scalar_t OMEGA = omega(Re, u_inf, nx);
                const scalar_t rho = (36 * rho_I * (OMEGA * m_yz_I - m_yz_I + 1)) / (OMEGA + 24);

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = (25 * m_yz_I - 1) / (9 * (OMEGA * m_yz_I - m_yz_I + 1));
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::WEST:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[16]);

                const scalar_t rho = (6 * rho_I) / 5;

                moments[4] = 0;
                moments[5] = (5 * m_xy_I) / 3;
                moments[6] = (5 * m_xz_I) / 3;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::EAST:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[15]);

                const scalar_t rho = (6 * rho_I) / 5;

                moments[4] = 0;
                moments[5] = (5 * m_xy_I) / 3;
                moments[6] = (5 * m_xz_I) / 3;
                moments[7] = 0;
                moments[8] = 0;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::SOUTH:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[18]);

                const scalar_t rho = (6 * rho_I) / 5;

                moments[4] = 0;
                moments[5] = (5 * m_xy_I) / 3;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = (5 * m_yz_I) / 3;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::NORTH:
            {
                moments[1] = u_inf;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[17]);

                const scalar_t rho = (6 * rho_I) / 5;

                moments[4] = (6 * u_inf * u_inf * rho_I) / 5;
                moments[5] = (5 * m_xy_I) / 3 - u_inf / 3;
                moments[6] = 0;
                moments[7] = 0;
                moments[8] = (5 * m_yz_I) / 3;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::BACK:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[17]);

                const scalar_t rho = (6 * rho_I) / 5;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = (5 * m_xz_I) / 3;
                moments[7] = 0;
                moments[8] = (5 * m_yz_I) / 3;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            case nodeType::FRONT:
            {
                moments[1] = 0;
                moments[2] = 0;
                moments[3] = 0;

                const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
                const scalar_t inv_rho_I = 1.0 / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[18]);

                const scalar_t rho = (6 * rho_I) / 5;

                moments[4] = 0;
                moments[5] = 0;
                moments[6] = (5 * m_xz_I) / 3;
                moments[7] = 0;
                moments[8] = (5 * m_yz_I) / 3;
                moments[9] = 0;

                moments[0] = rho;

                break;
            }
            }
        }

    private:
    };
}

#endif