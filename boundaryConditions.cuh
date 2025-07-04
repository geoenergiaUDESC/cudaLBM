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
    template <typename T>
    struct normalVector
    {
    public:
        __device__ [[nodiscard]] inline normalVector() noexcept
            : x(x_normal()),
              y(y_normal()),
              z(z_normal()){};

        const T x;
        const T y;
        const T z;

    private:
        __device__ [[nodiscard]] static inline T x_normal() noexcept
        {
            const label_t x_index = threadIdx.x + blockDim.x * blockIdx.x;
            return static_cast<T>((x_index == d_nx - 1) - (x_index == 0));
        }

        __device__ [[nodiscard]] static inline T y_normal() noexcept
        {
            const label_t y_index = threadIdx.y + blockDim.y * blockIdx.y;
            return static_cast<T>((y_index == d_ny - 1) - (y_index == 0));
        }

        __device__ [[nodiscard]] static inline T z_normal() noexcept
        {
            const label_t z_index = threadIdx.z + blockDim.z * blockIdx.z;
            return static_cast<T>((z_index == d_nz - 1) - (z_index == 0));
        }
    };

    class boundaryConditions
    {
    public:
        [[nodiscard]] inline consteval boundaryConditions() {};

        template <class VSet, const label_t q_>
        __device__ [[nodiscard]] static inline scalar_t rho_coefficient(const lattice_constant<q_> q) noexcept
        {
            const normalVector<int16_t> b_n;

            const bool cond_x = (b_n.x > 0 & VSet::nxNeg(q)) | (b_n.x < 0 & VSet::nxPos(q));
            const bool cond_y = (b_n.y > 0 & VSet::nyNeg(q)) | (b_n.y < 0 & VSet::nyPos(q));
            const bool cond_z = (b_n.z > 0 & VSet::nzNeg(q)) | (b_n.z < 0 & VSet::nzPos(q));

            return static_cast<scalar_t>(cond_x | cond_y | cond_z);
        }

        template <class VSet>
        __device__ static inline void calculateMoments(
            const scalar_t pop[VSet::Q()],
            scalar_t (&ptrRestrict moments)[10],
            const nodeType_t nodeType) noexcept
        {

            const scalar_t rho_I =
                ((rho_coefficient<VSet>(lattice_constant<0>()) * pop[0]) +
                 (rho_coefficient<VSet>(lattice_constant<1>()) * pop[1]) +
                 (rho_coefficient<VSet>(lattice_constant<2>()) * pop[2]) +
                 (rho_coefficient<VSet>(lattice_constant<3>()) * pop[3]) +
                 (rho_coefficient<VSet>(lattice_constant<4>()) * pop[4]) +
                 (rho_coefficient<VSet>(lattice_constant<5>()) * pop[5]) +
                 (rho_coefficient<VSet>(lattice_constant<6>()) * pop[6]) +
                 (rho_coefficient<VSet>(lattice_constant<7>()) * pop[7]) +
                 (rho_coefficient<VSet>(lattice_constant<8>()) * pop[8]) +
                 (rho_coefficient<VSet>(lattice_constant<9>()) * pop[9]) +
                 (rho_coefficient<VSet>(lattice_constant<10>()) * pop[10]) +
                 (rho_coefficient<VSet>(lattice_constant<11>()) * pop[11]) +
                 (rho_coefficient<VSet>(lattice_constant<12>()) * pop[12]) +
                 (rho_coefficient<VSet>(lattice_constant<13>()) * pop[13]) +
                 (rho_coefficient<VSet>(lattice_constant<14>()) * pop[14]) +
                 (rho_coefficient<VSet>(lattice_constant<15>()) * pop[15]) +
                 (rho_coefficient<VSet>(lattice_constant<16>()) * pop[16]) +
                 (rho_coefficient<VSet>(lattice_constant<17>()) * pop[17]) +
                 (rho_coefficient<VSet>(lattice_constant<18>()) * pop[18]));
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            switch (nodeType)
            {
            // Static boundaries
            case SOUTH_WEST_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH_WEST_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH_EAST_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH_EAST_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH_WEST:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * pop[8];
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                // moments[5] = (static_cast<scalar_t>(25) * m_xy_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1)));
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH_EAST:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (-pop[13]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xy_I - d_omega * m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                // moments[5] = (static_cast<scalar_t>(25) * m_xy_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xy_I - d_omega * m_xy_I + static_cast<scalar_t>(1)));
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case WEST_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[10]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case WEST_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (-pop[16]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case EAST_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (-pop[15]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case EAST_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[9]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (pop[12]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (static_cast<scalar_t>(25) * m_yz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (-pop[18]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (static_cast<scalar_t>(25) * m_yz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case WEST:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case EAST:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case SOUTH:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }

            // Lid boundaries
            case NORTH:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = d_u_inf;
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = (static_cast<scalar_t>(6) * d_u_inf * d_u_inf * rho_I) / static_cast<scalar_t>(5);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3) - d_u_inf / static_cast<scalar_t>(3);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_WEST_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_WEST_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_EAST_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_EAST_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_BACK:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (-pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(72) * rho_I * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1))) / (static_cast<scalar_t>(2) * d_omega + static_cast<scalar_t>(48) - static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0);                           // At the intersection of front and back, only z derivative exists
                moments[5] = -VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of front and back, only z derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_FRONT:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (pop[11]);
                const scalar_t rho = (static_cast<scalar_t>(72) * rho_I * (d_omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1))) / (static_cast<scalar_t>(2) * d_omega + static_cast<scalar_t>(48) - static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0);                          // At the intersection of front and back, only z derivative exists
                moments[5] = VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of front and back, only z derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_EAST:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * d_u_inf - static_cast<scalar_t>(18) * d_u_inf * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[5] = -static_cast<scalar_t>(2) * VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of East and West, only x derivative exists
                moments[4] = static_cast<scalar_t>(0);                                                      // At the intersection of East and West, only x derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case NORTH_WEST:
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * d_u_inf - static_cast<scalar_t>(18) * d_u_inf * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[5] = static_cast<scalar_t>(2) * VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of East and West, only x derivative exists
                moments[4] = static_cast<scalar_t>(0);                                                     // At the intersection of East and West, only x derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
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
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_WEST_BACK;
                }
                else
                {
                    return NORTH;
                }
            }
            else if (y == (ny - 1) && x == 0 && z == (nz - 1)) // NWF
            {
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_WEST_FRONT;
                }
                else
                {
                    return NORTH;
                }
            }
            else if (y == (ny - 1) && x == (nx - 1) && z == 0) // NEB
            {
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_EAST_BACK;
                }
                else
                {
                    return NORTH;
                }
            }
            else if (y == (ny - 1) && x == (nx - 1) && z == (nz - 1)) // NEF
            {
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_EAST_FRONT;
                }
                else
                {
                    return NORTH;
                }
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
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_WEST;
                }
                else
                {
                    return NORTH;
                }
            }
            else if (y == (ny - 1) && x == (nx - 1)) // NE
            {
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_EAST;
                }
                else
                {
                    return NORTH;
                }
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
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_BACK;
                }
                else
                {
                    return NORTH;
                }
            }
            else if (y == (ny - 1) && z == (nz - 1)) // NF
            {
                if constexpr (NORTH_BCS_READY)
                {
                    return NORTH_FRONT;
                }
                else
                {
                    return NORTH;
                }
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