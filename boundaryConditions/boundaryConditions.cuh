/**
Filename: boundaryConditions.cuh
Contents: A class applying boundary conditions to the lid driven cavity case
**/

#ifndef __MBLBM_BOUNDARYCONDITIONS_CUH
#define __MBLBM_BOUNDARYCONDITIONS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

#include "normalVector.cuh"

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
        __device__ static inline constexpr void calculateMoments(const scalar_t (&ptrRestrict pop)[VSet::Q()], scalar_t (&ptrRestrict moments)[10], const normalVector &b_n) noexcept
        {
            const scalar_t rho_I = VSet::rho_I(pop, b_n);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            switch (b_n.nodeType())
            {
            // Static boundaries
            case normalVector::SOUTH_WEST_BACK():
            {
                if constexpr (VSet::Q() == 19)
                {
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }
                else if constexpr (VSet::Q() == 27)
                {
                    // Modify this and leave it hard-coded for now
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }

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
            case normalVector::SOUTH_WEST_FRONT():
            {
                if constexpr (VSet::Q() == 19)
                {
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }
                else if constexpr (VSet::Q() == 27)
                {
                    // Modify this and leave it hard-coded for now
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }

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
            case normalVector::SOUTH_EAST_BACK():
            {
                if constexpr (VSet::Q() == 19)
                {
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }
                else if constexpr (VSet::Q() == 27)
                {
                    // Modify this and leave it hard-coded for now
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }

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
            case normalVector::SOUTH_EAST_FRONT():
            {
                if constexpr (VSet::Q() == 19)
                {
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }
                else if constexpr (VSet::Q() == 27)
                {
                    // Modify this and leave it hard-coded for now
                    const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                    moments[0] = rho;
                }
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
            case normalVector::SOUTH_WEST():
            {
                const scalar_t m_xy_I = inv_rho_I * pop[8];
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (device::omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
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
            case normalVector::SOUTH_EAST():
            {
                const scalar_t m_xy_I = inv_rho_I * (-pop[13]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xy_I - device::omega * m_xy_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
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
            case normalVector::WEST_BACK():
            {
                const scalar_t m_xz_I = inv_rho_I * (pop[10]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (device::omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (device::omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::WEST_FRONT():
            {
                const scalar_t m_xz_I = inv_rho_I * (-pop[16]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xz_I - device::omega * m_xz_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xz_I - device::omega * m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::EAST_BACK():
            {
                const scalar_t m_xz_I = inv_rho_I * (-pop[15]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xz_I - device::omega * m_xz_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xz_I - device::omega * m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::EAST_FRONT():
            {
                const scalar_t m_xz_I = inv_rho_I * (pop[9]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (device::omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (device::omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_BACK():
            {
                const scalar_t m_yz_I = inv_rho_I * (pop[12]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (device::omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
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
            case normalVector::SOUTH_FRONT():
            {
                const scalar_t m_yz_I = inv_rho_I * (-pop[18]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_yz_I - device::omega * m_yz_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24));
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
            case normalVector::WEST():
            {
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
            case normalVector::EAST():
            {
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
            case normalVector::SOUTH():
            {
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
            case normalVector::BACK():
            {
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
            case normalVector::FRONT():
            {
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
            case normalVector::NORTH():
            {
                const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = device::u_inf;
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = (static_cast<scalar_t>(6) * device::u_inf * device::u_inf * rho_I) / static_cast<scalar_t>(5);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3) - device::u_inf / static_cast<scalar_t>(3);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_WEST_BACK():
            {
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf - static_cast<scalar_t>(9) * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_WEST_FRONT():
            {
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf - static_cast<scalar_t>(9) * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_EAST_BACK():
            {
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf - static_cast<scalar_t>(9) * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_EAST_FRONT():
            {
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf - static_cast<scalar_t>(9) * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_BACK():
            {
                const scalar_t m_yz_I = inv_rho_I * (-pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(72) * rho_I * (m_yz_I - device::omega * m_yz_I + static_cast<scalar_t>(1))) / (static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(48) - static_cast<scalar_t>(3) * device::omega * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);                                       // At the intersection of front and back, only z derivative exists
                moments[5] = -VelocitySet::velocitySet::cs2() * device::tau * device::u_inf; // At the intersection of front and back, only z derivative exists
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_FRONT():
            {
                const scalar_t m_yz_I = inv_rho_I * (pop[11]);
                const scalar_t rho = (static_cast<scalar_t>(72) * rho_I * (device::omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1))) / (static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(48) - static_cast<scalar_t>(3) * device::omega * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);                                      // At the intersection of front and back, only z derivative exists
                moments[5] = VelocitySet::velocitySet::cs2() * device::tau * device::u_inf; // At the intersection of front and back, only z derivative exists
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_EAST():
            {
                const scalar_t m_xy_I = inv_rho_I * (pop[7]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (device::omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + static_cast<scalar_t>(3) * device::omega * device::u_inf + static_cast<scalar_t>(3) * device::omega * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[5] = -static_cast<scalar_t>(2) * VelocitySet::velocitySet::cs2() * device::tau * device::u_inf; // At the intersection of East and West, only x derivative exists
                moments[4] = static_cast<scalar_t>(0);                                                                  // At the intersection of East and West, only x derivative exists
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_WEST():
            {
                const scalar_t m_xy_I = inv_rho_I * (pop[7]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (device::omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (device::omega + static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + static_cast<scalar_t>(3) * device::omega * device::u_inf + static_cast<scalar_t>(3) * device::omega * device::u_inf * device::u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(2) * VelocitySet::velocitySet::cs2() * device::tau * device::u_inf; // At the intersection of East and West, only x derivative exists
                moments[4] = static_cast<scalar_t>(0);                                                                 // At the intersection of East and West, only x derivative exists
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            }
        }

    private:
    };
}

#endif