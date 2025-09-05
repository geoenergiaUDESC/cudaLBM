/**
Filename: boundaryConditions.cuh
Contents: A class applying boundary conditions to the lid driven cavity case
**/

#ifndef __MBLBM_BOUNDARYCONDITIONS_CUH
#define __MBLBM_BOUNDARYCONDITIONS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

#include "normalVector.cuh"
#include "boundaryField.cuh"

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
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_WEST_FRONT():
            {
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST_BACK():
            {
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST_FRONT():
            {
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_WEST():
            {
                const scalar_t mxy_I = pop[8] * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = mxy;                      // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST():
            {
                const scalar_t mxy_I = -pop[13] * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = mxy;                      // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST_BACK():
            {
                const scalar_t mxz_I = pop[10] * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST_FRONT():
            {
                const scalar_t mxz_I = -pop[16] * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST_BACK():
            {
                const scalar_t mxz_I = -pop[15] * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST_FRONT():
            {
                const scalar_t mxz_I = pop[9] * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_BACK():
            {
                const scalar_t myz_I = pop[12] * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = myz;                      // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_FRONT():
            {
                const scalar_t myz_I = -pop[18] * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = myz;                      // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST():
            {
                const scalar_t mxy_I = (pop[8] - pop[14]) * inv_rho_I;
                const scalar_t mxz_I = (pop[10] - pop[16]) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = mxy;                      // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST():
            {
                const scalar_t mxy_I = (pop[7] - pop[13]) * inv_rho_I;
                const scalar_t mxz_I = (pop[9] - pop[15]) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = mxy;                      // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = static_cast<scalar_t>(0); // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH():
            {
                const scalar_t mxy_I = (pop[8] - pop[13]) * inv_rho_I;
                const scalar_t myz_I = (pop[12] - pop[18]) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = mxy;                      // mxy
                moments[6] = static_cast<scalar_t>(0); // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = myz;                      // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::BACK():
            {
                const scalar_t mxz_I = (pop[10] - pop[15]) * inv_rho_I;
                const scalar_t myz_I = (pop[12] - pop[17]) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = myz;                      // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::FRONT():
            {
                const scalar_t mxz_I = (pop[9] - pop[16]) * inv_rho_I;
                const scalar_t myz_I = (pop[11] - pop[18]) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0); // ux
                moments[2] = static_cast<scalar_t>(0); // uy
                moments[3] = static_cast<scalar_t>(0); // uz
                moments[4] = static_cast<scalar_t>(0); // mxx
                moments[5] = static_cast<scalar_t>(0); // mxy
                moments[6] = mxz;                      // mxz
                moments[7] = static_cast<scalar_t>(0); // myy
                moments[8] = myz;                      // myz
                moments[9] = static_cast<scalar_t>(0); // mzz

                return;
            }

            // Lid boundaries
            case normalVector::NORTH():
            {
                const scalar_t mxy_I = (pop[7] - pop[14]) * inv_rho_I;
                const scalar_t myz_I = (pop[11] - pop[17]) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = (static_cast<scalar_t>(6) * mxy_I * rho_I - device::u_inf * rho) / (static_cast<scalar_t>(3) * rho);
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = mxy;                           // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = myz;                           // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = static_cast<scalar_t>(0);      // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = static_cast<scalar_t>(0);      // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = static_cast<scalar_t>(0);      // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = static_cast<scalar_t>(0);      // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = static_cast<scalar_t>(0);      // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = static_cast<scalar_t>(0);      // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = static_cast<scalar_t>(0);      // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = static_cast<scalar_t>(0);      // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_BACK():
            {
                const scalar_t myz_I = -pop[17] * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(72) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I + static_cast<scalar_t>(2) * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = static_cast<scalar_t>(0);      // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = myz;                           // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_FRONT():
            {
                const scalar_t myz_I = pop[11] * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(72) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I - static_cast<scalar_t>(2) * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = static_cast<scalar_t>(0);      // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = myz;                           // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST():
            {
                const scalar_t mxy_I = pop[7] * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega + static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho - static_cast<scalar_t>(3) * device::u_inf * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = mxy;                           // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = static_cast<scalar_t>(0);      // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST():
            {
                const scalar_t mxy_I = -pop[14] * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega - static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho - static_cast<scalar_t>(3) * device::u_inf * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments[0] = rho;
                moments[1] = device::u_inf;                 // ux
                moments[2] = static_cast<scalar_t>(0);      // uy
                moments[3] = static_cast<scalar_t>(0);      // uz
                moments[4] = device::u_inf * device::u_inf; // mxx
                moments[5] = mxy;                           // mxy
                moments[6] = static_cast<scalar_t>(0);      // mxz
                moments[7] = static_cast<scalar_t>(0);      // myy
                moments[8] = static_cast<scalar_t>(0);      // myz
                moments[9] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            }
        }

    private:
    };
}

#endif