/**
Filename: D3Q27BoundaryConditions.cuh
Contents: A class applying boundary conditions to the lid driven cavity case
**/

#ifndef __MBLBM_D3Q27BOUNDARYCONDITIONS_CUH
#define __MBLBM_D3Q27BOUNDARYCONDITIONS_CUH

namespace LBM
{
    __device__ static inline void apply_boundaries_D3Q27(
        const scalar_t (&ptrRestrict pop)[27],
        scalar_t (&ptrRestrict moments)[10],
        const normalVector &b_n) noexcept
    {

        switch (b_n.nodeType())
        {
        case normalVector::SOUTH_WEST_BACK():
        {
            const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[20];

            const scalar_t rho = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);

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
            const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18] + pop[22];

            const scalar_t rho = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);

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
            const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15] + pop[26];

            const scalar_t rho = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);

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
            const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18] + pop[23];

            const scalar_t rho = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);

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
            const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18] + pop[20] + pop[22];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = (pop[8] + pop[20] + pop[22]) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho); // mxy
            moments[6] = static_cast<scalar_t>(0);                                                             // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = static_cast<scalar_t>(0);                                                             // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::SOUTH_EAST():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18] + pop[23] + pop[26];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = -(pop[13] + pop[23] + pop[26]) * inv_rho_I;

            const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device_omega) / (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho); // mxy
            moments[6] = static_cast<scalar_t>(0);                                                             // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = static_cast<scalar_t>(0);                                                             // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::WEST_BACK():
        {
            const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17] + pop[20] + pop[24];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxz_I = (pop[10] + pop[20] + pop[24]) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho); // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = static_cast<scalar_t>(0);                                                             // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::WEST_FRONT():
        {
            const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18] + pop[22] + pop[25];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxz_I = -(pop[16] + pop[22] + pop[25]) * inv_rho_I;

            const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho); // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = static_cast<scalar_t>(0);                                                             // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::EAST_BACK():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17] + pop[21] + pop[26];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxz_I = -(pop[15] + pop[21] + pop[26]) * inv_rho_I;

            const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho); // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = static_cast<scalar_t>(0);                                                             // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::EAST_FRONT():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18] + pop[19] + pop[23];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxz_I = (pop[9] + pop[19] + pop[23]) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho); // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = static_cast<scalar_t>(0);                                                             // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::SOUTH_BACK():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[20] + pop[26];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t myz_I = (pop[12] + pop[20] + pop[26]) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = static_cast<scalar_t>(0);                                                             // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho); // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::SOUTH_FRONT():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18] + pop[22] + pop[23];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t myz_I = -(pop[18] + pop[22] + pop[23]) * inv_rho_I;

            const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                                                             // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = static_cast<scalar_t>(0);                                                             // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = static_cast<scalar_t>(0);                                                             // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho); // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::WEST():
        {
            const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18] + pop[20] + pop[22] + pop[24] + pop[25];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
            const scalar_t mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                       // ux
            moments[2] = static_cast<scalar_t>(0);                       // uy
            moments[3] = static_cast<scalar_t>(0);                       // uz
            moments[4] = static_cast<scalar_t>(0);                       // mxx
            moments[5] = static_cast<scalar_t>(2) * mxy_I * rho_I / rho; // mxy
            moments[6] = static_cast<scalar_t>(2) * mxz_I * rho_I / rho; // mxz
            moments[7] = static_cast<scalar_t>(0);                       // myy
            moments[8] = static_cast<scalar_t>(0);                       // myz
            moments[9] = static_cast<scalar_t>(0);                       // mzz

            return;
        }
        case normalVector::EAST():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
            const scalar_t mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                       // ux
            moments[2] = static_cast<scalar_t>(0);                       // uy
            moments[3] = static_cast<scalar_t>(0);                       // uz
            moments[4] = static_cast<scalar_t>(0);                       // mxx
            moments[5] = static_cast<scalar_t>(2) * mxy_I * rho_I / rho; // mxy
            moments[6] = static_cast<scalar_t>(2) * mxz_I * rho_I / rho; // mxz
            moments[7] = static_cast<scalar_t>(0);                       // myy
            moments[8] = static_cast<scalar_t>(0);                       // myz
            moments[9] = static_cast<scalar_t>(0);                       // mzz

            return;
        }
        case normalVector::SOUTH():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
            const scalar_t myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                       // ux
            moments[2] = static_cast<scalar_t>(0);                       // uy
            moments[3] = static_cast<scalar_t>(0);                       // uz
            moments[4] = static_cast<scalar_t>(0);                       // mxx
            moments[5] = static_cast<scalar_t>(2) * mxy_I * rho_I / rho; // mxy
            moments[6] = static_cast<scalar_t>(0);                       // mxz
            moments[7] = static_cast<scalar_t>(0);                       // myy
            moments[8] = static_cast<scalar_t>(2) * myz_I * rho_I / rho; // myz
            moments[9] = static_cast<scalar_t>(0);                       // mzz

            return;
        }
        case normalVector::BACK():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
            const scalar_t myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                       // ux
            moments[2] = static_cast<scalar_t>(0);                       // uy
            moments[3] = static_cast<scalar_t>(0);                       // uz
            moments[4] = static_cast<scalar_t>(0);                       // mxx
            moments[5] = static_cast<scalar_t>(0);                       // mxy
            moments[6] = static_cast<scalar_t>(2) * mxz_I * rho_I / rho; // mxz
            moments[7] = static_cast<scalar_t>(0);                       // myy
            moments[8] = static_cast<scalar_t>(2) * myz_I * rho_I / rho; // myz
            moments[9] = static_cast<scalar_t>(0);                       // mzz

            return;
        }
        case normalVector::FRONT():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;
            const scalar_t myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);

            moments[0] = rho;
            moments[1] = static_cast<scalar_t>(0);                       // ux
            moments[2] = static_cast<scalar_t>(0);                       // uy
            moments[3] = static_cast<scalar_t>(0);                       // uz
            moments[4] = static_cast<scalar_t>(0);                       // mxx
            moments[5] = static_cast<scalar_t>(0);                       // mxy
            moments[6] = static_cast<scalar_t>(2) * mxz_I * rho_I / rho; // mxz
            moments[7] = static_cast<scalar_t>(0);                       // myy
            moments[8] = static_cast<scalar_t>(2) * myz_I * rho_I / rho; // myz
            moments[9] = static_cast<scalar_t>(0);                       // mzz

            return;
        }

        // Lid boundaries
        case normalVector::NORTH():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
            const scalar_t myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);

            moments[0] = rho;
            moments[1] = device_u_inf;                                                                                       // ux
            moments[2] = static_cast<scalar_t>(0);                                                                           // uy
            moments[3] = static_cast<scalar_t>(0);                                                                           // uz
            moments[4] = device_u_inf * device_u_inf;                                                                        // mxx
            moments[5] = (static_cast<scalar_t>(6) * mxy_I * rho_I - device_u_inf * rho) / (static_cast<scalar_t>(3) * rho); // mxy
            moments[6] = static_cast<scalar_t>(0);                                                                           // mxz
            moments[7] = static_cast<scalar_t>(0);                                                                           // myy
            moments[8] = static_cast<scalar_t>(2) * myz_I * rho_I / rho;                                                     // myz
            moments[9] = static_cast<scalar_t>(0);                                                                           // mzz

            return;
        }
        case normalVector::NORTH_WEST_BACK():
        {
            const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17] + pop[24];

            const scalar_t rho = -static_cast<scalar_t>(216) * rho_I /
                                 (-static_cast<scalar_t>(125) - static_cast<scalar_t>(75) * device_u_inf + static_cast<scalar_t>(75) * device_u_inf * device_u_inf);

            moments[0] = rho;
            moments[1] = device_u_inf;                // ux
            moments[2] = static_cast<scalar_t>(0);    // uy
            moments[3] = static_cast<scalar_t>(0);    // uz
            moments[4] = device_u_inf * device_u_inf; // mxx
            moments[5] = static_cast<scalar_t>(0);    // mxy
            moments[6] = static_cast<scalar_t>(0);    // mxz
            moments[7] = static_cast<scalar_t>(0);    // myy
            moments[8] = static_cast<scalar_t>(0);    // myz
            moments[9] = static_cast<scalar_t>(0);    // mzz

            return;
        }
        case normalVector::NORTH_WEST_FRONT():
        {
            const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16] + pop[25];

            const scalar_t rho = -static_cast<scalar_t>(216) * rho_I /
                                 (-static_cast<scalar_t>(125) - static_cast<scalar_t>(75) * device_u_inf + static_cast<scalar_t>(75) * device_u_inf * device_u_inf);

            moments[0] = rho;
            moments[1] = device_u_inf;                // ux
            moments[2] = static_cast<scalar_t>(0);    // uy
            moments[3] = static_cast<scalar_t>(0);    // uz
            moments[4] = device_u_inf * device_u_inf; // mxx
            moments[5] = static_cast<scalar_t>(0);    // mxy
            moments[6] = static_cast<scalar_t>(0);    // mxz
            moments[7] = static_cast<scalar_t>(0);    // myy
            moments[8] = static_cast<scalar_t>(0);    // myz
            moments[9] = static_cast<scalar_t>(0);    // mzz

            return;
        }
        case normalVector::NORTH_EAST_BACK():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17] + pop[21];

            const scalar_t rho = -static_cast<scalar_t>(216) * rho_I /
                                 (-static_cast<scalar_t>(125) + static_cast<scalar_t>(75) * device_u_inf + static_cast<scalar_t>(75) * device_u_inf * device_u_inf);

            moments[0] = rho;
            moments[1] = device_u_inf;                // ux
            moments[2] = static_cast<scalar_t>(0);    // uy
            moments[3] = static_cast<scalar_t>(0);    // uz
            moments[4] = device_u_inf * device_u_inf; // mxx
            moments[5] = static_cast<scalar_t>(0);    // mxy
            moments[6] = static_cast<scalar_t>(0);    // mxz
            moments[7] = static_cast<scalar_t>(0);    // myy
            moments[8] = static_cast<scalar_t>(0);    // myz
            moments[9] = static_cast<scalar_t>(0);    // mzz

            return;
        }
        case normalVector::NORTH_EAST_FRONT():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[19];

            const scalar_t rho = -static_cast<scalar_t>(216) * rho_I /
                                 (-static_cast<scalar_t>(125) + static_cast<scalar_t>(75) * device_u_inf + static_cast<scalar_t>(75) * device_u_inf * device_u_inf);

            moments[0] = rho;
            moments[1] = device_u_inf;                // ux
            moments[2] = static_cast<scalar_t>(0);    // uy
            moments[3] = static_cast<scalar_t>(0);    // uz
            moments[4] = device_u_inf * device_u_inf; // mxx
            moments[5] = static_cast<scalar_t>(0);    // mxy
            moments[6] = static_cast<scalar_t>(0);    // mxz
            moments[7] = static_cast<scalar_t>(0);    // myy
            moments[8] = static_cast<scalar_t>(0);    // myz
            moments[9] = static_cast<scalar_t>(0);    // mzz

            return;
        }
        case normalVector::NORTH_BACK():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17] + pop[21] + pop[24];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t myz_I = -(pop[17] + pop[21] + pop[24]) * inv_rho_I;

            const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = device_u_inf;                                                                         // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = device_u_inf * device_u_inf;                                                          // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = static_cast<scalar_t>(0);                                                             // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho); // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::NORTH_FRONT():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16] + pop[19] + pop[25];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t myz_I = (pop[11] + pop[19] + pop[25]) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + device_omega);

            moments[0] = rho;
            moments[1] = device_u_inf;                                                                         // ux
            moments[2] = static_cast<scalar_t>(0);                                                             // uy
            moments[3] = static_cast<scalar_t>(0);                                                             // uz
            moments[4] = device_u_inf * device_u_inf;                                                          // mxx
            moments[5] = static_cast<scalar_t>(0);                                                             // mxy
            moments[6] = static_cast<scalar_t>(0);                                                             // mxz
            moments[7] = static_cast<scalar_t>(0);                                                             // myy
            moments[8] = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho); // myz
            moments[9] = static_cast<scalar_t>(0);                                                             // mzz

            return;
        }
        case normalVector::NORTH_EAST():
        {
            const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17] + pop[19] + pop[21];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = (pop[7] + pop[19] + pop[21]) * inv_rho_I;

            const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device_u_inf - static_cast<scalar_t>(18) * device_u_inf * device_u_inf + device_omega + static_cast<scalar_t>(3) * device_u_inf * device_omega + static_cast<scalar_t>(3) * device_u_inf * device_u_inf * device_omega);

            moments[0] = rho;
            moments[1] = device_u_inf;                // ux
            moments[2] = static_cast<scalar_t>(0);    // uy
            moments[3] = static_cast<scalar_t>(0);    // uz
            moments[4] = device_u_inf * device_u_inf; // mxx
            moments[5] = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho - static_cast<scalar_t>(3) * device_u_inf * rho - static_cast<scalar_t>(3) * device_u_inf * device_u_inf * rho) /
                         (static_cast<scalar_t>(9) * rho); // mxy
            moments[6] = static_cast<scalar_t>(0);         // mxz
            moments[7] = static_cast<scalar_t>(0);         // myy
            moments[8] = static_cast<scalar_t>(0);         // myz
            moments[9] = static_cast<scalar_t>(0);         // mzz

            return;
        }
        case normalVector::NORTH_WEST():
        {
            const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[24] + pop[25];
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            const scalar_t mxy_I = -(pop[14] + pop[24] + pop[25]) * inv_rho_I;

            const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device_omega) /
                                 (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device_u_inf - static_cast<scalar_t>(18) * device_u_inf * device_u_inf + device_omega - static_cast<scalar_t>(3) * device_u_inf * device_omega + static_cast<scalar_t>(3) * device_u_inf * device_u_inf * device_omega);

            moments[0] = rho;
            moments[1] = device_u_inf;                // ux
            moments[2] = static_cast<scalar_t>(0);    // uy
            moments[3] = static_cast<scalar_t>(0);    // uz
            moments[4] = device_u_inf * device_u_inf; // mxx
            moments[5] = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho - static_cast<scalar_t>(3) * device_u_inf * rho + static_cast<scalar_t>(3) * device_u_inf * device_u_inf * rho) /
                         (static_cast<scalar_t>(9) * rho); // mxy
            moments[6] = static_cast<scalar_t>(0);         // mxz
            moments[7] = static_cast<scalar_t>(0);         // myy
            moments[8] = static_cast<scalar_t>(0);         // myz
            moments[9] = static_cast<scalar_t>(0);         // mzz

            return;
        }
        }
    }
}

#endif