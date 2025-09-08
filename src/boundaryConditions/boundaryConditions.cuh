/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    A class applying boundary conditions to the lid driven cavity case

Namespace
    LBM

SourceFiles
    boundaryConditions.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BOUNDARYCONDITIONS_CUH
#define __MBLBM_BOUNDARYCONDITIONS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

#include "normalVector.cuh"
#include "boundaryValue.cuh"
#include "boundaryRegion.cuh"
#include "boundaryFields.cuh"

namespace LBM
{
    /**
     * @class boundaryConditions
     * @brief Applies boundary conditions for lid-driven cavity simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice model
     * in lid-driven cavity flow simulations. It handles both static wall boundaries and
     * moving lid boundaries using moment-based boundary conditions derived from the
     * regularized LBM approach.
     */
    class boundaryConditions
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         */
        [[nodiscard]] inline consteval boundaryConditions() {};

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment variables array to be populated
         * @param[in] boundaryNormal Normal vector information at boundary node
         *
         * This method implements the moment-based boundary condition treatment for
         * the D3Q19 lattice model. It handles various boundary types including:
         * - Static wall boundaries (all velocity components zero)
         * - Moving lid boundaries (prescribed tangential velocity)
         * - Corner and edge cases with specialized treatment
         *
         * The method uses the regularized LBM approach to reconstruct boundary
         * moments from available population information, ensuring mass conservation
         * and appropriate stress conditions at boundaries.
         */
        template <class VelocitySet>
        __device__ static inline constexpr void calculateMoments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            const normalVector &boundaryNormal) noexcept
        {
            const scalar_t rho_I = VelocitySet::rho_I(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            switch (boundaryNormal.nodeType())
            {
            // Static boundaries
            case normalVector::SOUTH_WEST_BACK():
            {
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_WEST_FRONT():
            {
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST_BACK():
            {
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST_FRONT():
            {
                const scalar_t rho = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_WEST():
            {
                const scalar_t mxy_I = pop(label_constant<8>()) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = mxy;                      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST():
            {
                const scalar_t mxy_I = -pop(label_constant<13>()) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = mxy;                      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST_BACK():
            {
                const scalar_t mxz_I = pop(label_constant<10>()) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST_FRONT():
            {
                const scalar_t mxz_I = -pop(label_constant<16>()) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST_BACK():
            {
                const scalar_t mxz_I = -pop(label_constant<15>()) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST_FRONT():
            {
                const scalar_t mxz_I = pop(label_constant<9>()) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_BACK():
            {
                const scalar_t myz_I = pop(label_constant<12>()) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = myz;                      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_FRONT():
            {
                const scalar_t myz_I = -pop(label_constant<18>()) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = myz;                      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST():
            {
                const scalar_t mxy_I = (pop(label_constant<8>()) - pop(label_constant<14>())) * inv_rho_I;
                const scalar_t mxz_I = (pop(label_constant<10>()) - pop(label_constant<16>())) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = mxy;                      // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST():
            {
                const scalar_t mxy_I = (pop(label_constant<7>()) - pop(label_constant<13>())) * inv_rho_I;
                const scalar_t mxz_I = (pop(label_constant<9>()) - pop(label_constant<15>())) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = mxy;                      // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH():
            {
                const scalar_t mxy_I = (pop(label_constant<8>()) - pop(label_constant<13>())) * inv_rho_I;
                const scalar_t myz_I = (pop(label_constant<12>()) - pop(label_constant<18>())) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = mxy;                      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = myz;                      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::BACK():
            {
                const scalar_t mxz_I = (pop(label_constant<10>()) - pop(label_constant<15>())) * inv_rho_I;
                const scalar_t myz_I = (pop(label_constant<12>()) - pop(label_constant<17>())) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = myz;                      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::FRONT():
            {
                const scalar_t mxz_I = (pop(label_constant<9>()) - pop(label_constant<16>())) * inv_rho_I;
                const scalar_t myz_I = (pop(label_constant<11>()) - pop(label_constant<18>())) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
                moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
                moments(label_constant<6>()) = mxz;                      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
                moments(label_constant<8>()) = myz;                      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

                return;
            }

            // Lid boundaries
            case normalVector::NORTH():
            {
                const scalar_t mxy_I = (pop(label_constant<7>()) - pop(label_constant<14>())) * inv_rho_I;
                const scalar_t myz_I = (pop(label_constant<11>()) - pop(label_constant<17>())) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = (static_cast<scalar_t>(6) * mxy_I * rho_I - device::u_inf * rho) / (static_cast<scalar_t>(3) * rho);
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = mxy;                           // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = myz;                           // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0);      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0);      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0);      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0);      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0);      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0);      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0);      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0);      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_BACK():
            {
                const scalar_t myz_I = -pop(label_constant<17>()) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(72) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I + static_cast<scalar_t>(2) * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0);      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = myz;                           // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_FRONT():
            {
                const scalar_t myz_I = pop(label_constant<11>()) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(72) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I - static_cast<scalar_t>(2) * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = static_cast<scalar_t>(0);      // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = myz;                           // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST():
            {
                const scalar_t mxy_I = pop(label_constant<7>()) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega + static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho - static_cast<scalar_t>(3) * device::u_inf * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = mxy;                           // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0);      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST():
            {
                const scalar_t mxy_I = -pop(label_constant<14>()) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega - static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho - static_cast<scalar_t>(3) * device::u_inf * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments(label_constant<0>()) = rho;
                moments(label_constant<1>()) = device::u_inf;                 // ux
                moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
                moments(label_constant<3>()) = static_cast<scalar_t>(0);      // uz
                moments(label_constant<4>()) = device::u_inf * device::u_inf; // mxx
                moments(label_constant<5>()) = mxy;                           // mxy
                moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
                moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
                moments(label_constant<8>()) = static_cast<scalar_t>(0);      // myz
                moments(label_constant<9>()) = static_cast<scalar_t>(0);      // mzz

                return;
            }
            }
        }

    private:
    };
}

#endif