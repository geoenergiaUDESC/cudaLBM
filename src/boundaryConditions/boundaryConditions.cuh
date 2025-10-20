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
Authors: Nathan Duggins, Vinicius Czarnobay, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    along with this program. If not, see <https://www.gnu.org/licenses/>.

Description
    A class applying boundary conditions to the turbulent jet case

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

// Fallback print tracker for all 26 boundary cases
__device__ int printedFallback[26] = {false};

namespace LBM
{
    /**
     * @class boundaryConditions
     * @brief Applies boundary conditions for turbulent jet simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice
     * model in turbulent jet flow simulations. It handles static walls, inflow, and
     * outflow boundaries using moment-based boundary conditions derived from the
     * regularized LBM approach.
     **/
    class boundaryConditions
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval boundaryConditions(){};

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment variables array to be populated
         * @param[in] boundaryNormal Normal vector information at boundary node
         *
         * This method implements the moment-based boundary condition treatment
         * for the D3Q19 lattice model. Currently, it handles both the inflow
         * (jet) boundary located at the BACK face of the domain and the outflow
         * boundary located at the FRONT face.
         *
         * The method uses the regularized LBM approach to reconstruct boundary
         * moments from available population information, ensuring mass conservation
         * and appropriate stress conditions at boundaries.
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculateMoments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            const normalVector &boundaryNormal,
            const scalar_t *const ptrRestrict shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: boundaryConditions::calculateMoments only supports D3Q19 and D3Q27.");

            const scalar_t rho_I = VelocitySet::rho_I(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            bool already_handled = false;

            switch (boundaryNormal.nodeType())
            {
            // Static boundaries
            case normalVector::SOUTH_WEST_BACK():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_WEST_FRONT():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST_BACK():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_EAST_FRONT():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_WEST():
            {
                const scalar_t mxy_I = SOUTH_WEST_mxy_I(pop, inv_rho_I);

                moments[m_i<0>()] = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                moments[m_i<1>()] = static_cast<scalar_t>(0);                                                                                         // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                                                                                         // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);                                                                                         // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                                                                                         // mxx
                moments[m_i<5>()] = (static_cast<scalar_t>(36) * mxy_I * rho_I - moments[m_i<0>()]) / (static_cast<scalar_t>(9) * moments[m_i<0>()]); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);                                                                                         // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                                                                                         // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);                                                                                         // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);                                                                                         // mzz

                return;
            }
            case normalVector::SOUTH_EAST():
            {
                const scalar_t mxy_I = SOUTH_EAST_mxy_I(pop, inv_rho_I);

                moments[m_i<0>()] = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                moments[m_i<1>()] = static_cast<scalar_t>(0);                                                                                         // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                                                                                         // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);                                                                                         // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                                                                                         // mxx
                moments[m_i<5>()] = (static_cast<scalar_t>(36) * mxy_I * rho_I + moments[m_i<0>()]) / (static_cast<scalar_t>(9) * moments[m_i<0>()]); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);                                                                                         // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                                                                                         // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);                                                                                         // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);                                                                                         // mzz

                return;
            }
            case normalVector::WEST_BACK():
            {
                const scalar_t mxz_I = WEST_BACK_mxz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST_FRONT():
            {
                const scalar_t mxz_I = WEST_FRONT_mxz_I(pop, inv_rho_I);

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST_BACK():
            {
                const scalar_t mxz_I = EAST_BACK_mxz_I(pop, inv_rho_I);

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST_FRONT():
            {
                const scalar_t mxz_I = EAST_FRONT_mxz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_BACK():
            {
                const scalar_t myz_I = SOUTH_BACK_myz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = myz;                      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH_FRONT():
            {
                const scalar_t myz_I = SOUTH_FRONT_myz_I(pop, inv_rho_I);

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = myz;                      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::WEST():
            {
                const scalar_t mxy_I = WEST_mxy_I(pop, inv_rho_I);
                const scalar_t mxz_I = WEST_mxz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::EAST():
            {
                const scalar_t mxy_I = EAST_mxy_I(pop, inv_rho_I);
                const scalar_t mxz_I = EAST_mxz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::SOUTH():
            {
                const scalar_t mxy_I = SOUTH_mxy_I(pop, inv_rho_I);
                const scalar_t myz_I = SOUTH_myz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = myz;                      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::BACK():
            {
                const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());

                const scalar_t mxz_I = BACK_mxz_I(pop, inv_rho_I);
                const scalar_t myz_I = BACK_myz_I(pop, inv_rho_I);

                const scalar_t rho = rho0<scalar_t>();
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = myz;                      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }
            case normalVector::FRONT():
            {
                const scalar_t mxz_I = FRONT_mxz_I(pop, inv_rho_I);
                const scalar_t myz_I = FRONT_myz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = static_cast<scalar_t>(0); // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0); // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0); // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0); // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0); // myy
                moments[m_i<8>()] = myz;                      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0); // mzz

                return;
            }

            // Lid boundaries
            case normalVector::NORTH():
            {
                const scalar_t mxy_I = NORTH_mxy_I(pop, inv_rho_I);
                const scalar_t myz_I = NORTH_myz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = (static_cast<scalar_t>(6) * mxy_I * rho_I - device::u_inf * rho) / (static_cast<scalar_t>(3) * rho);
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = mxy;                           // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = myz;                           // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_BACK():
            {
                const scalar_t myz_I = NORTH_BACK_myz_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(72) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I + static_cast<scalar_t>(2) * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = myz;                           // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_FRONT():
            {
                const scalar_t myz_I = NORTH_FRONT_myz_I(pop, inv_rho_I);

                const scalar_t rho = -static_cast<scalar_t>(72) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I - static_cast<scalar_t>(2) * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = myz;                           // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_EAST():
            {
                const scalar_t mxy_I = NORTH_EAST_mxy_I(pop, inv_rho_I);

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega + static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho - static_cast<scalar_t>(3) * device::u_inf * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = mxy;                           // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            case normalVector::NORTH_WEST():
            {
                const scalar_t mxy_I = NORTH_WEST_mxy_I(pop, inv_rho_I);

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega - static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho - static_cast<scalar_t>(3) * device::u_inf * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<1>()] = device::u_inf;                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);      // uy
                moments[m_i<3>()] = static_cast<scalar_t>(0);      // uz
                moments[m_i<4>()] = device::u_inf * device::u_inf; // mxx
                moments[m_i<5>()] = mxy;                           // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);      // myy
                moments[m_i<8>()] = static_cast<scalar_t>(0);      // myz
                moments[m_i<9>()] = static_cast<scalar_t>(0);      // mzz

                return;
            }
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_SOUTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return pop[q_i<8>()] * inv_rho_I;
            }
            else
            {
                return (pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_SOUTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -pop[q_i<13>()] * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<10>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<16>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<15>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<9>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<12>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<18>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<8>()]) - (pop[q_i<14>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) - (pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<10>()]) - (pop[q_i<16>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) - (pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<7>()]) - (pop[q_i<13>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) - (pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<9>()]) - (pop[q_i<15>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) - (pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<8>()]) - (pop[q_i<13>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) - (pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<12>()]) - (pop[q_i<18>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) - (pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<10>()]) - (pop[q_i<15>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) - (pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<12>()]) - (pop[q_i<17>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) - (pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<9>()]) - (pop[q_i<16>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) - (pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<11>()]) - (pop[q_i<18>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) - (pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<7>()]) - (pop[q_i<14>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) - (pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<11>()]) - (pop[q_i<17>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) - (pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<17>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<11>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_NORTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<7>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_NORTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<14>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()]) * inv_rho_I;
            }
        }
    };
}

#endif