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
     **/
    class boundaryConditions
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
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
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculateMoments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            const normalVector &boundaryNormal) noexcept
        {
            // Branchless computation of rho_I and 1 / rho_I
            const scalar_t rho_I = VelocitySet::rho_I(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            // Branchless computation of u, v and w
            moments(label_constant<1>()) =
                static_cast<scalar_t>(
                    normalVector::NORTH() |
                    normalVector::NORTH_WEST_BACK() |
                    normalVector::NORTH_WEST_FRONT() |
                    normalVector::NORTH_EAST_BACK() |
                    normalVector::NORTH_EAST_FRONT() |
                    normalVector::NORTH_BACK() |
                    normalVector::NORTH_FRONT() |
                    normalVector::NORTH_EAST() |
                    normalVector::NORTH_WEST()) *
                device::u_inf;
            moments(label_constant<2>()) = static_cast<scalar_t>(0);
            moments(label_constant<3>()) = static_cast<scalar_t>(0);

            // Branchless computation of mxx
            moments(label_constant<4>()) =
                static_cast<scalar_t>(
                    normalVector::NORTH() |
                    normalVector::NORTH_WEST_BACK() |
                    normalVector::NORTH_WEST_FRONT() |
                    normalVector::NORTH_EAST_BACK() |
                    normalVector::NORTH_EAST_FRONT() |
                    normalVector::NORTH_BACK() |
                    normalVector::NORTH_FRONT() |
                    normalVector::NORTH_EAST() |
                    normalVector::NORTH_WEST()) *
                (device::u_inf * device::u_inf);

            // Branchless computation of myy
            moments(label_constant<7>()) = static_cast<scalar_t>(0);

            // Branchless computation of mzz
            moments(label_constant<9>()) = static_cast<scalar_t>(0);

            // Branchless computation of mxy_I
            const scalar_t mxy_I =
                (normalVector::SOUTH_WEST() * (pop(label_constant<8>()) * inv_rho_I)) +
                (normalVector::SOUTH_EAST() * (-pop(label_constant<13>()) * inv_rho_I)) +
                (normalVector::WEST() * ((pop(label_constant<8>()) - pop(label_constant<14>())) * inv_rho_I)) +
                (normalVector::EAST() * ((pop(label_constant<7>()) - pop(label_constant<13>())) * inv_rho_I)) +
                (normalVector::SOUTH() * ((pop(label_constant<8>()) - pop(label_constant<13>())) * inv_rho_I)) +
                (normalVector::NORTH() * ((pop(label_constant<7>()) - pop(label_constant<14>())) * inv_rho_I)) +
                (normalVector::NORTH_EAST() * (pop(label_constant<7>()) * inv_rho_I)) +
                (normalVector::NORTH_WEST() * (-pop(label_constant<14>()) * inv_rho_I));

            // Branchless computation of mxz_I
            const scalar_t mxz_I =
                (normalVector::WEST_BACK() * (pop(label_constant<10>()) * inv_rho_I)) +
                (normalVector::WEST_FRONT() * (-pop(label_constant<16>()) * inv_rho_I)) +
                (normalVector::EAST_BACK() * (-pop(label_constant<15>()) * inv_rho_I)) +
                (normalVector::EAST_FRONT() * (pop(label_constant<9>()) * inv_rho_I)) +
                (normalVector::WEST() * ((pop(label_constant<10>()) - pop(label_constant<16>())) * inv_rho_I)) +
                (normalVector::EAST() * ((pop(label_constant<9>()) - pop(label_constant<15>())) * inv_rho_I)) +
                (normalVector::BACK() * ((pop(label_constant<10>()) - pop(label_constant<15>())) * inv_rho_I)) +
                (normalVector::FRONT() * ((pop(label_constant<9>()) - pop(label_constant<16>())) * inv_rho_I));

            // Branchless computation of myz_I
            const scalar_t myz_I =
                (normalVector::SOUTH_BACK() * (pop(label_constant<12>()) * inv_rho_I)) +
                (normalVector::SOUTH_FRONT() * (-pop(label_constant<18>()) * inv_rho_I)) +
                (normalVector::SOUTH() * ((pop(label_constant<12>()) - pop(label_constant<18>())) * inv_rho_I)) +
                (normalVector::BACK() * ((pop(label_constant<12>()) - pop(label_constant<17>())) * inv_rho_I)) +
                (normalVector::FRONT() * ((pop(label_constant<11>()) - pop(label_constant<18>())) * inv_rho_I)) +
                (normalVector::NORTH() * ((pop(label_constant<11>()) - pop(label_constant<17>())) * inv_rho_I)) +
                (normalVector::NORTH_BACK() * (-pop(label_constant<17>()) * inv_rho_I)) +
                (normalVector::NORTH_FRONT() * (pop(label_constant<11>()) * inv_rho_I));

            // Branchless computation of rho
            const scalar_t rho =
                (normalVector::SOUTH_WEST_BACK() * (static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7))) +
                (normalVector::SOUTH_WEST_FRONT() * (static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7))) +
                (normalVector::SOUTH_EAST_BACK() * (static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7))) +
                (normalVector::SOUTH_EAST_FRONT() * (static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7))) +
                (normalVector::SOUTH_WEST() * (static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::SOUTH_EAST() * (-static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::WEST_BACK() * (static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::WEST_FRONT() * (-static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::EAST_BACK() * (-static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::EAST_FRONT() * (static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::SOUTH_BACK() * (static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::SOUTH_FRONT() * (-static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (normalVector::WEST() * (static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5))) +
                (normalVector::EAST() * (static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5))) +
                (normalVector::SOUTH() * (static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5))) +
                (normalVector::BACK() * (static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5))) +
                (normalVector::FRONT() * (static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5))) +
                (normalVector::NORTH() * (static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5))) +
                (normalVector::NORTH_WEST_BACK() * (-static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (normalVector::NORTH_WEST_FRONT() * (-static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (normalVector::NORTH_EAST_BACK() * (-static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (normalVector::NORTH_EAST_FRONT() * (-static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (normalVector::NORTH_BACK() * (static_cast<scalar_t>(72) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega))) +
                (normalVector::NORTH_FRONT() * (-static_cast<scalar_t>(72) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega))) +
                (normalVector::NORTH_EAST() * (static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega + static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega))) +
                (normalVector::NORTH_WEST() * (-static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega - static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega)));
            moments(label_constant<0>()) = rho;

            // Branchless computation of mxy
            moments(label_constant<5>()) =
                (normalVector::SOUTH_WEST() * ((static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::SOUTH_EAST() * ((static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::WEST() * (static_cast<scalar_t>(2) * mxy_I * rho_I / rho)) +
                (normalVector::EAST() * (static_cast<scalar_t>(2) * mxy_I * rho_I / rho)) +
                (normalVector::SOUTH() * (static_cast<scalar_t>(2) * mxy_I * rho_I / rho)) +
                (normalVector::NORTH() * ((static_cast<scalar_t>(6) * mxy_I * rho_I - device::u_inf * rho) / (static_cast<scalar_t>(3) * rho))) +
                (normalVector::NORTH_EAST() * ((static_cast<scalar_t>(36) * mxy_I * rho_I - rho - static_cast<scalar_t>(3) * device::u_inf * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::NORTH_WEST() * ((static_cast<scalar_t>(36) * mxy_I * rho_I + rho - static_cast<scalar_t>(3) * device::u_inf * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(9) * rho)));

            // Branchless computation of mxz
            moments(label_constant<6>()) =
                (normalVector::WEST_BACK() * ((static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::WEST_FRONT() * ((static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::EAST_BACK() * ((static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::EAST_FRONT() * ((static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::WEST() * (static_cast<scalar_t>(2) * mxz_I * rho_I / rho)) +
                (normalVector::EAST() * (static_cast<scalar_t>(2) * mxz_I * rho_I / rho)) +
                (normalVector::BACK() * (static_cast<scalar_t>(2) * mxz_I * rho_I / rho)) +
                (normalVector::FRONT() * (static_cast<scalar_t>(2) * mxz_I * rho_I / rho));

            // Branchless computation of myz
            moments(label_constant<8>()) =
                (normalVector::SOUTH_BACK() * ((static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::SOUTH_FRONT() * ((static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho))) +
                (normalVector::SOUTH() * (static_cast<scalar_t>(2) * myz_I * rho_I / rho)) +
                (normalVector::BACK() * (static_cast<scalar_t>(2) * myz_I * rho_I / rho)) +
                (normalVector::FRONT() * (static_cast<scalar_t>(2) * myz_I * rho_I / rho)) +
                (normalVector::NORTH() * (static_cast<scalar_t>(2) * myz_I * rho_I / rho)) +
                (normalVector::NORTH_BACK() * ((static_cast<scalar_t>(72) * myz_I * rho_I + static_cast<scalar_t>(2) * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(18) * rho))) +
                (normalVector::NORTH_FRONT() * ((static_cast<scalar_t>(72) * myz_I * rho_I - static_cast<scalar_t>(2) * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(18) * rho)));
        }

    private:
    };
}

#endif