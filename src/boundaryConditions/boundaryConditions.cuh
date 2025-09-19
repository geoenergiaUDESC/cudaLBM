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
    template <class VelocitySet>
    class boundaryConditions
    {
    public:
        /**
         * @brief Default constructor
         * @param[in] pop Population density array at current lattice node
         * @param[in] boundaryNormal Normal vector information at boundary node
         **/
        __device__ [[nodiscard]] inline boundaryConditions(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            const normalVector &boundaryNormal) noexcept
            : rhoI(VelocitySet::rho_I(pop, boundaryNormal)),
              inv_rhoI(static_cast<scalar_t>(1) / rhoI),
              mxyI(mxy_I(pop, boundaryNormal, inv_rhoI)),
              mxzI(mxz_I(pop, boundaryNormal, inv_rhoI)),
              myzI(myz_I(pop, boundaryNormal, inv_rhoI)){};

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
        __device__ inline constexpr void calculateMoments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            const normalVector &boundaryNormal) const noexcept
        {
            // Branchless computation of rho
            const scalar_t rho =
                (static_cast<scalar_t>(boundaryNormal.SouthWestBack<bool>() || boundaryNormal.SouthWestFront<bool>() || boundaryNormal.SouthEastBack<bool>() || boundaryNormal.SouthEastFront<bool>()) *
                 (static_cast<scalar_t>(12) * rhoI / static_cast<scalar_t>(7))) +
                (boundaryNormal.SouthWest<scalar_t>() * (static_cast<scalar_t>(36) * (rhoI - mxyI * rhoI + mxyI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (boundaryNormal.SouthEast<scalar_t>() * (-static_cast<scalar_t>(36) * (-rhoI - mxyI * rhoI + mxyI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (boundaryNormal.WestBack<scalar_t>() * (static_cast<scalar_t>(36) * (rhoI - mxzI * rhoI + mxzI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (boundaryNormal.WestFront<scalar_t>() * (-static_cast<scalar_t>(36) * (-rhoI - mxzI * rhoI + mxzI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (boundaryNormal.EastBack<scalar_t>() * (-static_cast<scalar_t>(36) * (-rhoI - mxzI * rhoI + mxzI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (boundaryNormal.EastFront<scalar_t>() * (static_cast<scalar_t>(36) * (rhoI - mxzI * rhoI + mxzI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (boundaryNormal.SouthBack<scalar_t>() * (static_cast<scalar_t>(36) * (rhoI - myzI * rhoI + myzI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (boundaryNormal.SouthFront<scalar_t>() * (-static_cast<scalar_t>(36) * (-rhoI - myzI * rhoI + myzI * rhoI * device::omega) / (static_cast<scalar_t>(24) + device::omega))) +
                (static_cast<scalar_t>(boundaryNormal.West<bool>() || boundaryNormal.East<bool>() || boundaryNormal.South<bool>() || boundaryNormal.Back<bool>() || boundaryNormal.Front<bool>() || boundaryNormal.North<bool>()) *
                 (static_cast<scalar_t>(6) * rhoI / static_cast<scalar_t>(5))) +
                (boundaryNormal.NorthWestBack<scalar_t>() * (-static_cast<scalar_t>(24) * rhoI / (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (boundaryNormal.NorthWestFront<scalar_t>() * (-static_cast<scalar_t>(24) * rhoI / (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (boundaryNormal.NorthEastBack<scalar_t>() * (-static_cast<scalar_t>(24) * rhoI / (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (boundaryNormal.NorthEastFront<scalar_t>() * (-static_cast<scalar_t>(24) * rhoI / (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::u_inf + static_cast<scalar_t>(9) * device::u_inf * device::u_inf))) +
                (boundaryNormal.NorthBack<scalar_t>() * (static_cast<scalar_t>(72) * (-rhoI - myzI * rhoI + myzI * rhoI * device::omega) / (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega))) +
                (boundaryNormal.NorthFront<scalar_t>() * (-static_cast<scalar_t>(72) * (rhoI - myzI * rhoI + myzI * rhoI * device::omega) / (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega))) +
                (boundaryNormal.NorthEast<scalar_t>() * (static_cast<scalar_t>(36) * (rhoI - mxyI * rhoI + mxyI * rhoI * device::omega) / (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega + static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega))) +
                (boundaryNormal.NorthWest<scalar_t>() * (-static_cast<scalar_t>(36) * (-rhoI - mxyI * rhoI + mxyI * rhoI * device::omega) / (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device::u_inf - static_cast<scalar_t>(18) * device::u_inf * device::u_inf + device::omega - static_cast<scalar_t>(3) * device::u_inf * device::omega + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * device::omega)));
            moments(label_constant<0>()) = rho;

            // Branchless computation of u, v and w
            moments(label_constant<1>()) =
                (static_cast<scalar_t>(
                     boundaryNormal.North<bool>() ||
                     boundaryNormal.NorthWestBack<bool>() ||
                     boundaryNormal.NorthWestFront<bool>() ||
                     boundaryNormal.NorthEastBack<bool>() ||
                     boundaryNormal.NorthEastFront<bool>() ||
                     boundaryNormal.NorthBack<bool>() ||
                     boundaryNormal.NorthFront<bool>() ||
                     boundaryNormal.NorthEast<bool>() ||
                     boundaryNormal.NorthWest<bool>()) *
                 device::u_inf);
            moments(label_constant<2>()) = static_cast<scalar_t>(0);
            moments(label_constant<3>()) = static_cast<scalar_t>(0);

            // Branchless computation of mxx
            moments(label_constant<4>()) =
                (static_cast<scalar_t>(
                     boundaryNormal.North<bool>() ||
                     boundaryNormal.NorthWestBack<bool>() ||
                     boundaryNormal.NorthWestFront<bool>() ||
                     boundaryNormal.NorthEastBack<bool>() ||
                     boundaryNormal.NorthEastFront<bool>() ||
                     boundaryNormal.NorthBack<bool>() ||
                     boundaryNormal.NorthFront<bool>() ||
                     boundaryNormal.NorthEast<bool>() ||
                     boundaryNormal.NorthWest<bool>()) *
                 (device::u_inf * device::u_inf));

            // Branchless computation of mxy
            moments(label_constant<5>()) =
                ((boundaryNormal.SouthWest<scalar_t>() * ((static_cast<scalar_t>(36) * mxyI * rhoI - rho) / (static_cast<scalar_t>(9) * rho))) +
                 (boundaryNormal.SouthEast<scalar_t>() * ((static_cast<scalar_t>(36) * mxyI * rhoI + rho) / (static_cast<scalar_t>(9) * rho))) +
                 (static_cast<scalar_t>(boundaryNormal.West<bool>() || boundaryNormal.East<bool>() || boundaryNormal.South<bool>()) * (static_cast<scalar_t>(2) * mxyI * rhoI / rho)) +
                 (boundaryNormal.North<scalar_t>() * ((static_cast<scalar_t>(6) * mxyI * rhoI - device::u_inf * rho) / (static_cast<scalar_t>(3) * rho))) +
                 (boundaryNormal.NorthEast<scalar_t>() * ((static_cast<scalar_t>(36) * mxyI * rhoI - rho - static_cast<scalar_t>(3) * device::u_inf * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(9) * rho))) +
                 (boundaryNormal.NorthWest<scalar_t>() * ((static_cast<scalar_t>(36) * mxyI * rhoI + rho - static_cast<scalar_t>(3) * device::u_inf * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(9) * rho))));

            // Branchless computation of mxz
            moments(label_constant<6>()) =
                (boundaryNormal.WestBack<scalar_t>() * ((static_cast<scalar_t>(36) * mxzI * rhoI - rho) / (static_cast<scalar_t>(9) * rho))) +
                (boundaryNormal.WestFront<scalar_t>() * ((static_cast<scalar_t>(36) * mxzI * rhoI + rho) / (static_cast<scalar_t>(9) * rho))) +
                (boundaryNormal.EastBack<scalar_t>() * ((static_cast<scalar_t>(36) * mxzI * rhoI + rho) / (static_cast<scalar_t>(9) * rho))) +
                (boundaryNormal.EastFront<scalar_t>() * ((static_cast<scalar_t>(36) * mxzI * rhoI - rho) / (static_cast<scalar_t>(9) * rho))) +
                ((static_cast<scalar_t>(boundaryNormal.West<bool>() || boundaryNormal.East<bool>() || boundaryNormal.Back<bool>() || boundaryNormal.Front<bool>())) *
                 (static_cast<scalar_t>(2) * mxzI * rhoI / rho));

            // Branchless computation of myy
            moments(label_constant<7>()) = static_cast<scalar_t>(0);

            // Branchless computation of myz
            moments(label_constant<8>()) =
                ((boundaryNormal.SouthBack<scalar_t>() * ((static_cast<scalar_t>(36) * myzI * rhoI - rho) / (static_cast<scalar_t>(9) * rho))) +
                 (boundaryNormal.SouthFront<scalar_t>() * ((static_cast<scalar_t>(36) * myzI * rhoI + rho) / (static_cast<scalar_t>(9) * rho))) +
                 (boundaryNormal.South<scalar_t>() * (static_cast<scalar_t>(2) * myzI * rhoI / rho)) +
                 (boundaryNormal.Back<scalar_t>() * (static_cast<scalar_t>(2) * myzI * rhoI / rho)) +
                 (boundaryNormal.Front<scalar_t>() * (static_cast<scalar_t>(2) * myzI * rhoI / rho)) +
                 (boundaryNormal.North<scalar_t>() * (static_cast<scalar_t>(2) * myzI * rhoI / rho)) +
                 (boundaryNormal.NorthBack<scalar_t>() * ((static_cast<scalar_t>(72) * myzI * rhoI + static_cast<scalar_t>(2) * rho - static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(18) * rho))) +
                 (boundaryNormal.NorthFront<scalar_t>() * ((static_cast<scalar_t>(72) * myzI * rhoI - static_cast<scalar_t>(2) * rho + static_cast<scalar_t>(3) * device::u_inf * device::u_inf * rho) / (static_cast<scalar_t>(18) * rho))));

            // Branchless computation of mzz
            moments(label_constant<9>()) = static_cast<scalar_t>(0);
        }

    private:
        /**
         * @brief Stores the incoming density and moments
         **/
        const scalar_t rhoI;
        const scalar_t inv_rhoI;
        const scalar_t mxyI;
        const scalar_t mxzI;
        const scalar_t myzI;

        /**
         * @brief Branchless computation of the x/y off-diagonal incoming moment
         * @tparam Q Number of elements in the velocity set
         * @param[in] pop Population density array at current lattice node
         * @param[in] boundaryNormal Normal vector information at boundary node]
         * @param[in] inv_rho_I Inverse of the incoming density
         * @return The x/y off-diagonal incoming moment
         **/
        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const normalVector &boundaryNormal,
            const scalar_t inv_rho_I) noexcept
        {
            return inv_rho_I *
                   (pop(label_constant<8>()) * static_cast<scalar_t>(boundaryNormal.West<bool>() + boundaryNormal.South<bool>() + boundaryNormal.SouthWest<bool>()) +
                    pop(label_constant<7>()) * static_cast<scalar_t>(boundaryNormal.East<bool>() + boundaryNormal.North<bool>() + boundaryNormal.NorthEast<bool>()) -
                    pop(label_constant<13>()) * static_cast<scalar_t>(boundaryNormal.East<bool>() + boundaryNormal.South<bool>() + boundaryNormal.SouthEast<bool>()) -
                    pop(label_constant<14>()) * static_cast<scalar_t>(boundaryNormal.West<bool>() + boundaryNormal.North<bool>() + boundaryNormal.NorthWest<bool>()));
        }

        /**
         * @brief Branchless computation of the x/z off-diagonal incoming moment
         * @tparam Q Number of elements in the velocity set
         * @param[in] pop Population density array at current lattice node
         * @param[in] boundaryNormal Normal vector information at boundary node]
         * @param[in] inv_rho_I Inverse of the incoming density
         * @return The x/z off-diagonal incoming moment
         **/
        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const normalVector &boundaryNormal,
            const scalar_t inv_rho_I) noexcept
        {
            return inv_rho_I *
                   (pop(label_constant<9>()) * static_cast<scalar_t>(boundaryNormal.East<bool>() + boundaryNormal.Front<bool>() + boundaryNormal.EastFront<bool>()) +
                    pop(label_constant<10>()) * static_cast<scalar_t>(boundaryNormal.West<bool>() + boundaryNormal.Back<bool>() + boundaryNormal.WestBack<bool>()) -
                    pop(label_constant<15>()) * static_cast<scalar_t>(boundaryNormal.East<bool>() + boundaryNormal.Back<bool>() + boundaryNormal.EastBack<bool>()) -
                    pop(label_constant<16>()) * static_cast<scalar_t>(boundaryNormal.West<bool>() + boundaryNormal.Front<bool>() + boundaryNormal.WestFront<bool>()));
        }

        /**
         * @brief Branchless computation of the y/z off-diagonal incoming moment
         * @tparam Q Number of elements in the velocity set
         * @param[in] pop Population density array at current lattice node
         * @param[in] boundaryNormal Normal vector information at boundary node]
         * @param[in] inv_rho_I Inverse of the incoming density
         * @return The y/z off-diagonal incoming moment
         **/
        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t myz_I(
            const thread::array<scalar_t, Q> &pop,
            const normalVector &boundaryNormal,
            const scalar_t inv_rho_I) noexcept
        {
            return inv_rho_I *
                   (pop(label_constant<11>()) * static_cast<scalar_t>(boundaryNormal.North<bool>() + boundaryNormal.Front<bool>() + boundaryNormal.NorthFront<bool>()) +
                    pop(label_constant<12>()) * static_cast<scalar_t>(boundaryNormal.South<bool>() + boundaryNormal.Back<bool>() + boundaryNormal.SouthBack<bool>()) -
                    pop(label_constant<17>()) * static_cast<scalar_t>(boundaryNormal.North<bool>() + boundaryNormal.Back<bool>() + boundaryNormal.NorthBack<bool>()) -
                    pop(label_constant<18>()) * static_cast<scalar_t>(boundaryNormal.South<bool>() + boundaryNormal.Front<bool>() + boundaryNormal.SouthFront<bool>()));
        }
    };
}

#endif