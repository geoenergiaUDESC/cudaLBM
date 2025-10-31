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

#include "include/printOnce.cuh"
#include "normalVector.cuh"
#include "boundaryValue.cuh"
#include "boundaryRegion.cuh"
#include "boundaryFields.cuh"

#define FACES
#define CORNERS
// #define EDGES

namespace LBM
{
    /**
     * @class boundaryConditions
     * @brief Applies boundary conditions for lid-driven cavity simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice
     * model in turbulent jet flow simulations. It handles static wall, inflow, and
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
                // Round inflow + static when is_jet == 0
                case normalVector::BACK():
                {
                    const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                    const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                    const scalar_t R = static_cast<scalar_t>(0.5) * device::L_char;
                    const scalar_t is_jet = static_cast<scalar_t>(
                        (static_cast<scalar_t>(x) - static_cast<scalar_t>(device::nx - 1) / static_cast<scalar_t>(2)) *
                        (static_cast<scalar_t>(x) - static_cast<scalar_t>(device::nx - 1) / static_cast<scalar_t>(2)) +
                        (static_cast<scalar_t>(y) - static_cast<scalar_t>(device::ny - 1) / static_cast<scalar_t>(2)) *
                        (static_cast<scalar_t>(y) - static_cast<scalar_t>(device::ny - 1) / static_cast<scalar_t>(2)) <
                        (R * R));

                    const scalar_t mxz_I = BACK_mxz_I(pop, inv_rho_I);
                    const scalar_t myz_I = BACK_myz_I(pop, inv_rho_I);

                    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                    const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                    const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                    moments(label_constant<0>()) = rho;
                    moments(label_constant<1>()) = static_cast<scalar_t>(0);
                    moments(label_constant<2>()) = static_cast<scalar_t>(0);
                    moments(label_constant<3>()) = is_jet * device::u_inf;
                    moments(label_constant<4>()) = static_cast<scalar_t>(0);
                    moments(label_constant<5>()) = static_cast<scalar_t>(0);
                    moments(label_constant<6>()) = mxz;
                    moments(label_constant<7>()) = static_cast<scalar_t>(0);
                    moments(label_constant<8>()) = myz;
                    moments(label_constant<9>()) = is_jet * ((static_cast<scalar_t>(6) * device::u_inf * device::u_inf * rho_I) / static_cast<scalar_t>(5));

                    already_handled = true;

                    return;
                }

                // Boundary cases are ordered following:
                // WEST, EAST comes first
                // SOUTH, NORTH comes next
                // BACK, FRONT comes last

                // Outflow (zero-gradient) boundaries
                #include "include/IRBCNeumann.cuh"
                //#include "include/Neumann.cuh"

                // Call static boundaries for uncovered cases
                default:
                {
                    if (!already_handled)
                    {
                        switch (boundaryNormal.nodeType())
                        {
                            #include "include/fallback.cuh"
                        }
                    }

                    break;
                }
            }
        }

    private:
        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_WEST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return pop(label_constant<8>()) * inv_rho_I;
            }
            else
            {
                return (pop(label_constant<8>()) + pop(label_constant<20>()) + pop(label_constant<22>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_EAST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -pop(label_constant<13>()) * inv_rho_I;
            }
            else
            {
                return -(pop(label_constant<13>()) + pop(label_constant<23>()) + pop(label_constant<26>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop(label_constant<10>())) * inv_rho_I;
            }
            else
            {
                return (pop(label_constant<10>()) + pop(label_constant<20>()) + pop(label_constant<24>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop(label_constant<16>())) * inv_rho_I;
            }
            else
            {
                return -(pop(label_constant<16>()) + pop(label_constant<22>()) + pop(label_constant<25>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop(label_constant<15>())) * inv_rho_I;
            }
            else
            {
                return -(pop(label_constant<15>()) + pop(label_constant<21>()) + pop(label_constant<26>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop(label_constant<9>())) * inv_rho_I;
            }
            else
            {
                return (pop(label_constant<9>()) + pop(label_constant<19>()) + pop(label_constant<23>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop(label_constant<12>())) * inv_rho_I;
            }
            else
            {
                return (pop(label_constant<12>()) + pop(label_constant<20>()) + pop(label_constant<26>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop(label_constant<18>())) * inv_rho_I;
            }
            else
            {
                return -(pop(label_constant<18>()) + pop(label_constant<22>()) + pop(label_constant<23>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<8>())) - (pop(label_constant<14>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<8>()) + pop(label_constant<20>()) + pop(label_constant<22>())) - (pop(label_constant<14>()) + pop(label_constant<24>()) + pop(label_constant<25>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<10>())) - (pop(label_constant<16>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<10>()) + pop(label_constant<20>()) + pop(label_constant<24>())) - (pop(label_constant<16>()) + pop(label_constant<22>()) + pop(label_constant<25>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<7>())) - (pop(label_constant<13>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<7>()) + pop(label_constant<19>()) + pop(label_constant<21>())) - (pop(label_constant<13>()) + pop(label_constant<23>()) + pop(label_constant<26>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<9>())) - (pop(label_constant<15>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<9>()) + pop(label_constant<19>()) + pop(label_constant<23>())) - (pop(label_constant<15>()) + pop(label_constant<21>()) + pop(label_constant<26>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<8>())) - (pop(label_constant<13>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<8>()) + pop(label_constant<20>()) + pop(label_constant<22>())) - (pop(label_constant<13>()) + pop(label_constant<23>()) + pop(label_constant<26>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<12>())) - (pop(label_constant<18>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<12>()) + pop(label_constant<20>()) + pop(label_constant<26>())) - (pop(label_constant<18>()) + pop(label_constant<22>()) + pop(label_constant<23>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<10>())) - (pop(label_constant<15>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<10>()) + pop(label_constant<20>()) + pop(label_constant<24>())) - (pop(label_constant<15>()) + pop(label_constant<21>()) + pop(label_constant<26>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<12>())) - (pop(label_constant<17>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<12>()) + pop(label_constant<20>()) + pop(label_constant<26>())) - (pop(label_constant<17>()) + pop(label_constant<21>()) + pop(label_constant<24>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<9>())) - (pop(label_constant<16>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<9>()) + pop(label_constant<19>()) + pop(label_constant<23>())) - (pop(label_constant<16>()) + pop(label_constant<22>()) + pop(label_constant<25>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<11>())) - (pop(label_constant<18>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<11>()) + pop(label_constant<19>()) + pop(label_constant<25>())) - (pop(label_constant<18>()) + pop(label_constant<22>()) + pop(label_constant<23>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<7>())) - (pop(label_constant<14>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<7>()) + pop(label_constant<19>()) + pop(label_constant<21>())) - (pop(label_constant<14>()) + pop(label_constant<24>()) + pop(label_constant<25>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop(label_constant<11>())) - (pop(label_constant<17>()))) * inv_rho_I;
            }
            else
            {
                return ((pop(label_constant<11>()) + pop(label_constant<19>()) + pop(label_constant<25>())) - (pop(label_constant<17>()) + pop(label_constant<21>()) + pop(label_constant<24>()))) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop(label_constant<17>())) * inv_rho_I;
            }
            else
            {
                return -(pop(label_constant<17>()) + pop(label_constant<21>()) + pop(label_constant<24>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop(label_constant<11>())) * inv_rho_I;
            }
            else
            {
                return (pop(label_constant<11>()) + pop(label_constant<19>()) + pop(label_constant<25>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_EAST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop(label_constant<7>())) * inv_rho_I;
            }
            else
            {
                return (pop(label_constant<7>()) + pop(label_constant<19>()) + pop(label_constant<21>())) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_WEST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop(label_constant<14>())) * inv_rho_I;
            }
            else
            {
                return -(pop(label_constant<14>()) + pop(label_constant<24>()) + pop(label_constant<25>())) * inv_rho_I;
            }
        }
    };
}

#endif