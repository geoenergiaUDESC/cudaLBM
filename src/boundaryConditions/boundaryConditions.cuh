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

namespace LBM
{
    /**
     * @class boundaryConditions
     * @brief Applies boundary conditions for turbulent jet simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice 
     * model in turbulent jet flow simulations. It handles both static wall and
     * inflow boundaries using moment-based boundary conditions derived from the
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
         * for the D3Q19 lattice model. Currently, it handles only the inflow  
         * (jet) boundary located at the BACK face of the domain. The inflow 
         * imposes a round, piecewise prescribed axial velocity profile in the
         * center of the inlet plane. All other boundaries are treated as static.
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
            const scalar_t* shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: boundaryConditions::calculateMoments only supports D3Q19 and D3Q27.");

            const scalar_t rho_I = velocitySet::rho_I<VelocitySet>(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            bool already_handled = false;

            switch (boundaryNormal.nodeType())
            {   
                // Round inflow + no-slip 
                case normalVector::BACK():
                {
                    const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                    const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                    constexpr const scalar_t R = static_cast<scalar_t>(6);
                    const scalar_t is_jet = static_cast<scalar_t>( 
                        (x - (device::nx - 1) / 2) * (x - (device::nx - 1) / 2) + 
                        (y - (device::ny - 1) / 2) * (y - (device::ny - 1) / 2) < 
                        (R * R) 
                    );

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

                // No-slip at the corners and edges
                #include "backNoSlip.cuh"

                // Outflow (zero-gradient) boundaries
                //case normalVector::EAST():
                //case normalVector::WEST():
                //case normalVector::NORTH():
                //case normalVector::SOUTH():
                case normalVector::FRONT():
                {
                    const int3 offset = boundaryNormal.interiorOffset();
                    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto moment)
                        {
                            const label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
                            moments[moment] = shared_buffer[ID];
                        });

                    already_handled = true;
                        
                    return;
                }
                
                // Call static boundaries for uncovered cases
                default:
                {
                    if (!already_handled)
                    {
                        switch (boundaryNormal.nodeType())
                        {
                            #include "jetFallback.cuh"
                        }
                    }

                    break;
                }
            }
        }

    private:
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