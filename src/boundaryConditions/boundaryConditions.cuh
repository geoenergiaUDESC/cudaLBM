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

                // Outflow (zero-gradient) boundaries
                #include "include/IRBCNeumann.cuh"
                // #include "include/experimentalBC.cuh"

                // Call static boundaries for uncovered cases
                default:
                {
                    if (!already_handled)
                    {
                        switch (boundaryNormal.nodeType())
                        {
                            #include "boundaryFallback.cuh"
                        }
                    }

                    break;
                }
            }
        }

    private:
        template <typename T = uint8_t>
        __device__ static inline void printOnce(
            const T caseId, 
            const char* name) noexcept
        {
            if (atomicExch(&printedFallback[caseId], true) == false)
            {
                printf("[fallback] %s applied\n", name);
            }
        }

        __device__ [[nodiscard]] static inline constexpr int boundaryTarget(
            const int nThreads, 
            const int offset) noexcept
        {
            return (offset > 0) ? 0 : (offset < 0 ? nThreads - 1 : (nThreads >> 1));
        }

        __device__ [[nodiscard]] static inline bool isBoundaryThread(const int3 offset) noexcept
        {
            const int tx = boundaryTarget(block::nx(), offset.x);
            const int ty = boundaryTarget(block::ny(), offset.y);
            const int tz = boundaryTarget(block::nz(), offset.z);

            return (static_cast<int>(threadIdx.x) == tx) && (static_cast<int>(threadIdx.y) == ty) && (static_cast<int>(threadIdx.z) == tz);
        }

        template <typename T = const char*>
        __device__ static inline void printThreadMapping(
            const T label,
            const int3 offset) noexcept
        {
            if (!isBoundaryThread(offset)) return;

            const int gx = threadIdx.x + block::nx() * blockIdx.x;
            const int gy = threadIdx.y + block::ny() * blockIdx.y;
            const int gz = threadIdx.z + block::nz() * blockIdx.z;

            const int gx_int = gx + offset.x;
            const int gy_int = gy + offset.y;
            const int gz_int = gz + offset.z;

            const int bx = (offset.x == 0); 
            const int by = (offset.y == 0); 
            const int bz = (offset.z == 0); 
            const int key = (bx) | (by << 1) | (bz << 2);

            switch (key)
            {
                case 0: 
                    printf("[%s] global=(%d,%d,%d) -> interior=(%d,%d,%d)\n",
                        label, gx, gy, gz, gx_int, gy_int, gz_int);
                    break;
                case 1: 
                    printf("[%s] global=(%c,%d,%d) -> interior=(%c,%d,%d)\n",
                        label, 'x', gy, gz, 'x', gy_int, gz_int);
                    break;
                case 2:
                    printf("[%s] global=(%d,%c,%d) -> interior=(%d,%c,%d)\n",
                        label, gx, 'y', gz, gx_int, 'y', gz_int);
                    break;
                case 3: 
                    printf("[%s] global=(%c,%c,%d) -> interior=(%c,%c,%d)\n",
                        label, 'x', 'y', gz, 'x', 'y', gz_int);
                    break;
                case 4: 
                    printf("[%s] global=(%d,%d,%c) -> interior=(%d,%d,%c)\n",
                        label, gx, gy, 'z', gx_int, gy_int, 'z');
                    break;
                case 5:
                    printf("[%s] global=(%c,%d,%c) -> interior=(%c,%d,%c)\n",
                        label, 'x', gy, 'z', 'x', gy_int, 'z');
                    break;
                case 6: 
                    printf("[%s] global=(%d,%c,%c) -> interior=(%d,%c,%c)\n",
                        label, gx, 'y', 'z', gx_int, 'y', 'z');
                    break;
                case 7: 
                    printf("[%s] global=(%c,%c,%c) -> interior=(%c,%c,%c)\n",
                        label, 'x', 'y', 'z', 'x', 'y', 'z');
                    break;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_SOUTH_mxy_I(
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
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_SOUTH_mxy_I(
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
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_NORTH_mxy_I(
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
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_NORTH_mxy_I(
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