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
    along with this program. If not, see <https://www.gnu.org/licenses/>.

Description
    Definition of the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    D3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_D3Q19_CUH
#define __MBLBM_D3Q19_CUH

#include "velocitySet.cuh"

namespace LBM
{
    /**
     * @class D3Q19
     * @brief Implements the D3Q19 velocity set for 3D Lattice Boltzmann simulations
     * @extends velocitySet
     *
     * This class provides the specific implementation for the D3Q19 lattice model,
     * which includes 19 discrete velocity directions in 3D space. It contains:
     * - Velocity components (cx, cy, cz) for each direction
     * - Weight coefficients for each direction
     * - Methods for moment calculation and population reconstruction
     * - Equilibrium distribution functions
     **/
    class D3Q19 : private velocitySet
    {
    public:
        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval D3Q19(){};

        __device__ __host__ [[nodiscard]] static inline consteval bool isPhaseField() noexcept
        {
            return false;
        }

        /**
         * @brief Get number of discrete velocity directions
         * @return 19 (number of directions in D3Q19 lattice)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return Q_;
        }

        /**
         * @brief Get number of velocity components on a lattice face
         * @return 5 (number of directions crossing each face in D3Q19)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t QF() noexcept
        {
            return QF_;
        }

        /**
         * @brief Get weight for stationary component (q=0)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_0() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(3));
        }

        /**
         * @brief Get weight for orthogonal directions (q=1-6)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(18));
        }

        /**
         * @brief Get weight for diagonal directions (q=7-18)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_2() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(36));
        }

        /**
         * @brief Get all weights for host computation
         * @return Array of 19 weights in D3Q19 order
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 19> host_w_q() noexcept
        {
            // Return the component
            return {
                w_0<T>(),
                w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(),
                w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>()};
        }

        /**
         * @brief Get all weights for device computation
         * @return Thread array of 19 weights in D3Q19 order
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> w_q() noexcept
        {
            // Return the component
            return {
                w_0<T>(),
                w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(),
                w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>()};
        }

        /**
         * @brief Get weight for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return Weight for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ [[nodiscard]] static inline consteval T w_q(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function w(q)");

            // Return the component
            return w_q<T>()[q()];
        }

        /**
         * @brief Get x-components for all directions (host version)
         * @return Array of 19 x-velocity components
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 19> host_cx() noexcept
        {
            // Return the component
            return {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
        }

        /**
         * @brief Get x-components for all directions (device version)
         * @return Thread array of 19 x-velocity components
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> cx() noexcept
        {
            // Return the component
            return {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
        }

        /**
         * @brief Get x-component for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return x-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ [[nodiscard]] static inline consteval T cx(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function cx(q)");

            // Return the component
            return cx<T>()[q()];
        }

        /**
         * @brief Check if x-component is negative for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return True if x-component is negative
         **/
        template <const label_t q_>
        __device__ [[nodiscard]] static inline consteval bool nxNeg(const q_i<q_> q) noexcept
        {
            return (cx<int>(q) < 0);
        }

        /**
         * @brief Check if x-component is positive for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return True if x-component is positive
         **/
        template <const label_t q_>
        __device__ [[nodiscard]] static inline consteval bool nxPos(const q_i<q_> q) noexcept
        {
            return (cx<int>(q) > 0);
        }

        /**
         * @brief Get y-components for all directions (host version)
         * @return Array of 19 y-velocity components
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 19> host_cy() noexcept
        {
            // Return the component
            return {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
        }

        /**
         * @brief Get y-components for all directions (device version)
         * @return Thread array of 19 y-velocity components
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> cy() noexcept
        {
            // Return the component
            return {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
        }

        /**
         * @brief Get y-component for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return y-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ [[nodiscard]] static inline consteval T cy(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function cy(q)");

            // Return the component
            return cy<T>()[q()];
        }

        /**
         * @brief Check if y-component is negative for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return True if y-component is negative
         **/
        template <const label_t q_>
        __device__ [[nodiscard]] static inline consteval bool nyNeg(const q_i<q_> q) noexcept
        {
            return (cy<int>(q) < 0);
        }

        /**
         * @brief Check if x-component is positive for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return True if x-component is positive
         **/
        template <const label_t q_>
        __device__ [[nodiscard]] static inline consteval bool nyPos(const q_i<q_> q) noexcept
        {
            return (cy<int>(q) > 0);
        }

        /**
         * @brief Get z-components for all directions (host version)
         * @return Array of 19 z-velocity components
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 19> host_cz() noexcept
        {
            // Return the component
            return {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
        }

        /**
         * @brief Get z-components for all directions (device version)
         * @return Thread array of 19 z-velocity components
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> cz() noexcept
        {
            // Return the component
            return {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
        }

        /**
         * @brief Get z-component for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return z-component for specified direction
         **/
        template <typename T, const label_t q_>
        __device__ [[nodiscard]] static inline consteval T cz(const q_i<q_> q) noexcept
        {
            // Check that we are accessing a valid member
            static_assert(q() < Q_, "Invalid velocity set index in member function cz(q)");

            // Return the component
            return cz<T>()[q()];
        }

        /**
         * @brief Check if z-component is negative for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return True if z-component is positive
         **/
        template <const label_t q_>
        __device__ [[nodiscard]] static inline consteval bool nzNeg(const q_i<q_> q) noexcept
        {
            return (cz<int>(q) < 0);
        }

        /**
         * @brief Check if z-component is positive for specific direction
         * @tparam q_ Direction index (0-18)
         * @param[in] q Direction index as compile-time constant
         * @return True if z-component is positive
         **/
        template <const label_t q_>
        __device__ [[nodiscard]] static inline consteval bool nzPos(const q_i<q_> q) noexcept
        {
            return (cz<int>(q) > 0);
        }

        /**
         * @brief Calculate equilibrium distribution function for a direction
         * @tparam T Data type for calculation
         * @param[in] rhow Weighted density (w_q[q] * rho)
         * @param[in] uc3 3 * (u·c_q) = 3*(u*cx + v*cy + w*cz)
         * @param[in] p1_muu 1 - 1.5*(u² + v² + w²)
         * @return Second-order equilibrium distribution value for the direction
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline constexpr T f_eq(const T rhow, const T uc3, const T p1_muu) noexcept
        {
            return (rhow * (p1_muu + uc3 * (static_cast<T>(1.0) + uc3 * static_cast<T>(0.5))));
        }

        /**
         * @brief Calculate full equilibrium distribution for given velocity
         * @tparam T Data type for calculation
         * @param[in] u x-component of velocity
         * @param[in] v y-component of velocity
         * @param[in] w z-component of velocity
         * @return Array of 19 second-order equilibrium distribution values
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline constexpr const std::array<T, 19> F_eq(const T u, const T v, const T w) noexcept
        {
            std::array<T, Q_> pop;

            for (label_t q = 0; q < Q_; q++)
            {
                pop[q] = f_eq<T>(
                    host_w_q<T>()[q],
                    static_cast<T>(3) * ((u * host_cx<T>()[q]) + (v * host_cy<T>()[q]) + (w * host_cz<T>()[q])),
                    static_cast<T>(1) - static_cast<T>(1.5) * ((u * u) + (v * v) + (w * w)));
            }

            return pop;
        }

        /**
         * @brief Reconstruct population distribution from moments (in-place)
         * @param[out] pop Population array to be filled
         * @param[in] moments Moment array (10 components)
         **/
        __device__ static inline void reconstruct(thread::array<scalar_t, 19> &pop, const thread::array<scalar_t, 10> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            pop[q_i<0>()] = rhow_0 * pics2;

            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
            pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
            pop[q_i<7>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<8>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<9>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<10>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<11>()] = rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<12>()] = rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<13>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<14>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<15>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<16>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<17>()] = rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
            pop[q_i<18>()] = rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
        }

        /**
         * @overload Reconstruct population distribution from moments (in-place)
         * @param[out] pop Population array to be filled
         * @param[in] moments Moment array (11 components)
         **/
        __device__ static inline void reconstruct(thread::array<scalar_t, 19> &pop, const thread::array<scalar_t, 11> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            pop[q_i<0>()] = rhow_0 * pics2;

            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
            pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
            pop[q_i<7>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<8>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<9>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<10>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<11>()] = rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<12>()] = rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<13>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<14>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<15>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<16>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<17>()] = rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
            pop[q_i<18>()] = rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
        }

        /**
         * @brief Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (10 components)
         * @return Population array with 19 components
         **/
        __device__ static inline thread::array<scalar_t, 19> reconstruct(const thread::array<scalar_t, 10> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();

            return {
                rhow_0 * pics2,
                rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]),
                rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]),
                rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]),
                rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]),
                rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]),
                rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]),
                rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]),
                rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]),
                rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]),
                rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()])};
        }

        /**
         * @overload Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (11 components)
         * @return Population array with 19 components
         **/
        __device__ static inline thread::array<scalar_t, 19> reconstruct(const thread::array<scalar_t, 11> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();

            return {
                rhow_0 * pics2,
                rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]),
                rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]),
                rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]),
                rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]),
                rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]),
                rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]),
                rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]),
                rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]),
                rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]),
                rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]),
                rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]),
                rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()])};
        }

        /**
         * @brief Reconstruct population distribution from moments (host version)
         * @param[in] moments Moment array (10 components)
         * @return Population array with 19 components
         **/
        __host__ [[nodiscard]] static const std::array<scalar_t, 19> reconstruct(const std::array<scalar_t, 10> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();

            std::array<scalar_t, 19> pop;

            pop[q_i<0>()] = rhow_0 * pics2;

            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
            pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
            pop[q_i<7>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<8>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<9>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<10>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<11>()] = rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<12>()] = rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<13>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<14>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<15>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<16>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<17>()] = rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
            pop[q_i<18>()] = rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);

            return pop;
        }

        /**
         * @overload Reconstruct population distribution from moments (host version)
         * @param[in] moments Moment array (11 components)
         * @return Population array with 19 components
         **/
        __host__ [[nodiscard]] static const std::array<scalar_t, 19> reconstruct(const std::array<scalar_t, 11> &moments) noexcept
        {
            const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

            const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();

            std::array<scalar_t, 19> pop;

            pop[q_i<0>()] = rhow_0 * pics2;

            const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
            pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
            pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
            pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
            pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

            const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
            pop[q_i<7>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<8>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
            pop[q_i<9>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<10>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
            pop[q_i<11>()] = rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<12>()] = rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
            pop[q_i<13>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<14>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
            pop[q_i<15>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<16>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
            pop[q_i<17>()] = rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
            pop[q_i<18>()] = rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);

            return pop;
        }

        /**
         * @brief Calculate specific moment from population distribution (host version)
         * @tparam moment_ Moment index to calculate (0-9)
         * @param[in] pop Population array (19 components)
         * @return Calculated moment value
         **/
        template <const label_t moment_>
        __host__ [[nodiscard]] inline static constexpr scalar_t calculateMoment(const std::array<scalar_t, 19> &pop) noexcept
        {
            if constexpr (moment_ == 0)
            {
                return pop[q_i<0>()] + pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<5>()] + pop[q_i<6>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<9>()] + pop[q_i<10>()] + pop[q_i<11>()] + pop[q_i<12>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<15>()] + pop[q_i<16>()] + pop[q_i<17>()] + pop[q_i<18>()];
            }
            else if constexpr (moment_ == 1)
            {
                return pop[q_i<1>()] - pop[q_i<2>()] + pop[q_i<7>()] - pop[q_i<8>()] + pop[q_i<9>()] - pop[q_i<10>()] + pop[q_i<13>()] - pop[q_i<14>()] + pop[q_i<15>()] - pop[q_i<16>()];
            }
            else if constexpr (moment_ == 2)
            {
                return pop[q_i<3>()] - pop[q_i<4>()] + pop[q_i<7>()] - pop[q_i<8>()] + pop[q_i<11>()] - pop[q_i<12>()] + pop[q_i<14>()] - pop[q_i<13>()] + pop[q_i<17>()] - pop[q_i<18>()];
            }
            else if constexpr (moment_ == 3)
            {
                return pop[q_i<5>()] - pop[q_i<6>()] + pop[q_i<9>()] - pop[q_i<10>()] + pop[q_i<11>()] - pop[q_i<12>()] + pop[q_i<16>()] - pop[q_i<15>()] + pop[q_i<18>()] - pop[q_i<17>()];
            }
            else if constexpr (moment_ == 4)
            {
                return pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<9>()] + pop[q_i<10>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<15>()] + pop[q_i<16>()];
            }
            else if constexpr (moment_ == 5)
            {
                return pop[q_i<7>()] - pop[q_i<13>()] + pop[q_i<8>()] - pop[q_i<14>()];
            }
            else if constexpr (moment_ == 6)
            {
                return pop[q_i<9>()] - pop[q_i<15>()] + pop[q_i<10>()] - pop[q_i<16>()];
            }
            else if constexpr (moment_ == 7)
            {
                return pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<11>()] + pop[q_i<12>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<17>()] + pop[q_i<18>()];
            }
            else if constexpr (moment_ == 8)
            {
                return pop[q_i<11>()] - pop[q_i<17>()] + pop[q_i<12>()] - pop[q_i<18>()];
            }
            else if constexpr (moment_ == 9)
            {
                return pop[q_i<5>()] + pop[q_i<6>()] + pop[q_i<9>()] + pop[q_i<10>()] + pop[q_i<11>()] + pop[q_i<12>()] + pop[q_i<15>()] + pop[q_i<16>()] + pop[q_i<17>()] + pop[q_i<18>()];
            }
            else
            {
                return static_cast<scalar_t>(0);
            }
        }

        /**
         * @brief Calculate moments from population distribution
         * @param[in] pop Population array (19 components)
         * @param[out] moments Moment array to be filled (10 components)
         **/
        __device__ inline static void calculateMoments(const thread::array<scalar_t, 19> &pop, thread::array<scalar_t, 10> &moments) noexcept
        {
            // Equation 3
            moments[m_i<0>()] = pop.sum();
            const scalar_t invRho = static_cast<scalar_t>(1) / moments[m_i<0>()];

            // Equation 4 + force correction
            moments[m_i<1>()] = ((pop[q_i<1>()] - pop[q_i<2>()] + pop[q_i<7>()] - pop[q_i<8>()] + pop[q_i<9>()] - pop[q_i<10>()] + pop[q_i<13>()] - pop[q_i<14>()] + pop[q_i<15>()] - pop[q_i<16>()])) * invRho;
            moments[m_i<2>()] = ((pop[q_i<3>()] - pop[q_i<4>()] + pop[q_i<7>()] - pop[q_i<8>()] + pop[q_i<11>()] - pop[q_i<12>()] + pop[q_i<14>()] - pop[q_i<13>()] + pop[q_i<17>()] - pop[q_i<18>()])) * invRho;
            moments[m_i<3>()] = ((pop[q_i<5>()] - pop[q_i<6>()] + pop[q_i<9>()] - pop[q_i<10>()] + pop[q_i<11>()] - pop[q_i<12>()] + pop[q_i<16>()] - pop[q_i<15>()] + pop[q_i<18>()] - pop[q_i<17>()])) * invRho;

            // Equation 5
            moments[m_i<4>()] = (pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<9>()] + pop[q_i<10>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<15>()] + pop[q_i<16>()]) * invRho - cs2<scalar_t>();
            moments[m_i<5>()] = (pop[q_i<7>()] - pop[q_i<13>()] + pop[q_i<8>()] - pop[q_i<14>()]) * invRho;
            moments[m_i<6>()] = (pop[q_i<9>()] - pop[q_i<15>()] + pop[q_i<10>()] - pop[q_i<16>()]) * invRho;
            moments[m_i<7>()] = (pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<11>()] + pop[q_i<12>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<17>()] + pop[q_i<18>()]) * invRho - cs2<scalar_t>();
            moments[m_i<8>()] = (pop[q_i<11>()] - pop[q_i<17>()] + pop[q_i<12>()] - pop[q_i<18>()]) * invRho;
            moments[m_i<9>()] = (pop[q_i<5>()] + pop[q_i<6>()] + pop[q_i<9>()] + pop[q_i<10>()] + pop[q_i<11>()] + pop[q_i<12>()] + pop[q_i<15>()] + pop[q_i<16>()] + pop[q_i<17>()] + pop[q_i<18>()]) * invRho - cs2<scalar_t>();
        }

        /**
         * @overload Calculate moments from population distribution
         * @param[in] pop Population array (19 components)
         * @param[out] moments Moment array to be filled (11 components)
         **/
        __device__ inline static void calculateMoments(const thread::array<scalar_t, 19> &pop, thread::array<scalar_t, 11> &moments) noexcept
        {
            // Density
            moments[m_i<0>()] = pop.sum();
            const scalar_t invRho = static_cast<scalar_t>(1) / moments[m_i<0>()];

            // Velocities
            moments[m_i<1>()] = ((pop[q_i<1>()] - pop[q_i<2>()] + pop[q_i<7>()] - pop[q_i<8>()] + pop[q_i<9>()] - pop[q_i<10>()] + pop[q_i<13>()] - pop[q_i<14>()] + pop[q_i<15>()] - pop[q_i<16>()])) * invRho;
            moments[m_i<2>()] = ((pop[q_i<3>()] - pop[q_i<4>()] + pop[q_i<7>()] - pop[q_i<8>()] + pop[q_i<11>()] - pop[q_i<12>()] + pop[q_i<14>()] - pop[q_i<13>()] + pop[q_i<17>()] - pop[q_i<18>()])) * invRho;
            moments[m_i<3>()] = ((pop[q_i<5>()] - pop[q_i<6>()] + pop[q_i<9>()] - pop[q_i<10>()] + pop[q_i<11>()] - pop[q_i<12>()] + pop[q_i<16>()] - pop[q_i<15>()] + pop[q_i<18>()] - pop[q_i<17>()])) * invRho;

            // Higher order moments
            moments[m_i<4>()] = (pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<9>()] + pop[q_i<10>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<15>()] + pop[q_i<16>()]) * invRho - cs2<scalar_t>();
            moments[m_i<5>()] = (pop[q_i<7>()] - pop[q_i<13>()] + pop[q_i<8>()] - pop[q_i<14>()]) * invRho;
            moments[m_i<6>()] = (pop[q_i<9>()] - pop[q_i<15>()] + pop[q_i<10>()] - pop[q_i<16>()]) * invRho;
            moments[m_i<7>()] = (pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<7>()] + pop[q_i<8>()] + pop[q_i<11>()] + pop[q_i<12>()] + pop[q_i<13>()] + pop[q_i<14>()] + pop[q_i<17>()] + pop[q_i<18>()]) * invRho - cs2<scalar_t>();
            moments[m_i<8>()] = (pop[q_i<11>()] - pop[q_i<17>()] + pop[q_i<12>()] - pop[q_i<18>()]) * invRho;
            moments[m_i<9>()] = (pop[q_i<5>()] + pop[q_i<6>()] + pop[q_i<9>()] + pop[q_i<10>()] + pop[q_i<11>()] + pop[q_i<12>()] + pop[q_i<15>()] + pop[q_i<16>()] + pop[q_i<17>()] + pop[q_i<18>()]) * invRho - cs2<scalar_t>();
        }

        /**
         * @brief Print velocity set information to terminal
         **/
        __host__ static void print() noexcept
        {
            std::cout << "D3Q19 {w, cx, cy, cz}:" << std::endl;
            std::cout << "{" << std::endl;
            printAll();
            std::cout << "};" << std::endl;
            std::cout << std::endl;
        }

    private:
        /**
         * @brief Number of velocity components in the lattice
         **/
        static constexpr const label_t Q_ = 19;

        /**
         * @brief Number of velocity components on each lattice face
         **/
        static constexpr const label_t QF_ = 5;

        /**
         * @brief Implementation of the print loop
         * @note This function effectively unrolls the loop at compile-time and checks for its bounds
         **/
        template <const label_t q_ = 0>
        __host__ static inline void printAll(const q_i<q_> q = q_i<0>()) noexcept
        {
            // Loop over the velocity set, print to terminal
            host::constexpr_for<q(), Q()>(
                [&](const auto Q)
                {
                    std::cout
                        << "    [" << q_i<Q>() << "] = {"
                        << host_w_q<double>()[q_i<Q>()] << ", "
                        << host_cx<int>()[q_i<Q>()] << ", "
                        << host_cy<int>()[q_i<Q>()] << ", "
                        << host_cz<int>()[q_i<Q>()] << "};" << std::endl;
                });
        }
    };
}

#endif