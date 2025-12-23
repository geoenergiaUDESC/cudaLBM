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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Definition of the D3Q7 velocity set

Namespace
    LBM

SourceFiles
    D3Q7.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_D3Q7_CUH
#define __MBLBM_D3Q7_CUH

#include "velocitySet.cuh"

namespace LBM
{
    /**
     * @class D3Q7
     * @brief Implements the D3Q7 velocity set for 3D Lattice Boltzmann simulations
     * @extends velocitySet
     *
     * This class provides the specific implementation for the D3Q7 lattice model,
     * which includes 7 discrete velocity directions in 3D space. It contains:
     * - Velocity components (cx, cy, cz) for each direction
     * - Weight coefficients for each direction
     * - Methods for moment calculation and population reconstruction
     * - Equilibrium distribution functions
     **/
    class D3Q7 : private velocitySet
    {
    public:
        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval D3Q7(){};

        __device__ __host__ [[nodiscard]] static inline consteval bool isPhaseField() noexcept
        {
            return true;
        }

        /**
         * @brief Get number of discrete velocity directions
         * @return 7 (number of directions in D3Q7 lattice)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return Q_;
        }

        /**
         * @brief Get number of velocity components on a lattice face
         * @return 5 (number of directions crossing each face in D3Q7)
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
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(4));
        }

        /**
         * @brief Get weight for orthogonal directions (q=1-6)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
        {
            return static_cast<T>(static_cast<double>(1) / static_cast<double>(8));
        }

        /**
         * @brief Get all weights for host computation
         * @return Array of 7 weights in D3Q7 order
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 7> host_w_q() noexcept
        {
            // Return the component
            return {
                w_0<T>(),
                w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>()};
        }

        /**
         * @brief Get all weights for device computation
         * @return Thread array of 7 weights in D3Q7 order
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 7> w_q() noexcept
        {
            // Return the component
            return {
                w_0<T>(),
                w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>()};
        }

        /**
         * @brief Get weight for specific direction
         * @tparam q_ Direction index (0-6)
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
         * @return Array of 7 x-velocity components
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 7> host_cx() noexcept
        {
            // Return the component
            return {0, 1, -1, 0, 0, 0, 0};
        }

        /**
         * @brief Get x-components for all directions (device version)
         * @return Thread array of 7 x-velocity components
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 7> cx() noexcept
        {
            // Return the component
            return {0, 1, -1, 0, 0, 0, 0};
        }

        /**
         * @brief Get x-component for specific direction
         * @tparam q_ Direction index (0-6)
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
         * @tparam q_ Direction index (0-6)
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
         * @tparam q_ Direction index (0-6)
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
         * @return Array of 7 y-velocity components
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 7> host_cy() noexcept
        {
            // Return the component
            return {0, 0, 0, 1, -1, 0, 0};
        }

        /**
         * @brief Get y-components for all directions (device version)
         * @return Thread array of 7 y-velocity components
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 7> cy() noexcept
        {
            // Return the component
            return {0, 0, 0, 1, -1, 0, 0};
        }

        /**
         * @brief Get y-component for specific direction
         * @tparam q_ Direction index (0-6)
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
         * @tparam q_ Direction index (0-6)
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
         * @tparam q_ Direction index (0-6)
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
         * @return Array of 7 z-velocity components
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline consteval const std::array<T, 7> host_cz() noexcept
        {
            // Return the component
            return {0, 0, 0, 0, 0, 1, -1};
        }

        /**
         * @brief Get z-components for all directions (device version)
         * @return Thread array of 7 z-velocity components
         **/
        template <typename T>
        __device__ [[nodiscard]] static inline consteval const thread::array<T, 7> cz() noexcept
        {
            // Return the component
            return {0, 0, 0, 0, 0, 1, -1};
        }

        /**
         * @brief Get z-component for specific direction
         * @tparam q_ Direction index (0-6)
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
         * @tparam q_ Direction index (0-6)
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
         * @tparam q_ Direction index (0-6)
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
         * @param[in] phiw Weighted density (w_q[q] * phi)
         * @param[in] uc4 4 * (u·c_q) = 4*(u*cx + v*cy + w*cz)
         * @return First-order equilibrium distribution value for the direction
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline constexpr T f_eq(const T phiw, const T uc4) noexcept
        {
            return (phiw * (static_cast<scalar_t>(1) + uc4));
        }

        /**
         * @brief Calculate full equilibrium distribution for given velocity
         * @tparam T Data type for calculation
         * @param[in] u x-component of velocity
         * @param[in] v y-component of velocity
         * @param[in] w z-component of velocity
         * @return Array of 7 second-order equilibrium distribution values
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline constexpr const std::array<T, 7> F_eq(const T u, const T v, const T w) noexcept
        {
            std::array<T, Q_> pop;

            for (label_t q = 0; q < Q_; q++)
            {
                pop[q] = f_eq<T>(
                    host_w_q<T>()[q],
                    static_cast<T>(4) * ((u * host_cx<T>()[q]) + (v * host_cy<T>()[q]) + (w * host_cz<T>()[q])));
            }

            return pop;
        }

        /**
         * @brief Reconstruct population distribution from moments (in-place)
         * @param[out] pop Population array to be filled
         * @param[in] moments Moment array (11 components)
         **/
        __device__ static inline void reconstruct(thread::array<scalar_t, 7> &pop, const thread::array<scalar_t, 11> &moments, const scalar_t normx, const scalar_t normy, const scalar_t normz) noexcept
        {
            const scalar_t phiw_0 = moments[m_i<10>()] * w_0<scalar_t>();
            const scalar_t phiw_1 = moments[m_i<10>()] * w_1<scalar_t>();

            pop[q_i<0>()] = phiw_0;

            scalar_t anti_diff = w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normx;
            pop[q_i<1>()] = phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<1>()]) + anti_diff;
            pop[q_i<2>()] = phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<1>()]) - anti_diff;

            anti_diff = w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normy;
            pop[q_i<3>()] = phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<2>()]) + anti_diff;
            pop[q_i<4>()] = phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<2>()]) - anti_diff;

            anti_diff = w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normz;
            pop[q_i<5>()] = phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<3>()]) + anti_diff;
            pop[q_i<6>()] = phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<3>()]) - anti_diff;
        }

        /**
         * @brief Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (11 components)
         * @return Population array with 7 components
         **/
        __device__ static inline thread::array<scalar_t, 7> reconstruct(const thread::array<scalar_t, 11> &moments, const scalar_t normx, const scalar_t normy, const scalar_t normz) noexcept
        {
            const scalar_t phiw_0 = moments[m_i<10>()] * w_0<scalar_t>();
            const scalar_t phiw_1 = moments[m_i<10>()] * w_1<scalar_t>();

            return {
                phiw_0,
                phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<1>()]) + w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normx,
                phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<1>()]) - w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normx,
                phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<2>()]) + w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normy,
                phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<2>()]) - w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normy,
                phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<3>()]) + w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normz,
                phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<3>()]) - w_1<scalar_t>() * device::gamma * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * normz};
        }

        /**
         * @brief Reconstruct population distribution from moments (host version)
         * @param[in] moments Moment array (11 components)
         * @return Population array with 7 components
         **/
        __host__ [[nodiscard]] static const std::array<scalar_t, 7> reconstruct(const std::array<scalar_t, 14> &moments) noexcept
        {
            std::array<scalar_t, 7> pop;

            const scalar_t phiw_0 = moments[m_i<10>()] * w_0<scalar_t>();
            const scalar_t phiw_1 = moments[m_i<10>()] * w_1<scalar_t>();

            pop[q_i<0>()] = phiw_0;

            scalar_t anti_diff = w_1<scalar_t>() * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * moments[m_i<11>()];
            pop[q_i<1>()] = phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<1>()]) + anti_diff;
            pop[q_i<2>()] = phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<1>()]) - anti_diff;

            anti_diff = w_1<scalar_t>() * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * moments[m_i<12>()];
            pop[q_i<3>()] = phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<2>()]) + anti_diff;
            pop[q_i<4>()] = phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<2>()]) - anti_diff;

            anti_diff = w_1<scalar_t>() * moments[m_i<10>()] * (static_cast<scalar_t>(1) - moments[m_i<10>()]) * moments[m_i<13>()];
            pop[q_i<5>()] = phiw_1 * (static_cast<scalar_t>(1) + moments[m_i<3>()]) + anti_diff;
            pop[q_i<6>()] = phiw_1 * (static_cast<scalar_t>(1) - moments[m_i<3>()]) - anti_diff;

            return pop;
        }

        /**
         * @brief Calculate specific moment from population distribution (host version)
         * @tparam moment_ Moment index to calculate (0-9)
         * @param[in] pop Population array (7 components)
         * @return Calculated moment value
         **/
        template <const label_t moment_>
        __host__ [[nodiscard]] inline static constexpr scalar_t calculateMoment(const std::array<scalar_t, 7> &pop) noexcept
        {
            if constexpr (moment_ == 10)
            {
                return pop[q_i<0>()] + pop[q_i<1>()] + pop[q_i<2>()] + pop[q_i<3>()] + pop[q_i<4>()] + pop[q_i<5>()] + pop[q_i<6>()];
            }
            else
            {
                return static_cast<scalar_t>(0);
            }
        }

        /**
         * @brief Calculate phi from population distribution
         * @param[in] pop Population array (7 components)
         * @param[out] moments Moment array to be filled (11 components)
         **/
        __device__ inline static void calculatePhi(const thread::array<scalar_t, 7> &pop, thread::array<scalar_t, 11> &moments) noexcept
        {
            moments[m_i<10>()] = pop.sum();
        }

        /**
         * @brief Print velocity set information to terminal
         **/
        __host__ static void print() noexcept
        {
            std::cout << "D3Q7 {w, cx, cy, cz}:" << std::endl;
            std::cout << "{" << std::endl;
            printAll();
            std::cout << "};" << std::endl;
            std::cout << std::endl;
        }

    private:
        /**
         * @brief Number of velocity components in the lattice
         **/
        static constexpr const label_t Q_ = 7;

        /**
         * @brief Number of velocity components on each lattice face
         **/
        static constexpr const label_t QF_ = 1;

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