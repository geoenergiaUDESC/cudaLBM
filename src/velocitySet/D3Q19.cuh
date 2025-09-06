/**
Filename: D3Q19.cuh
Contents: Definition of the D3Q19 velocity set
**/

#ifndef __MBLBM_D3Q19_CUH
#define __MBLBM_D3Q19_CUH

#include "velocitySet.cuh"

namespace LBM
{
    namespace VelocitySet
    {
        class D3Q19 : private velocitySet
        {
        public:
            /**
             * @brief Constructor for the D3Q19 velocity set
             * @return A D3Q19 velocity set
             * @note This constructor is consteval
             **/
            [[nodiscard]] inline consteval D3Q19() {};

            /**
             * @brief Number of velocity components
             * @return 19
             **/
            [[nodiscard]] static inline consteval label_t Q() noexcept
            {
                return Q_;
            }

            /**
             * @brief Number of velocity components on a lattice face
             * @return 5
             **/
            [[nodiscard]] static inline consteval label_t QF() noexcept
            {
                return QF_;
            }

            /**
             * @brief Returns the unique lattice weights
             * @return The unique lattice weights for the D3Q19 velocity set
             **/
            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T w_0() noexcept
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(3));
            }
            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(18));
            }
            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T w_2() noexcept
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(36));
            }

            /**
             * @brief Returns the weight for a given lattice point
             * @return w_q[q]
             * @param q The lattice point
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
            template <typename T>
            __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> w_q() noexcept
            {
                // Return the component
                return {
                    w_0<T>(),
                    w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(),
                    w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>()};
            }
            template <typename T, const label_t q_>
            __device__ [[nodiscard]] static inline consteval T w_q(const label_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function w(q)");

                // Return the component
                return w_q<T>()[q()];
            }

            /**
             * @brief Returns the x lattice coefficient for a given lattice point
             * @return c_x[q]
             * @param q The lattice point
             **/
            template <typename T>
            __host__ [[nodiscard]] static inline consteval const std::array<T, 19> host_cx() noexcept
            {
                // Return the component
                return {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
            }
            template <typename T>
            __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> cx() noexcept
            {
                // Return the component
                return {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
            }
            template <typename T, const label_t q_>
            __device__ [[nodiscard]] static inline consteval T cx(const label_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cx(q)");

                // Return the component
                return cx<T>()[q()];
            }
            template <const label_t q_>
            __device__ [[nodiscard]] static inline consteval bool nxNeg(const label_constant<q_> q) noexcept
            {
                return (cx<int>(q) < 0);
            }
            template <const label_t q_>
            __device__ [[nodiscard]] static inline consteval bool nxPos(const label_constant<q_> q) noexcept
            {
                return (cx<int>(q) > 0);
            }

            /**
             * @brief Returns the y lattice coefficient for a given lattice point
             * @return c_y[q]
             * @param q The lattice point
             **/
            template <typename T>
            __host__ [[nodiscard]] static inline consteval const std::array<T, 19> host_cy() noexcept
            {
                // Return the component
                return {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
            }
            template <typename T>
            __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> cy() noexcept
            {
                // Return the component
                return {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
            }
            template <typename T, const label_t q_>
            __device__ [[nodiscard]] static inline consteval T cy(const label_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cy(q)");

                // Return the component
                return cy<T>()[q()];
            }
            template <const label_t q_>
            __device__ [[nodiscard]] static inline consteval bool nyNeg(const label_constant<q_> q) noexcept
            {
                return (cy<int>(q) < 0);
            }
            template <const label_t q_>
            __device__ [[nodiscard]] static inline consteval bool nyPos(const label_constant<q_> q) noexcept
            {
                return (cy<int>(q) > 0);
            }

            /**
             * @brief Returns the z lattice coefficient for a given lattice point
             * @return c_z[q]
             * @param q The lattice point
             **/
            template <typename T>
            __host__ [[nodiscard]] static inline consteval const std::array<T, 19> host_cz() noexcept
            {
                // Return the component
                return {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
            }
            template <typename T>
            __device__ [[nodiscard]] static inline consteval const thread::array<T, 19> cz() noexcept
            {
                // Return the component
                return {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
            }
            template <typename T, const label_t q_>
            __device__ [[nodiscard]] static inline consteval T cz(const label_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cz(q)");

                // Return the component
                return cz<T>()[q()];
            }
            template <const label_t q_>
            __device__ [[nodiscard]] static inline consteval bool nzNeg(const label_constant<q_> q) noexcept
            {
                return (cz<int>(q) < 0);
            }
            template <const label_t q_>
            __device__ [[nodiscard]] static inline consteval bool nzPos(const label_constant<q_> q) noexcept
            {
                return (cz<int>(q) > 0);
            }

            /**
             * @brief Returns the equilibrium distribution for a given lattice index
             * @return f_eq[q]
             * @param rhow w_q[q] * rho
             * @param uc3 3 * ((u * cx[q]) + (v * cy[q]) + (w * [cz]))
             * @param p1_muu 1 - 1.5 * ((u ^ 2) + (v ^ 2) + (w ^ 2))
             **/
            template <typename T>
            [[nodiscard]] static inline constexpr T f_eq(const T rhow, const T uc3, const T p1_muu) noexcept
            {
                return (rhow * (p1_muu + uc3 * (static_cast<T>(1.0) + uc3 * static_cast<T>(0.5))));
            }

            /**
             * @brief Returns the population density
             * @return f_eq
             * @param u The x-component of velocity
             * @param v The y-component of velocity
             * @param w The z-component of velocity
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
             * @brief Reconstructs the population at a given lattice point
             * @param pop The reconstructed population
             * @param moments The moments from which the population is to be reconstructed
             **/
            __device__ static inline void reconstruct(thread::array<scalar_t, 19> &pop, const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
            {
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments(label_constant<4>()) + moments(label_constant<7>()) + moments(label_constant<9>()));

                const scalar_t multiplyTerm_0 = moments(label_constant<0>()) * w_0<scalar_t>();
                pop(label_constant<0>()) = multiplyTerm_0 * pics2;

                const scalar_t multiplyTerm_1 = moments(label_constant<0>()) * w_1<scalar_t>();
                pop(label_constant<1>()) = multiplyTerm_1 * (pics2 + moments(label_constant<1>()) + moments(label_constant<4>()));
                pop(label_constant<2>()) = multiplyTerm_1 * (pics2 - moments(label_constant<1>()) + moments(label_constant<4>()));
                pop(label_constant<3>()) = multiplyTerm_1 * (pics2 + moments(label_constant<2>()) + moments(label_constant<7>()));
                pop(label_constant<4>()) = multiplyTerm_1 * (pics2 - moments(label_constant<2>()) + moments(label_constant<7>()));
                pop(label_constant<5>()) = multiplyTerm_1 * (pics2 + moments(label_constant<3>()) + moments(label_constant<9>()));
                pop(label_constant<6>()) = multiplyTerm_1 * (pics2 - moments(label_constant<3>()) + moments(label_constant<9>()));

                const scalar_t multiplyTerm_2 = moments(label_constant<0>()) * w_2<scalar_t>();
                pop(label_constant<7>()) = multiplyTerm_2 * (pics2 + moments(label_constant<1>()) + moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) + moments(label_constant<5>()));
                pop(label_constant<8>()) = multiplyTerm_2 * (pics2 - moments(label_constant<1>()) - moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) + moments(label_constant<5>()));
                pop(label_constant<9>()) = multiplyTerm_2 * (pics2 + moments(label_constant<1>()) + moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) + moments(label_constant<6>()));
                pop(label_constant<10>()) = multiplyTerm_2 * (pics2 - moments(label_constant<1>()) - moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) + moments(label_constant<6>()));
                pop(label_constant<11>()) = multiplyTerm_2 * (pics2 + moments(label_constant<2>()) + moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) + moments(label_constant<8>()));
                pop(label_constant<12>()) = multiplyTerm_2 * (pics2 - moments(label_constant<2>()) - moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) + moments(label_constant<8>()));
                pop(label_constant<13>()) = multiplyTerm_2 * (pics2 + moments(label_constant<1>()) - moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) - moments(label_constant<5>()));
                pop(label_constant<14>()) = multiplyTerm_2 * (pics2 - moments(label_constant<1>()) + moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) - moments(label_constant<5>()));
                pop(label_constant<15>()) = multiplyTerm_2 * (pics2 + moments(label_constant<1>()) - moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) - moments(label_constant<6>()));
                pop(label_constant<16>()) = multiplyTerm_2 * (pics2 - moments(label_constant<1>()) + moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) - moments(label_constant<6>()));
                pop(label_constant<17>()) = multiplyTerm_2 * (pics2 + moments(label_constant<2>()) - moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) - moments(label_constant<8>()));
                pop(label_constant<18>()) = multiplyTerm_2 * (pics2 - moments(label_constant<2>()) + moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) - moments(label_constant<8>()));
            }

            /**
             * @brief Reconstructs the population at a given lattice point
             * @return The reconstructed population
             * @param moments The moments from which the population is to be reconstructed
             **/
            __device__ static inline thread::array<scalar_t, 19> reconstruct(const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
            {
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments(label_constant<4>()) + moments(label_constant<7>()) + moments(label_constant<9>()));

                const scalar_t multiplyTerm_0 = moments(label_constant<0>()) * w_0<scalar_t>();
                const scalar_t multiplyTerm_1 = moments(label_constant<0>()) * w_1<scalar_t>();
                const scalar_t multiplyTerm_2 = moments(label_constant<0>()) * w_2<scalar_t>();

                return {
                    multiplyTerm_0 * pics2,
                    multiplyTerm_1 * (pics2 + moments(label_constant<1>()) + moments(label_constant<4>())),
                    multiplyTerm_1 * (pics2 - moments(label_constant<1>()) + moments(label_constant<4>())),
                    multiplyTerm_1 * (pics2 + moments(label_constant<2>()) + moments(label_constant<7>())),
                    multiplyTerm_1 * (pics2 - moments(label_constant<2>()) + moments(label_constant<7>())),
                    multiplyTerm_1 * (pics2 + moments(label_constant<3>()) + moments(label_constant<9>())),
                    multiplyTerm_1 * (pics2 - moments(label_constant<3>()) + moments(label_constant<9>())),
                    multiplyTerm_2 * (pics2 + moments(label_constant<1>()) + moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) + moments(label_constant<5>())),
                    multiplyTerm_2 * (pics2 - moments(label_constant<1>()) - moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) + moments(label_constant<5>())),
                    multiplyTerm_2 * (pics2 + moments(label_constant<1>()) + moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) + moments(label_constant<6>())),
                    multiplyTerm_2 * (pics2 - moments(label_constant<1>()) - moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) + moments(label_constant<6>())),
                    multiplyTerm_2 * (pics2 + moments(label_constant<2>()) + moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) + moments(label_constant<8>())),
                    multiplyTerm_2 * (pics2 - moments(label_constant<2>()) - moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) + moments(label_constant<8>())),
                    multiplyTerm_2 * (pics2 + moments(label_constant<1>()) - moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) - moments(label_constant<5>())),
                    multiplyTerm_2 * (pics2 - moments(label_constant<1>()) + moments(label_constant<2>()) + moments(label_constant<4>()) + moments(label_constant<7>()) - moments(label_constant<5>())),
                    multiplyTerm_2 * (pics2 + moments(label_constant<1>()) - moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) - moments(label_constant<6>())),
                    multiplyTerm_2 * (pics2 - moments(label_constant<1>()) + moments(label_constant<3>()) + moments(label_constant<4>()) + moments(label_constant<9>()) - moments(label_constant<6>())),
                    multiplyTerm_2 * (pics2 + moments(label_constant<2>()) - moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) - moments(label_constant<8>())),
                    multiplyTerm_2 * (pics2 - moments(label_constant<2>()) + moments(label_constant<3>()) + moments(label_constant<7>()) + moments(label_constant<9>()) - moments(label_constant<8>()))};
            }

            /**
             * @brief Reconstructs the population at a given lattice point
             * @param moments The moments from which the population is to be reconstructed
             * @return The reconstructed population
             **/
            __host__ [[nodiscard]] static const std::array<scalar_t, 19> host_reconstruct(const std::array<scalar_t, 10> &moments) noexcept
            {
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[4] + moments[7] + moments[9]);

                const scalar_t multiplyTerm_0 = moments[0] * w_0<scalar_t>();

                std::array<scalar_t, 19> pop;

                pop[0] = multiplyTerm_0 * pics2;

                const scalar_t multiplyTerm_1 = moments[0] * w_1<scalar_t>();
                pop[1] = multiplyTerm_1 * (pics2 + moments[1] + moments[4]);
                pop[2] = multiplyTerm_1 * (pics2 - moments[1] + moments[4]);
                pop[3] = multiplyTerm_1 * (pics2 + moments[2] + moments[7]);
                pop[4] = multiplyTerm_1 * (pics2 - moments[2] + moments[7]);
                pop[5] = multiplyTerm_1 * (pics2 + moments[3] + moments[9]);
                pop[6] = multiplyTerm_1 * (pics2 - moments[3] + moments[9]);

                const scalar_t multiplyTerm_2 = moments[0] * w_2<scalar_t>();
                pop[7] = multiplyTerm_2 * (pics2 + moments[1] + moments[2] + moments[4] + moments[7] + moments[5]);
                pop[8] = multiplyTerm_2 * (pics2 - moments[1] - moments[2] + moments[4] + moments[7] + moments[5]);
                pop[9] = multiplyTerm_2 * (pics2 + moments[1] + moments[3] + moments[4] + moments[9] + moments[6]);
                pop[10] = multiplyTerm_2 * (pics2 - moments[1] - moments[3] + moments[4] + moments[9] + moments[6]);
                pop[11] = multiplyTerm_2 * (pics2 + moments[2] + moments[3] + moments[7] + moments[9] + moments[8]);
                pop[12] = multiplyTerm_2 * (pics2 - moments[2] - moments[3] + moments[7] + moments[9] + moments[8]);
                pop[13] = multiplyTerm_2 * (pics2 + moments[1] - moments[2] + moments[4] + moments[7] - moments[5]);
                pop[14] = multiplyTerm_2 * (pics2 - moments[1] + moments[2] + moments[4] + moments[7] - moments[5]);
                pop[15] = multiplyTerm_2 * (pics2 + moments[1] - moments[3] + moments[4] + moments[9] - moments[6]);
                pop[16] = multiplyTerm_2 * (pics2 - moments[1] + moments[3] + moments[4] + moments[9] - moments[6]);
                pop[17] = multiplyTerm_2 * (pics2 + moments[2] - moments[3] + moments[7] + moments[9] - moments[8]);
                pop[18] = multiplyTerm_2 * (pics2 - moments[2] + moments[3] + moments[7] + moments[9] - moments[8]);

                return pop;
            }

            /**
             * @brief Calculates the moments from the population density
             * @param pop The lattice population density
             * @param moments The lattice moments
             **/
            __device__ inline static void calculateMoments(const thread::array<scalar_t, 19> &pop, thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
            {
                // Equation 3
                moments(label_constant<0>()) = pop(label_constant<0>()) + pop(label_constant<1>()) + pop(label_constant<2>()) + pop(label_constant<3>()) + pop(label_constant<4>()) + pop(label_constant<5>()) + pop(label_constant<6>()) + pop(label_constant<7>()) + pop(label_constant<8>()) + pop(label_constant<9>()) + pop(label_constant<10>()) + pop(label_constant<11>()) + pop(label_constant<12>()) + pop(label_constant<13>()) + pop(label_constant<14>()) + pop(label_constant<15>()) + pop(label_constant<16>()) + pop(label_constant<17>()) + pop(label_constant<18>());
                const scalar_t invRho = static_cast<scalar_t>(1) / moments(label_constant<0>());

                // Equation 4 + force correction
                moments(label_constant<1>()) = ((pop(label_constant<1>()) - pop(label_constant<2>()) + pop(label_constant<7>()) - pop(label_constant<8>()) + pop(label_constant<9>()) - pop(label_constant<10>()) + pop(label_constant<13>()) - pop(label_constant<14>()) + pop(label_constant<15>()) - pop(label_constant<16>()))) * invRho;
                moments(label_constant<2>()) = ((pop(label_constant<3>()) - pop(label_constant<4>()) + pop(label_constant<7>()) - pop(label_constant<8>()) + pop(label_constant<11>()) - pop(label_constant<12>()) + pop(label_constant<14>()) - pop(label_constant<13>()) + pop(label_constant<17>()) - pop(label_constant<18>()))) * invRho;
                moments(label_constant<3>()) = ((pop(label_constant<5>()) - pop(label_constant<6>()) + pop(label_constant<9>()) - pop(label_constant<10>()) + pop(label_constant<11>()) - pop(label_constant<12>()) + pop(label_constant<16>()) - pop(label_constant<15>()) + pop(label_constant<18>()) - pop(label_constant<17>()))) * invRho;

                // Equation 5
                moments(label_constant<4>()) = (pop(label_constant<1>()) + pop(label_constant<2>()) + pop(label_constant<7>()) + pop(label_constant<8>()) + pop(label_constant<9>()) + pop(label_constant<10>()) + pop(label_constant<13>()) + pop(label_constant<14>()) + pop(label_constant<15>()) + pop(label_constant<16>())) * invRho - cs2<scalar_t>();
                moments(label_constant<5>()) = (pop(label_constant<7>()) - pop(label_constant<13>()) + pop(label_constant<8>()) - pop(label_constant<14>())) * invRho;
                moments(label_constant<6>()) = (pop(label_constant<9>()) - pop(label_constant<15>()) + pop(label_constant<10>()) - pop(label_constant<16>())) * invRho;
                moments(label_constant<7>()) = (pop(label_constant<3>()) + pop(label_constant<4>()) + pop(label_constant<7>()) + pop(label_constant<8>()) + pop(label_constant<11>()) + pop(label_constant<12>()) + pop(label_constant<13>()) + pop(label_constant<14>()) + pop(label_constant<17>()) + pop(label_constant<18>())) * invRho - cs2<scalar_t>();
                moments(label_constant<8>()) = (pop(label_constant<11>()) - pop(label_constant<17>()) + pop(label_constant<12>()) - pop(label_constant<18>())) * invRho;
                moments(label_constant<9>()) = (pop(label_constant<5>()) + pop(label_constant<6>()) + pop(label_constant<9>()) + pop(label_constant<10>()) + pop(label_constant<11>()) + pop(label_constant<12>()) + pop(label_constant<15>()) + pop(label_constant<16>()) + pop(label_constant<17>()) + pop(label_constant<18>())) * invRho - cs2<scalar_t>();
            }

            /**
             * @brief Calculate rho_I based upon the boundary normal vector
             * @return rho_I as a scalar_t
             * @param pop The population density
             * @param b_n The boundary normal vector at the current lattice node
             **/
            template <class B_N>
            __device__ [[nodiscard]] static inline constexpr scalar_t rho_I(const thread::array<scalar_t, 19> &pop, const B_N &b_n) noexcept
            {
                return (
                    (incomingSwitch<scalar_t>(label_constant<0>(), b_n) * pop(label_constant<0>())) +
                    (incomingSwitch<scalar_t>(label_constant<1>(), b_n) * pop(label_constant<1>())) +
                    (incomingSwitch<scalar_t>(label_constant<2>(), b_n) * pop(label_constant<2>())) +
                    (incomingSwitch<scalar_t>(label_constant<3>(), b_n) * pop(label_constant<3>())) +
                    (incomingSwitch<scalar_t>(label_constant<4>(), b_n) * pop(label_constant<4>())) +
                    (incomingSwitch<scalar_t>(label_constant<5>(), b_n) * pop(label_constant<5>())) +
                    (incomingSwitch<scalar_t>(label_constant<6>(), b_n) * pop(label_constant<6>())) +
                    (incomingSwitch<scalar_t>(label_constant<7>(), b_n) * pop(label_constant<7>())) +
                    (incomingSwitch<scalar_t>(label_constant<8>(), b_n) * pop(label_constant<8>())) +
                    (incomingSwitch<scalar_t>(label_constant<9>(), b_n) * pop(label_constant<9>())) +
                    (incomingSwitch<scalar_t>(label_constant<10>(), b_n) * pop(label_constant<10>())) +
                    (incomingSwitch<scalar_t>(label_constant<11>(), b_n) * pop(label_constant<11>())) +
                    (incomingSwitch<scalar_t>(label_constant<12>(), b_n) * pop(label_constant<12>())) +
                    (incomingSwitch<scalar_t>(label_constant<13>(), b_n) * pop(label_constant<13>())) +
                    (incomingSwitch<scalar_t>(label_constant<14>(), b_n) * pop(label_constant<14>())) +
                    (incomingSwitch<scalar_t>(label_constant<15>(), b_n) * pop(label_constant<15>())) +
                    (incomingSwitch<scalar_t>(label_constant<16>(), b_n) * pop(label_constant<16>())) +
                    (incomingSwitch<scalar_t>(label_constant<17>(), b_n) * pop(label_constant<17>())) +
                    (incomingSwitch<scalar_t>(label_constant<18>(), b_n) * pop(label_constant<18>())));
            }

            /**
             * @brief Prints information about the velocity set to the terminal
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
            __host__ static inline void printAll(const label_constant<q_> q = label_constant<0>()) noexcept
            {
                // Loop over the velocity set, print to terminal
                host::constexpr_for<q(), Q()>(
                    [&](const auto Q)
                    {
                        std::cout
                            << "    [" << label_constant<Q>() << "] = {"
                            << host_w_q<double>()[label_constant<Q>()] << ", "
                            << host_cx<int>()[label_constant<Q>()] << ", "
                            << host_cy<int>()[label_constant<Q>()] << ", "
                            << host_cz<int>()[label_constant<Q>()] << "};" << std::endl;
                    });
            }

            template <typename T, class B_N, const label_t q_>
            __device__ [[nodiscard]] static inline constexpr T incomingSwitch(const label_constant<q_> q, const B_N &b_n) noexcept
            {
                // b_n.x > 0  => EAST boundary
                // b_n.x < 0  => WEST boundary
                const bool cond_x = (b_n.isEast() & nxNeg(q)) | (b_n.isWest() & nxPos(q));

                // b_n.y > 0  => NORTH boundary
                // b_n.y < 0  => SOUTH boundary
                const bool cond_y = (b_n.isNorth() & nyNeg(q)) | (b_n.isSouth() & nyPos(q));

                // b_n.z > 0  => FRONT boundary
                // b_n.z < 0  => BACK boundary
                const bool cond_z = (b_n.isFront() & nzNeg(q)) | (b_n.isBack() & nzPos(q));

                return static_cast<T>(!(cond_x | cond_y | cond_z));
            }
        };
    }
}

#endif