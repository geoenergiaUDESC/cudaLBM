/**
Filename: D3Q27.cuh
Contents: Definition of the D3Q27 velocity set
**/

#ifndef __MBLBM_D3Q27_CUH
#define __MBLBM_D3Q27_CUH

#include "velocitySet.cuh"

namespace LBM
{
    namespace VelocitySet
    {
        class D3Q27 : private velocitySet
        {
        public:
            /**
             * @brief Constructor for the D3Q27 velocity set
             * @return A D3Q27 velocity set
             * @note This constructor is consteval
             **/
            [[nodiscard]] inline consteval D3Q27() {};

            /**
             * @brief Number of velocity components
             * @return 27
             **/
            [[nodiscard]] static inline consteval label_t Q() noexcept
            {
                return Q_;
            }

            /**
             * @brief Number of velocity components on a lattice face
             * @return 9
             **/
            [[nodiscard]] static inline consteval label_t QF() noexcept
            {
                return QF_;
            }

            /**
             * @brief Returns the weight for a given lattice point
             * @return w_q[q]
             * @param q The lattice point
             **/
            template <const label_t q_>
            [[nodiscard]] static inline consteval scalar_t w_q(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function w(q)");

                // This will be eliminated by the compiler because the function is consteval
                constexpr const scalar_t W[Q_] =
                    {w_0(),
                     w_1(), w_1(), w_1(), w_1(), w_1(), w_1(),
                     w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(),
                     w_3(), w_3(), w_3(), w_3(), w_3(), w_3(), w_3(), w_3()};

                // Return the component
                return W[q()];
            }

            /**
             * @brief Returns the unique lattice weights
             * @return The unique lattice weights for the D3Q27 velocity set
             **/
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_0() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(8) / static_cast<double>(27));
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_1() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(2) / static_cast<double>(27));
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_2() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(54));
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_3() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(216));
            }

            /**
             * @brief Returns the x lattice coefficient for a given lattice point
             * @return c_x[q]
             * @param q The lattice point
             **/
            template <const label_t q_>
            [[nodiscard]] static inline consteval scalar_t cx(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cx(q)");

                // Return the component
                return static_cast<scalar_t>(cx_[q()]);
            }
            template <const label_t q_>
            [[nodiscard]] static inline consteval bool nxNeg(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cx(q)");

                // Return the component
                return cx_[q()] < 0;
            }
            template <const label_t q_>
            [[nodiscard]] static inline consteval bool nxPos(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cx(q)");

                // Return the component
                return cx_[q()] > 0;
            }

            /**
             * @brief Returns the y lattice coefficient for a given lattice point
             * @return c_y[q]
             * @param q The lattice point
             **/
            template <const label_t q_>
            [[nodiscard]] static inline consteval scalar_t cy(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cy(q)");

                // Return the component
                return static_cast<scalar_t>(cy_[q()]);
            }
            template <const label_t q_>
            [[nodiscard]] static inline consteval bool nyNeg(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cy(q)");

                // Return the component
                return cy_[q()] < 0;
            }
            template <const label_t q_>
            [[nodiscard]] static inline consteval bool nyPos(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cy(q)");

                // Return the component
                return cy_[q()] > 0;
            }

            /**
             * @brief Returns the z lattice coefficient for a given lattice point
             * @return c_z[q]
             * @param q The lattice point
             **/
            template <const label_t q_>
            [[nodiscard]] static inline consteval scalar_t cz(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cz(q)");

                // Return the component
                return static_cast<scalar_t>(cz_[q()]);
            }
            template <const label_t q_>
            [[nodiscard]] static inline consteval bool nzNeg(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cz(q)");

                // Return the component
                return cz_[q()] < 0;
            }
            template <const label_t q_>
            [[nodiscard]] static inline consteval bool nzPos(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function cz(q)");

                // Return the component
                return cz_[q()] > 0;
            }

            /**
             * @brief Returns the population density
             * @return f_eq
             * @param u The x-component of velocity
             * @param v The y-component of velocity
             * @param w The z-component of velocity
             **/
            __host__ [[nodiscard]] static inline constexpr const std::array<scalar_t, 27> F_eq(const scalar_t u, const scalar_t v, const scalar_t w) noexcept
            {
                std::array<scalar_t, Q_> pop;

                host::constexpr_for<0, (Q_ - 1)>(
                    [&](const auto q_)
                    {
                        pop[q_] = f_eq(
                            w_q(lattice_constant<q_>()),
                            static_cast<scalar_t>(3) * ((u * cx(lattice_constant<q_>())) + (v * cy(lattice_constant<q_>())) + (w * cz(lattice_constant<q_>()))),
                            static_cast<scalar_t>(1) - static_cast<scalar_t>(1.5) * ((u * u) + (v * v) + (w * w)));
                    });

                return pop;
            }

            /**
             * @brief Reconstructs the population at a given lattice point
             * @param pop The reconstructed population
             * @param moments The moments from which the population is to be reconstructed
             **/
            __device__ static inline void reconstruct(scalar_t (&ptrRestrict pop)[27], const scalar_t (&ptrRestrict moments)[10]) noexcept
            {
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2() * (moments[4] + moments[7] + moments[9]);

                const scalar_t multiplyTerm_0 = moments[0] * w_0();
                pop[0] = multiplyTerm_0 * pics2;

                const scalar_t multiplyTerm_1 = moments[0] * w_1();
                pop[1] = multiplyTerm_1 * (pics2 + moments[1] + moments[4]);
                pop[2] = multiplyTerm_1 * (pics2 - moments[1] + moments[4]);
                pop[3] = multiplyTerm_1 * (pics2 + moments[2] + moments[7]);
                pop[4] = multiplyTerm_1 * (pics2 - moments[2] + moments[7]);
                pop[5] = multiplyTerm_1 * (pics2 + moments[3] + moments[9]);
                pop[6] = multiplyTerm_1 * (pics2 - moments[3] + moments[9]);

                const scalar_t multiplyTerm_2 = moments[0] * w_2();
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

                const scalar_t multiplyTerm_3 = moments[0] * w_3();
                pop[19] = multiplyTerm_3 * (pics2 + moments[1] + moments[2] + moments[3] + moments[4] + moments[5] + moments[6] + moments[7] + moments[8] + moments[9]);
                pop[20] = multiplyTerm_3 * (pics2 - moments[1] - moments[2] - moments[3] + moments[4] + moments[5] + moments[6] + moments[7] + moments[8] + moments[9]);
                pop[21] = multiplyTerm_3 * (pics2 + moments[1] + moments[2] - moments[3] + moments[4] + moments[5] - moments[6] + moments[7] - moments[8] + moments[9]);
                pop[22] = multiplyTerm_3 * (pics2 - moments[1] - moments[2] + moments[3] + moments[4] + moments[5] - moments[6] + moments[7] - moments[8] + moments[9]);
                pop[23] = multiplyTerm_3 * (pics2 + moments[1] - moments[2] + moments[3] + moments[4] - moments[5] + moments[6] + moments[7] - moments[8] + moments[9]);
                pop[24] = multiplyTerm_3 * (pics2 - moments[1] + moments[2] - moments[3] + moments[4] - moments[5] + moments[6] + moments[7] - moments[8] + moments[9]);
                pop[25] = multiplyTerm_3 * (pics2 - moments[1] + moments[2] + moments[3] + moments[4] - moments[5] - moments[6] + moments[7] + moments[8] + moments[9]);
                pop[26] = multiplyTerm_3 * (pics2 + moments[1] - moments[2] - moments[3] + moments[4] - moments[5] - moments[6] + moments[7] + moments[8] + moments[9]);
            }

            /**
             * @brief Reconstructs the population at a given lattice point
             * @return The reconstructed population
             * @param moments The moments from which the population is to be reconstructed
             **/
            __device__ static inline threadArray<scalar_t, 27> reconstruct(const threadArray<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
            {
                threadArray<scalar_t, VelocitySet::D3Q27::Q()> pop;

                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2() * (moments.arr[4] + moments.arr[7] + moments.arr[9]);

                const scalar_t multiplyTerm_0 = moments.arr[0] * w_0();
                pop.arr[0] = multiplyTerm_0 * pics2;

                const scalar_t multiplyTerm_1 = moments.arr[0] * w_1();
                pop.arr[1] = multiplyTerm_1 * (pics2 + moments.arr[1] + moments.arr[4]);
                pop.arr[2] = multiplyTerm_1 * (pics2 - moments.arr[1] + moments.arr[4]);
                pop.arr[3] = multiplyTerm_1 * (pics2 + moments.arr[2] + moments.arr[7]);
                pop.arr[4] = multiplyTerm_1 * (pics2 - moments.arr[2] + moments.arr[7]);
                pop.arr[5] = multiplyTerm_1 * (pics2 + moments.arr[3] + moments.arr[9]);
                pop.arr[6] = multiplyTerm_1 * (pics2 - moments.arr[3] + moments.arr[9]);

                const scalar_t multiplyTerm_2 = moments.arr[0] * w_2();
                pop.arr[7] = multiplyTerm_2 * (pics2 + moments.arr[1] + moments.arr[2] + moments.arr[4] + moments.arr[7] + moments.arr[5]);
                pop.arr[8] = multiplyTerm_2 * (pics2 - moments.arr[1] - moments.arr[2] + moments.arr[4] + moments.arr[7] + moments.arr[5]);
                pop.arr[9] = multiplyTerm_2 * (pics2 + moments.arr[1] + moments.arr[3] + moments.arr[4] + moments.arr[9] + moments.arr[6]);
                pop.arr[10] = multiplyTerm_2 * (pics2 - moments.arr[1] - moments.arr[3] + moments.arr[4] + moments.arr[9] + moments.arr[6]);
                pop.arr[11] = multiplyTerm_2 * (pics2 + moments.arr[2] + moments.arr[3] + moments.arr[7] + moments.arr[9] + moments.arr[8]);
                pop.arr[12] = multiplyTerm_2 * (pics2 - moments.arr[2] - moments.arr[3] + moments.arr[7] + moments.arr[9] + moments.arr[8]);
                pop.arr[13] = multiplyTerm_2 * (pics2 + moments.arr[1] - moments.arr[2] + moments.arr[4] + moments.arr[7] - moments.arr[5]);
                pop.arr[14] = multiplyTerm_2 * (pics2 - moments.arr[1] + moments.arr[2] + moments.arr[4] + moments.arr[7] - moments.arr[5]);
                pop.arr[15] = multiplyTerm_2 * (pics2 + moments.arr[1] - moments.arr[3] + moments.arr[4] + moments.arr[9] - moments.arr[6]);
                pop.arr[16] = multiplyTerm_2 * (pics2 - moments.arr[1] + moments.arr[3] + moments.arr[4] + moments.arr[9] - moments.arr[6]);
                pop.arr[17] = multiplyTerm_2 * (pics2 + moments.arr[2] - moments.arr[3] + moments.arr[7] + moments.arr[9] - moments.arr[8]);
                pop.arr[18] = multiplyTerm_2 * (pics2 - moments.arr[2] + moments.arr[3] + moments.arr[7] + moments.arr[9] - moments.arr[8]);

                const scalar_t multiplyTerm_3 = moments.arr[0] * w_3();
                pop.arr[19] = multiplyTerm_3 * (pics2 + moments.arr[1] + moments.arr[2] + moments.arr[3] + moments.arr[4] + moments.arr[5] + moments.arr[6] + moments.arr[7] + moments.arr[8] + moments.arr[9]);
                pop.arr[20] = multiplyTerm_3 * (pics2 - moments.arr[1] - moments.arr[2] - moments.arr[3] + moments.arr[4] + moments.arr[5] + moments.arr[6] + moments.arr[7] + moments.arr[8] + moments.arr[9]);
                pop.arr[21] = multiplyTerm_3 * (pics2 + moments.arr[1] + moments.arr[2] - moments.arr[3] + moments.arr[4] + moments.arr[5] - moments.arr[6] + moments.arr[7] - moments.arr[8] + moments.arr[9]);
                pop.arr[22] = multiplyTerm_3 * (pics2 - moments.arr[1] - moments.arr[2] + moments.arr[3] + moments.arr[4] + moments.arr[5] - moments.arr[6] + moments.arr[7] - moments.arr[8] + moments.arr[9]);
                pop.arr[23] = multiplyTerm_3 * (pics2 + moments.arr[1] - moments.arr[2] + moments.arr[3] + moments.arr[4] - moments.arr[5] + moments.arr[6] + moments.arr[7] - moments.arr[8] + moments.arr[9]);
                pop.arr[24] = multiplyTerm_3 * (pics2 - moments.arr[1] + moments.arr[2] - moments.arr[3] + moments.arr[4] - moments.arr[5] + moments.arr[6] + moments.arr[7] - moments.arr[8] + moments.arr[9]);
                pop.arr[25] = multiplyTerm_3 * (pics2 - moments.arr[1] + moments.arr[2] + moments.arr[3] + moments.arr[4] - moments.arr[5] - moments.arr[6] + moments.arr[7] + moments.arr[8] + moments.arr[9]);
                pop.arr[26] = multiplyTerm_3 * (pics2 + moments.arr[1] - moments.arr[2] - moments.arr[3] + moments.arr[4] - moments.arr[5] - moments.arr[6] + moments.arr[7] + moments.arr[8] + moments.arr[9]);

                return pop;
            }

            /**
             * @brief Reconstructs the population at a given lattice point
             * @param moments The moments from which the population is to be reconstructed
             * @return The reconstructed population
             **/
            __host__ [[nodiscard]] static const std::array<scalar_t, 27> host_reconstruct(const std::array<scalar_t, 10> &moments) noexcept
            {
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2() * (moments[4] + moments[7] + moments[9]);

                const scalar_t multiplyTerm_0 = moments[0] * w_0();

                std::array<scalar_t, 27> pop;

                pop[0] = multiplyTerm_0 * pics2;

                const scalar_t multiplyTerm_1 = moments[0] * w_1();
                pop[1] = multiplyTerm_1 * (pics2 + moments[1] + moments[4]);
                pop[2] = multiplyTerm_1 * (pics2 - moments[1] + moments[4]);
                pop[3] = multiplyTerm_1 * (pics2 + moments[2] + moments[7]);
                pop[4] = multiplyTerm_1 * (pics2 - moments[2] + moments[7]);
                pop[5] = multiplyTerm_1 * (pics2 + moments[3] + moments[9]);
                pop[6] = multiplyTerm_1 * (pics2 - moments[3] + moments[9]);

                const scalar_t multiplyTerm_2 = moments[0] * w_2();
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

                const scalar_t multiplyTerm_3 = moments[0] * w_3();
                pop[19] = multiplyTerm_3 * (pics2 + moments[1] + moments[2] + moments[3] + moments[4] + moments[5] + moments[6] + moments[7] + moments[8] + moments[9]);
                pop[20] = multiplyTerm_3 * (pics2 - moments[1] - moments[2] - moments[3] + moments[4] + moments[5] + moments[6] + moments[7] + moments[8] + moments[9]);
                pop[21] = multiplyTerm_3 * (pics2 + moments[1] + moments[2] - moments[3] + moments[4] + moments[5] - moments[6] + moments[7] - moments[8] + moments[9]);
                pop[22] = multiplyTerm_3 * (pics2 - moments[1] - moments[2] + moments[3] + moments[4] + moments[5] - moments[6] + moments[7] - moments[8] + moments[9]);
                pop[23] = multiplyTerm_3 * (pics2 + moments[1] - moments[2] + moments[3] + moments[4] - moments[5] + moments[6] + moments[7] - moments[8] + moments[9]);
                pop[24] = multiplyTerm_3 * (pics2 - moments[1] + moments[2] - moments[3] + moments[4] - moments[5] + moments[6] + moments[7] - moments[8] + moments[9]);
                pop[25] = multiplyTerm_3 * (pics2 - moments[1] + moments[2] + moments[3] + moments[4] - moments[5] - moments[6] + moments[7] + moments[8] + moments[9]);
                pop[26] = multiplyTerm_3 * (pics2 + moments[1] - moments[2] - moments[3] + moments[4] - moments[5] - moments[6] + moments[7] + moments[8] + moments[9]);

                return pop;
            }

            /**
             * @brief Calculates the moments from the population density
             * @param pop The lattice population density
             * @param moments The lattice moments
             **/
            __device__ inline static void calculateMoments(const scalar_t (&ptrRestrict pop)[27], scalar_t (&ptrRestrict moments)[10]) noexcept
            {
                // Equation 3
                moments[0] = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
                const scalar_t invRho = static_cast<scalar_t>(1) / moments[0];

                // Equation 4 + force correction
                moments[1] = ((pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] - pop[25] + pop[26])) * invRho;
                moments[2] = ((pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] - pop[23] + pop[24] + pop[25] - pop[26])) * invRho;
                moments[3] = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17] + pop[19] - pop[20] - pop[21] + pop[22] + pop[23] - pop[24] + pop[25] - pop[26])) * invRho;

                // Equation 5
                moments[4] = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - cs2();
                moments[5] = (pop[7] - pop[13] + pop[8] - pop[14] + pop[19] + pop[20] + pop[21] + pop[22] - pop[23] - pop[24] - pop[25] - pop[26]) * invRho;
                moments[6] = (pop[9] - pop[15] + pop[10] - pop[16] + pop[19] + pop[20] - pop[21] - pop[22] + pop[23] + pop[24] - pop[25] - pop[26]) * invRho;
                moments[7] = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - cs2();
                moments[8] = (pop[11] - pop[17] + pop[12] - pop[18] + pop[19] + pop[20] - pop[21] - pop[22] - pop[23] - pop[24] + pop[25] + pop[26]) * invRho;
                moments[9] = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - cs2();
            }

            /**
             * @brief Calculate rho_I based upon the boundary normal vector
             * @return rho_I as a scalar_t
             * @param pop The population density
             * @param b_n The boundary normal vector at the current lattice node
             **/
            template <class B_N>
            __device__ [[nodiscard]] static inline constexpr scalar_t rho_I(const scalar_t (&ptrRestrict pop)[27], const B_N &b_n) noexcept
            {
                return (
                    (incomingSwitch<scalar_t>(lattice_constant<0>(), b_n) * pop[0]) +
                    (incomingSwitch<scalar_t>(lattice_constant<1>(), b_n) * pop[1]) +
                    (incomingSwitch<scalar_t>(lattice_constant<2>(), b_n) * pop[2]) +
                    (incomingSwitch<scalar_t>(lattice_constant<3>(), b_n) * pop[3]) +
                    (incomingSwitch<scalar_t>(lattice_constant<4>(), b_n) * pop[4]) +
                    (incomingSwitch<scalar_t>(lattice_constant<5>(), b_n) * pop[5]) +
                    (incomingSwitch<scalar_t>(lattice_constant<6>(), b_n) * pop[6]) +
                    (incomingSwitch<scalar_t>(lattice_constant<7>(), b_n) * pop[7]) +
                    (incomingSwitch<scalar_t>(lattice_constant<8>(), b_n) * pop[8]) +
                    (incomingSwitch<scalar_t>(lattice_constant<9>(), b_n) * pop[9]) +
                    (incomingSwitch<scalar_t>(lattice_constant<10>(), b_n) * pop[10]) +
                    (incomingSwitch<scalar_t>(lattice_constant<11>(), b_n) * pop[11]) +
                    (incomingSwitch<scalar_t>(lattice_constant<12>(), b_n) * pop[12]) +
                    (incomingSwitch<scalar_t>(lattice_constant<13>(), b_n) * pop[13]) +
                    (incomingSwitch<scalar_t>(lattice_constant<14>(), b_n) * pop[14]) +
                    (incomingSwitch<scalar_t>(lattice_constant<15>(), b_n) * pop[15]) +
                    (incomingSwitch<scalar_t>(lattice_constant<16>(), b_n) * pop[16]) +
                    (incomingSwitch<scalar_t>(lattice_constant<17>(), b_n) * pop[17]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[18]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[19]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[20]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[21]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[22]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[23]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[24]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[25]) +
                    (incomingSwitch<scalar_t>(lattice_constant<18>(), b_n) * pop[26]));
            }

            /**
             * @brief Prints information about the velocity set to the terminal
             **/
            static void print() noexcept
            {
                std::cout << "D3Q27 {w, cx, cy, cz}:" << std::endl;
                std::cout << "{" << std::endl;
                printAll();
                std::cout << "};" << std::endl;
                std::cout << std::endl;
            }

        private:
            /**
             * @brief Number of velocity components in the lattice
             **/
            static constexpr const label_t Q_ = 27;

            /**
             * @brief Number of velocity components on each lattice face
             **/
            static constexpr const label_t QF_ = 9;

            /**
             * @brief Velocity sey coefficients
             **/
            static constexpr const int cx_[27] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
            static constexpr const int cy_[27] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
            static constexpr const int cz_[27] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};

            /**
             * @brief Implementation of the print loop
             * @note This function effectively unrolls the loop at compile-time and checks for its bounds
             **/
            template <const label_t q_ = 0>
            static inline void printAll(const lattice_constant<q_> q = lattice_constant<0>()) noexcept
            {
                // Print the lattice weight to the terminal
                std::cout << "    [" << lattice_constant<q_>() << "] = {" << w_q(lattice_constant<q_>()) << ", " << static_cast<int>(cx(lattice_constant<q_>())) << ", " << static_cast<int>(cy(lattice_constant<q_>())) << ", " << static_cast<int>(cz(lattice_constant<q_>())) << "};" << std::endl;

                // Check that we have not reached the end of the loop
                if constexpr (q() < Q_ - 1)
                {
                    // Continue if the next iteration is not the last
                    printAll(lattice_constant<q_ + 1>());
                }
            }

            /**
             * @brief Determine whether or not a component of the velocity set is incoming or outgoing based upon the boundary normal
             * @return 1 as type T if it is an incoming velocity component, 0 otherwise
             * @param q The index of the velocity set
             * @param b_n The boundary normal vector at the current lattice node
             **/
            template <typename T, class B_N, const label_t q_>
            __device__ [[nodiscard]] static inline constexpr T incomingSwitch(const lattice_constant<q_> q, const B_N &b_n) noexcept
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