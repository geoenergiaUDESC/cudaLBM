/**
Filename: D3Q19.cuh
Contents: Definition of the D3Q19 velocity set
**/

#ifndef __MBLBM_D3Q19_CUH
#define __MBLBM_D3Q19_CUH

#include "velocitySet.cuh"

namespace mbLBM
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
                     w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2(), w_2()};

                // Return the component
                return W[q()];
            }

            /**
             * @brief Returns the unique lattice weights
             * @return The unique lattice weights for the D3Q19 velocity set
             **/
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_0() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(1.0) / static_cast<double>(3.0));
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_1() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(1.0) / static_cast<double>(18.0));
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_2() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(1.0) / static_cast<double>(36.0));
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

                // This will be eliminated by the compiler because the function is consteval
                constexpr const scalar_t CX[Q_] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};

                // Return the component
                return CX[q()];
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

                // This will be eliminated by the compiler because the function is consteval
                constexpr const scalar_t CY[Q_] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};

                // Return the component
                return CY[q()];
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

                // This will be eliminated by the compiler because the function is consteval
                constexpr const scalar_t CZ[Q_] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};

                // Return the component
                return CZ[q()];
            }

            /**
             * @brief Returns the equilibrium distribution for a given lattice index
             * @return f_eq[q]
             * @param rhow w_q[q] * rho
             * @param uc3 3 * ((u * cx[q]) + (v * cy[q]) + (w * [cz]))
             * @param p1_muu 1 - 1.5 * ((u ^ 2) + (v ^ 2) + (w ^ 2))
             **/
            [[nodiscard]] static inline constexpr scalar_t f_eq(const scalar_t rhow, const scalar_t uc3, const scalar_t p1_muu) noexcept
            {
                return (rhow * (p1_muu + uc3 * (static_cast<scalar_t>(1.0) + uc3 * static_cast<scalar_t>(0.5))));
            }

            /**
             * @brief Returns the population density
             * @return f_eq
             * @param rho Density
             * @param u The x-component of velocity
             * @param v The y-component of velocity
             * @param w The z-component of velocity
             **/
            [[nodiscard]] static inline constexpr std::array<scalar_t, 19> f_eq(const scalar_t rho, const scalar_t u, const scalar_t v, const scalar_t w) noexcept
            {
                // Define equilibrium populations
                std::array<scalar_t, 19> pop{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                f_eq_loop<0>(pop, rho, u, v, w);

                return pop;
            }

            /**
             * @brief Reconstructs the population at a given lattice point
             * @param moments The 10 moments at the lattice point
             * @param pop The population to be reconstructed
             **/
            __device__ static inline void reconstruct(
                scalar_t pop[19],
                const scalar_t moments[10]) noexcept
            {
                const scalar_t pics2 = 1.0 - cs2() * (moments[4] + moments[7] + moments[9]);

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
            }

            __host__ static inline constexpr std::array<scalar_t, 19> reconstruct(const std::array<scalar_t, 10> moments) noexcept
            {
                const scalar_t multiplyTerm_0 = moments[0] * w_0();
                const scalar_t multiplyTerm_1 = moments[0] * w_1();
                const scalar_t multiplyTerm_2 = moments[0] * w_2();
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2() * (moments[4] + moments[7] + moments[9]);

                return std::array<scalar_t, 19>{
                    multiplyTerm_0 * (pics2),
                    multiplyTerm_1 * (pics2 + moments[1] + moments[4]),
                    multiplyTerm_1 * (pics2 - moments[1] + moments[4]),
                    multiplyTerm_1 * (pics2 + moments[2] + moments[7]),
                    multiplyTerm_1 * (pics2 - moments[2] + moments[7]),
                    multiplyTerm_1 * (pics2 + moments[3] + moments[9]),
                    multiplyTerm_1 * (pics2 - moments[3] + moments[9]),
                    multiplyTerm_2 * (pics2 + moments[1] + moments[2] + moments[4] + moments[7] + moments[5]),
                    multiplyTerm_2 * (pics2 - moments[1] - moments[2] + moments[4] + moments[7] + moments[5]),
                    multiplyTerm_2 * (pics2 + moments[1] + moments[3] + moments[4] + moments[9] + moments[6]),
                    multiplyTerm_2 * (pics2 - moments[1] - moments[3] + moments[4] + moments[9] + moments[6]),
                    multiplyTerm_2 * (pics2 + moments[2] + moments[3] + moments[7] + moments[9] + moments[8]),
                    multiplyTerm_2 * (pics2 - moments[2] - moments[3] + moments[7] + moments[9] + moments[8]),
                    multiplyTerm_2 * (pics2 + moments[1] - moments[2] + moments[4] + moments[7] - moments[5]),
                    multiplyTerm_2 * (pics2 - moments[1] + moments[2] + moments[4] + moments[7] - moments[5]),
                    multiplyTerm_2 * (pics2 + moments[1] - moments[3] + moments[4] + moments[9] - moments[6]),
                    multiplyTerm_2 * (pics2 - moments[1] + moments[3] + moments[4] + moments[9] - moments[6]),
                    multiplyTerm_2 * (pics2 + moments[2] - moments[3] + moments[7] + moments[9] - moments[8]),
                    multiplyTerm_2 * (pics2 - moments[2] + moments[3] + moments[7] + moments[9] - moments[8])};
            }

            /**
             * @brief Saves pop into the shared memory array s_pop
             * @param pop The population to be set in shared memory
             * @param s_pop The shared memory array
             **/
            __device__ static inline void popSave(
                const scalar_t pop[19],
                scalar_t s_pop[block::size() * 18]) noexcept
            {
                s_pop[idxPopBlock<0>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[1];
                s_pop[idxPopBlock<1>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[2];
                s_pop[idxPopBlock<2>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[3];
                s_pop[idxPopBlock<3>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[4];
                s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[5];
                s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[6];
                s_pop[idxPopBlock<6>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[7];
                s_pop[idxPopBlock<7>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[8];
                s_pop[idxPopBlock<8>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[9];
                s_pop[idxPopBlock<9>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[10];
                s_pop[idxPopBlock<10>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[11];
                s_pop[idxPopBlock<11>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[12];
                s_pop[idxPopBlock<12>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[13];
                s_pop[idxPopBlock<13>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[14];
                s_pop[idxPopBlock<14>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[15];
                s_pop[idxPopBlock<15>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[16];
                s_pop[idxPopBlock<16>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[17];
                s_pop[idxPopBlock<17>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[18];

                // Call the implementation of the loop unrolled at compile time
                // popSave_loop(pop, s_pop);
                __syncthreads();
            }

            /**
             * @brief Pulls s_pop from shared memory into pop
             * @param s_pop The shared memory array from which pop is pulled
             * @param pop The array into which s_pop is pulled
             **/
            __device__ static inline void popPull(
                scalar_t pop[19],
                const scalar_t s_pop[block::size() * 18]) noexcept
            {
                const label_t xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
                const label_t xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

                const label_t yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
                const label_t ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

                const label_t zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
                const label_t zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;

                pop[1] = s_pop[idxPopBlock<0>(xm1, threadIdx.y, threadIdx.z)];
                pop[2] = s_pop[idxPopBlock<1>(xp1, threadIdx.y, threadIdx.z)];
                pop[3] = s_pop[idxPopBlock<2>(threadIdx.x, ym1, threadIdx.z)];
                pop[4] = s_pop[idxPopBlock<3>(threadIdx.x, yp1, threadIdx.z)];
                pop[5] = s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, zm1)];
                pop[6] = s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, zp1)];
                pop[7] = s_pop[idxPopBlock<6>(xm1, ym1, threadIdx.z)];
                pop[8] = s_pop[idxPopBlock<7>(xp1, yp1, threadIdx.z)];
                pop[9] = s_pop[idxPopBlock<8>(xm1, threadIdx.y, zm1)];
                pop[10] = s_pop[idxPopBlock<9>(xp1, threadIdx.y, zp1)];
                pop[11] = s_pop[idxPopBlock<10>(threadIdx.x, ym1, zm1)];
                pop[12] = s_pop[idxPopBlock<11>(threadIdx.x, yp1, zp1)];
                pop[13] = s_pop[idxPopBlock<12>(xm1, yp1, threadIdx.z)];
                pop[14] = s_pop[idxPopBlock<13>(xp1, ym1, threadIdx.z)];
                pop[15] = s_pop[idxPopBlock<14>(xm1, threadIdx.y, zp1)];
                pop[16] = s_pop[idxPopBlock<15>(xp1, threadIdx.y, zm1)];
                pop[17] = s_pop[idxPopBlock<16>(threadIdx.x, ym1, zp1)];
                pop[18] = s_pop[idxPopBlock<17>(threadIdx.x, yp1, zm1)];
            }

            /**
             * @brief Loads the population densities from neighbouring CUDA blocks
             * @param mesh The global mesh
             * @param interface The ghost interface used to exchange pop
             * @param pop The array into which the population densities from the neighbouring block is loaded
             **/
            __device__ static inline void popLoad(
                const scalar_t *__restrict__ const x0,
                const scalar_t *__restrict__ const x1,
                const scalar_t *__restrict__ const y0,
                const scalar_t *__restrict__ const y1,
                const scalar_t *__restrict__ const z0,
                const scalar_t *__restrict__ const z1,
                scalar_t pop[19]) noexcept
            {
                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;

                const label_t txp1 = (tx + 1 + BLOCK_NX) % BLOCK_NX;
                const label_t txm1 = (tx - 1 + BLOCK_NX) % BLOCK_NX;

                const label_t typ1 = (ty + 1 + BLOCK_NY) % BLOCK_NY;
                const label_t tym1 = (ty - 1 + BLOCK_NY) % BLOCK_NY;

                const label_t tzp1 = (tz + 1 + BLOCK_NZ) % BLOCK_NZ;
                const label_t tzm1 = (tz - 1 + BLOCK_NZ) % BLOCK_NZ;

                const label_t bxm1 = (bx - 1 + NUM_BLOCK_X) % NUM_BLOCK_X;
                const label_t bxp1 = (bx + 1 + NUM_BLOCK_X) % NUM_BLOCK_X;

                const label_t bym1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
                const label_t byp1 = (by + 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;

                const label_t bzm1 = (bz - 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;
                const label_t bzp1 = (bz + 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;

                if (tx == 0)
                { // w
                    pop[1] = x1[idxPopX<0>(ty, tz, bxm1, by, bz)];
                    pop[7] = x1[idxPopX<1>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)];
                    pop[9] = x1[idxPopX<2>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))];
                    pop[13] = x1[idxPopX<3>(typ1, tz, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bz)];
                    pop[15] = x1[idxPopX<4>(ty, tzp1, bxm1, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                }
                else if (tx == (BLOCK_NX - 1))
                { // e
                    pop[2] = x0[idxPopX<0>(ty, tz, bxp1, by, bz)];
                    pop[8] = x0[idxPopX<1>(typ1, tz, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bz)];
                    pop[10] = x0[idxPopX<2>(ty, tzp1, bxp1, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                    pop[14] = x0[idxPopX<3>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)];
                    pop[16] = x0[idxPopX<4>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))];
                }

                if (ty == 0)
                { // s
                    pop[3] = y1[idxPopY<0>(tx, tz, bx, bym1, bz)];
                    pop[7] = y1[idxPopY<1>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)];
                    pop[11] = y1[idxPopY<2>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))];
                    pop[14] = y1[idxPopY<3>(txp1, tz, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), bym1, bz)];
                    pop[17] = y1[idxPopY<4>(tx, tzp1, bx, bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                }
                else if (ty == (BLOCK_NY - 1))
                { // n
                    pop[4] = y0[idxPopY<0>(tx, tz, bx, byp1, bz)];
                    pop[8] = y0[idxPopY<1>(txp1, tz, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), byp1, bz)];
                    pop[12] = y0[idxPopY<2>(tx, tzp1, bx, byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                    pop[13] = y0[idxPopY<3>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)];
                    pop[18] = y0[idxPopY<4>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))];
                }

                if (tz == 0)
                { // b
                    pop[5] = z1[idxPopZ<0>(tx, ty, bx, by, bzm1)];
                    pop[9] = z1[idxPopZ<1>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)];
                    pop[11] = z1[idxPopZ<2>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)];
                    pop[16] = z1[idxPopZ<3>(txp1, ty, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bzm1)];
                    pop[18] = z1[idxPopZ<4>(tx, typ1, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzm1)];
                }
                else if (tz == (BLOCK_NZ - 1))
                { // f
                    pop[6] = z0[idxPopZ<0>(tx, ty, bx, by, bzp1)];
                    pop[10] = z0[idxPopZ<1>(txp1, ty, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bzp1)];
                    pop[12] = z0[idxPopZ<2>(tx, typ1, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzp1)];
                    pop[15] = z0[idxPopZ<3>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)];
                    pop[17] = z0[idxPopZ<4>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)];
                }
            }

            /**
             * @brief Prints information about the velocity set to the terminal
             **/
            static void print() noexcept
            {
                std::cout << "D3Q19 {w, cx, cy, cz}:" << std::endl;
                std::cout << "{" << std::endl;
                printAll();
                std::cout << "}" << std::endl;
                std::cout << std::endl;
            }

            __device__ static inline void calculateMoments(
                const scalar_t pop[19],
                scalar_t moments[10]) noexcept
            {
                // Equation 3
                moments[0] = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
                const scalar_t invRho = 1.0 / moments[0];

                // Equation 4
                moments[1] = (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]) * invRho;
                moments[2] = (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]) * invRho;
                moments[3] = (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]) * invRho;

                // Equation 5
                moments[4] = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2();
                moments[5] = (pop[7] - pop[13] + pop[8] - pop[14]) * invRho;
                moments[6] = (pop[9] - pop[15] + pop[10] - pop[16]) * invRho;
                moments[7] = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2();
                moments[8] = (pop[11] - pop[17] + pop[12] - pop[18]) * invRho;
                moments[9] = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2();
            }

        private:
            /**
             * @brief Lattice weights
             **/
            // static constexpr const scalar_t w_0_ = static_cast<scalar_t>(static_cast<long double>(1.0) / static_cast<long double>(3.0));
            // static constexpr const scalar_t w_1_ = static_cast<scalar_t>(static_cast<long double>(1.0) / static_cast<long double>(18.0));
            // static constexpr const scalar_t w_2_ = static_cast<scalar_t>(static_cast<long double>(1.0) / static_cast<long double>(36.0));

            /**
             * @brief Number of velocity components in the lattice
             **/
            static constexpr const label_t Q_ = 19;

            /**
             * @brief Number of velocity components on each lattice face
             **/
            static constexpr const label_t QF_ = 5;

            /**
             * @brief Implementation of the population save loop
             * @param pop The population to be set in shared memory
             * @param s_pop The shared memory array
             * @note This function effectively unrolls the loop at compile-time and checks for its bounds
             **/
            template <const label_t q_ = 0>
            __device__ static inline constexpr void popSave_loop(const scalar_t pop[19], scalar_t s_pop[block::size() * 18]) noexcept
            {
                // Check at compile time that the loop is correctly bounded
                static_assert(q_ + 1 < 19, "Compile error in popSave: Loop is incorrectly bounded");

                // Put pop[q + 1] into s_pop[q]
                s_pop[idxPopBlock<q_>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[q_ + 1];

                // Check that we have not reached the end of the loop
                if constexpr (q_ < Q_ - 2)
                {
                    // Continue if the next iteration is not the last
                    popSave_loop<q_ + 1>(pop, s_pop);
                }
            }

            /**
             * @brief Implementation of the equilibrium distribution loop
             * @param pop The population to be set
             * @param rho Density
             * @param u The x-component of velocity
             * @param v The y-component of velocity
             * @param w The z-component of velocity
             * @note This function effectively unrolls the loop at compile-time and checks for its bounds
             **/
            template <const label_t q_ = 0>
            static inline constexpr void f_eq_loop(
                std::array<scalar_t, 19> pop,
                const scalar_t rho,
                const scalar_t u, const scalar_t v, const scalar_t w) noexcept
            {
                // Check at compile time that the loop is correctly bounded
                static_assert(q_ + 1 < 19, "Compile error in f_eq: Loop is incorrectly bounded");

                // Compute the equilibrium distribution for q
                pop[q_] = f_eq(
                    w_q(lattice_constant<q_>()) * rho,
                    static_cast<scalar_t>(3.0) * ((u * cx(lattice_constant<q_>())) + (v * cy(lattice_constant<q_>())) + (w * cz(lattice_constant<q_>()))),
                    static_cast<scalar_t>(1.0) - static_cast<scalar_t>(1.5) * ((u * u) + (v * v) + (w * w)));

                // Check that we have not reached the end of the loop
                if constexpr (q_ < Q_ - 2)
                {
                    // Continue if the next iteration is not the last
                    f_eq_loop<q_ + 1>(pop, rho, u, v, w);
                }
            }

            /**
             * @brief Implementation of the print loop
             * @note This function effectively unrolls the loop at compile-time and checks for its bounds
             **/
            template <const label_t q_ = 0>
            static inline void printAll(const lattice_constant<q_> q = lattice_constant<0>()) noexcept
            {
                // Print the lattice weight to the terminal
                std::cout << "    [" << lattice_constant<q_>() << "] = {" << w_q(lattice_constant<q_>()) << ", " << cx(lattice_constant<q_>()) << ", " << cy(lattice_constant<q_>()) << ", " << cz(lattice_constant<q_>()) << "};" << std::endl;

                // Check that we have not reached the end of the loop
                if constexpr (q() < Q_ - 1)
                {
                    // Continue if the next iteration is not the last
                    printAll(lattice_constant<q_ + 1>());
                }
            }
        };
    }
}

#endif