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
            [[nodiscard]] static inline consteval auto Q() noexcept
            {
                return Q_;
            }

            /**
             * @brief Number of velocity components on a lattice face
             * @return 5
             **/
            [[nodiscard]] static inline consteval auto QF() noexcept
            {
                return QF_;
            }

            /**
             * @brief Returns the weight for a given lattice point
             * @return w[q]
             * @param q The lattice point
             **/
            template <const label_t q_>
            [[nodiscard]] static inline consteval scalar_t w(const lattice_constant<q_> q) noexcept
            {
                // Check that we are accessing a valid member
                static_assert(q() < Q_, "Invalid velocity set index in member function w(q)");

                // This will be eliminated by the compiler because the function is consteval
                constexpr const scalar_t W[Q_] =
                    {w0_,
                     w1_, w1_, w1_, w1_, w1_, w1_,
                     w2_, w2_, w2_, w2_, w2_, w2_, w2_, w2_, w2_, w2_, w2_, w2_};

                // Return the component
                return W[q()];
            }

            /**
             * @brief Returns the unique lattice weights
             * @return The unique lattice weights for the D3Q19 velocity set
             **/
            [[nodiscard]] static inline consteval scalar_t w0() noexcept
            {
                return w0_;
            }
            [[nodiscard]] static inline consteval scalar_t w1() noexcept
            {
                return w1_;
            }
            [[nodiscard]] static inline consteval scalar_t w2() noexcept
            {
                return w2_;
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
             * @brief Reconstructs the population at a given lattice point
             * @param moments The 10 moments at the lattice point
             * @param pop The population to be reconstructed
             **/
            __device__ static inline void reconstruct(const scalar_t (&moments)[10], scalar_t (&pop)[19]) noexcept
            {
                const scalar_t multiplyTerm_0 = moments[0] * w0_;
                const scalar_t pics2 = 1.0 - cs2() * (moments[4] + moments[7] + moments[9]);
                pop[0] = multiplyTerm_0 * (pics2);
                const scalar_t multiplyTerm_1 = moments[0] * w1_;
                pop[1] = multiplyTerm_1 * (pics2 + moments[1] + moments[4]);
                pop[2] = multiplyTerm_1 * (pics2 - moments[1] + moments[4]);
                pop[3] = multiplyTerm_1 * (pics2 + moments[2] + moments[7]);
                pop[4] = multiplyTerm_1 * (pics2 - moments[2] + moments[7]);
                pop[5] = multiplyTerm_1 * (pics2 + moments[3] + moments[9]);
                pop[6] = multiplyTerm_1 * (pics2 - moments[3] + moments[9]);
                const scalar_t multiplyTerm_2 = moments[0] * w2_;
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

            /**
             * @brief Saves pop into the shared memory array s_pop
             * @param pop The population to be set in shared memory
             * @param s_pop The shared memory array
             **/
            __device__ static inline void popSave(const scalar_t (&pop)[19], scalar_t (&s_pop)[block::size<std::size_t>() * 18]) noexcept
            {
                // Call the implementation of the loop unrolled at compile time
                popSave_loop(pop, s_pop);
            }

            /**
             * @brief Pulls s_pop from shared memory into pop
             * @param s_pop The shared memory array from which pop is pulled
             * @param pop The array into which s_pop is pulled
             **/
            __device__ static inline void popPull(const scalar_t (&s_pop)[block::size<std::size_t>() * 18], scalar_t (&pop)[19]) noexcept
            {
                const std::size_t xp1 = (threadIdx.x + 1 + block::nx<std::size_t>()) % block::nx<std::size_t>();
                const std::size_t xm1 = (threadIdx.x - 1 + block::nx<std::size_t>()) % block::nx<std::size_t>();
                const std::size_t yp1 = (threadIdx.y + 1 + block::ny<std::size_t>()) % block::ny<std::size_t>();
                const std::size_t ym1 = (threadIdx.y - 1 + block::ny<std::size_t>()) % block::ny<std::size_t>();
                const std::size_t zp1 = (threadIdx.z + 1 + block::nz<std::size_t>()) % block::nz<std::size_t>();
                const std::size_t zm1 = (threadIdx.z - 1 + block::nz<std::size_t>()) % block::nz<std::size_t>();

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
            template <class M, class G>
            __device__ static inline void popLoad(const M &mesh, const G &interface, scalar_t (&pop)[19]) noexcept
            {
                const std::size_t tx = threadIdx.x;
                const std::size_t ty = threadIdx.y;
                const std::size_t tz = threadIdx.z;
                const std::size_t bx = blockIdx.x;
                const std::size_t by = blockIdx.y;
                const std::size_t bz = blockIdx.z;
                const std::size_t txm1 = (tx - 1 + block::nx<std::size_t>()) % block::nx<std::size_t>();
                const std::size_t txp1 = (tx + 1 + block::nx<std::size_t>()) % block::nx<std::size_t>();
                const std::size_t tym1 = (ty - 1 + block::ny<std::size_t>()) % block::ny<std::size_t>();
                const std::size_t typ1 = (ty + 1 + block::ny<std::size_t>()) % block::ny<std::size_t>();
                const std::size_t tzm1 = (tz - 1 + block::nz<std::size_t>()) % block::nz<std::size_t>();
                const std::size_t tzp1 = (tz + 1 + block::nz<std::size_t>()) % block::nz<std::size_t>();
                const std::size_t bxm1 = (bx - 1 + block::nxBlocks<std::size_t>(mesh.nx())) % block::nxBlocks<std::size_t>(mesh.nx());
                const std::size_t bxp1 = (bx + 1 + block::nxBlocks<std::size_t>(mesh.nx())) % block::nxBlocks<std::size_t>(mesh.nx());
                const std::size_t bym1 = (by - 1 + block::nyBlocks<std::size_t>(mesh.ny())) % block::nyBlocks<std::size_t>(mesh.ny());
                const std::size_t byp1 = (by + 1 + block::nyBlocks<std::size_t>(mesh.ny())) % block::nyBlocks<std::size_t>(mesh.ny());
                const std::size_t bzm1 = (bz - 1 + block::nzBlocks<std::size_t>(mesh.nz())) % block::nzBlocks<std::size_t>(mesh.nz());
                const std::size_t bzp1 = (bz + 1 + block::nzBlocks<std::size_t>(mesh.nz())) % block::nzBlocks<std::size_t>(mesh.nz());

                if (tx == block::West<std::size_t>()) // West
                {
                    pop[1] = interface.fGhost().x1()[idxPopX<M>(ty, tz, bxm1, by, bz, QF_, 0, mesh)];
                    pop[7] = interface.fGhost().x1()[idxPopX<M>(tym1, tz, bxm1, ((block::South(ty)) ? bym1 : by), bz, QF_, 1, mesh)];
                    pop[9] = interface.fGhost().x1()[idxPopX<M>(ty, tzm1, bxm1, by, ((block::Back(tz)) ? bzm1 : bz), QF_, 2, mesh)];
                    pop[13] = interface.fGhost().x1()[idxPopX<M>(typ1, tz, bxm1, ((block::North(ty)) ? byp1 : by), bz, QF_, 3, mesh)];
                    pop[15] = interface.fGhost().x1()[idxPopX<M>(ty, tzp1, bxm1, by, ((block::Front(tz)) ? bzp1 : bz), QF_, 4, mesh)];
                }
                else if (tx == block::East<std::size_t>()) // East
                {
                    pop[2] = interface.fGhost().x0()[idxPopX<M>(ty, tz, bxp1, by, bz, QF_, 0, mesh)];
                    pop[8] = interface.fGhost().x0()[idxPopX<M>(typ1, tz, bxp1, ((block::North(ty)) ? byp1 : by), bz, QF_, 1, mesh)];
                    pop[10] = interface.fGhost().x0()[idxPopX<M>(ty, tzp1, bxp1, by, ((block::Front(tz)) ? bzp1 : bz), QF_, 2, mesh)];
                    pop[14] = interface.fGhost().x0()[idxPopX<M>(tym1, tz, bxp1, ((block::South(ty)) ? bym1 : by), bz, QF_, 3, mesh)];
                    pop[16] = interface.fGhost().x0()[idxPopX<M>(ty, tzm1, bxp1, by, ((block::Front(tz)) ? bzm1 : bz), QF_, 4, mesh)];
                }

                if (ty == block::South<std::size_t>()) // South
                {
                    pop[3] = interface.fGhost().y1()[idxPopY<M>(tx, tz, bx, bym1, bz, QF_, 0, mesh)];
                    pop[7] = interface.fGhost().y1()[idxPopY<M>(txm1, tz, ((block::West(tx)) ? bxm1 : bx), bym1, bz, QF_, 1, mesh)];
                    pop[11] = interface.fGhost().y1()[idxPopY<M>(tx, tzm1, bx, bym1, ((block::Front(tz)) ? bzm1 : bz), QF_, 2, mesh)];
                    pop[14] = interface.fGhost().y1()[idxPopY<M>(txp1, tz, ((block::East(tx)) ? bxp1 : bx), bym1, bz, QF_, 3, mesh)];
                    pop[17] = interface.fGhost().y1()[idxPopY<M>(tx, tzp1, bx, bym1, ((block::Front(tz)) ? bzp1 : bz), QF_, 4, mesh)];
                }
                else if (ty == block::North<std::size_t>()) // North
                {
                    pop[4] = interface.fGhost().y0()[idxPopY<M>(tx, tz, bx, byp1, bz, QF_, 0, mesh)];
                    pop[8] = interface.fGhost().y0()[idxPopY<M>(txp1, tz, ((block::East(tx)) ? bxp1 : bx), byp1, bz, QF_, 1, mesh)];
                    pop[12] = interface.fGhost().y0()[idxPopY<M>(tx, tzp1, bx, byp1, ((block::Front(tz)) ? bzp1 : bz), QF_, 2, mesh)];
                    pop[13] = interface.fGhost().y0()[idxPopY<M>(txm1, tz, ((block::West(tx)) ? bxm1 : bx), byp1, bz, QF_, 3, mesh)];
                    pop[18] = interface.fGhost().y0()[idxPopY<M>(tx, tzm1, bx, byp1, ((block::Front(tz)) ? bzm1 : bz), QF_, 4, mesh)];
                }

                if (tz == block::Back<std::size_t>()) // Back
                {
                    pop[5] = interface.fGhost().z1()[idxPopZ<M>(tx, ty, bx, by, bzm1, QF_, 0, mesh)];
                    pop[9] = interface.fGhost().z1()[idxPopZ<M>(txm1, ty, ((block::West(tx)) ? bxm1 : bx), by, bzm1, QF_, 1, mesh)];
                    pop[11] = interface.fGhost().z1()[idxPopZ<M>(tx, tym1, bx, ((block::South(ty)) ? bym1 : by), bzm1, QF_, 2, mesh)];
                    pop[16] = interface.fGhost().z1()[idxPopZ<M>(txp1, ty, ((block::East(tx)) ? bxp1 : bx), by, bzm1, QF_, 3, mesh)];
                    pop[18] = interface.fGhost().z1()[idxPopZ<M>(tx, typ1, bx, ((block::North(ty)) ? byp1 : by), bzm1, QF_, 4, mesh)];
                }
                else if (tz == block::Front<std::size_t>()) // Front
                {
                    pop[6] = interface.fGhost().z0()[idxPopZ<M>(tx, ty, bx, by, bzp1, QF_, 0, mesh)];
                    pop[10] = interface.fGhost().z0()[idxPopZ<M>(txp1, ty, ((block::East(tx)) ? bxp1 : bx), by, bzp1, QF_, 1, mesh)];
                    pop[12] = interface.fGhost().z0()[idxPopZ<M>(tx, typ1, bx, ((block::North(ty)) ? byp1 : by), bzp1, QF_, 2, mesh)];
                    pop[15] = interface.fGhost().z0()[idxPopZ<M>(txm1, ty, ((block::West(tx)) ? bxm1 : bx), by, bzp1, QF_, 3, mesh)];
                    pop[17] = interface.fGhost().z0()[idxPopZ<M>(tx, tym1, bx, ((block::South(ty)) ? bym1 : by), bzp1, QF_, 4, mesh)];
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

        private:
            /**
             * @brief Lattice weights
             **/
            static constexpr const scalar_t w0_ = 1.0 / 3.0;
            static constexpr const scalar_t w1_ = 1.0 / 18.0;
            static constexpr const scalar_t w2_ = 1.0 / 36.0;

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
            __device__ static inline void popSave_loop(const scalar_t (&pop)[19], scalar_t (&s_pop)[block::size<std::size_t>() * 18]) noexcept
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

            template <const label_t q_ = 0>
            static inline void printAll(const lattice_constant<q_> q = lattice_constant<0>()) noexcept
            {
                // Print the lattice weight to the terminal
                std::cout << "    [" << lattice_constant<q_>() << "] = {" << w(lattice_constant<q_>()) << ", " << cx(lattice_constant<q_>()) << ", " << cy(lattice_constant<q_>()) << ", " << cz(lattice_constant<q_>()) << "};" << std::endl;

                // Check that we have not reached the end of the loop
                if constexpr (q_ < Q_ - 1)
                {
                    // Continue if the next iteration is not the last
                    printAll(lattice_constant<q_ + 1>());
                }
            }
        };
    }
}

#endif