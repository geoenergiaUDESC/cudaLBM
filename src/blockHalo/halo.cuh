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
    A class handling the device halo. This class is used to exchange the
    microscopic velocity components at the edge of a CUDA block

Namespace
    LBM::device

SourceFiles
    halo.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALO_CUH
#define __MBLBM_HALO_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @class halo
         * @brief Manages halo regions for inter-block communication in CUDA LBM simulations
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         *
         * This class handles the exchange of distribution functions between adjacent
         * CUDA blocks during LBM simulations. It maintains double-buffered halo regions
         * to support efficient ping-pong swapping between computation steps.
         **/
        template <class VelocitySet>
        class halo
        {
        public:
            /**
             * @brief Constructs halo regions from moment data and mesh
             * @param[in] mesh Lattice mesh defining simulation domain
             * @param[in] programCtrl Program control parameters
             **/
            [[nodiscard]] halo(
                const host::latticeMesh &mesh,
                const programControl &programCtrl) noexcept
                : fGhost_(haloFace<VelocitySet>(
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("rho", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("u", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("v", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("w", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_xx", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_xy", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_xz", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_yy", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_yz", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_zz", mesh, programCtrl),
                      mesh)),
                  gGhost_(haloFace<VelocitySet>(
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("rho", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("u", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("v", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("w", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_xx", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_xy", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_xz", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_yy", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_yz", mesh, programCtrl),
                      host::array<scalar_t, VelocitySet, tType::instantaneous>("m_zz", mesh, programCtrl),
                      mesh)) {};

            /**
             * @brief Constructs halo regions from moment data and mesh
             * @param[in] rho,u,v,w,m_xx,m_xy,m_xz,m_yy,m_yz,m_zz Moment representation of distribution functions
             * @param[in] mesh Lattice mesh defining simulation domain
             **/
            [[nodiscard]] halo(
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &rho,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &u,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &v,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &w,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &m_xx,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &m_xy,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &m_xz,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &m_yy,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &m_yz,
                const host::array<scalar_t, VelocitySet, tType::instantaneous> &m_zz,
                const host::latticeMesh &mesh) noexcept
                : fGhost_(haloFace<VelocitySet>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)),
                  gGhost_(haloFace<VelocitySet>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)) {};

            /**
             * @brief Default destructor
             **/
            ~halo() {};

            /**
             * @brief Swaps read and write halo buffers
             * @note Synchronizes device before swapping to ensure all operations complete
             **/
            __host__ inline void swap() noexcept
            {
                checkCudaErrorsInline(cudaDeviceSynchronize());
                std::swap(fGhost_.x0Ref(), gGhost_.x0Ref());
                std::swap(fGhost_.x1Ref(), gGhost_.x1Ref());
                std::swap(fGhost_.y0Ref(), gGhost_.y0Ref());
                std::swap(fGhost_.y1Ref(), gGhost_.y1Ref());
                std::swap(fGhost_.z0Ref(), gGhost_.z0Ref());
                std::swap(fGhost_.z1Ref(), gGhost_.z1Ref());
            }

            /**
             * @brief Provides read-only access to the current read halo
             * @return Collection of const pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> fGhost() const noexcept
            {
                return {fGhost_.x0Const(), fGhost_.x1Const(), fGhost_.y0Const(), fGhost_.y1Const(), fGhost_.z0Const(), fGhost_.z1Const()};
            }

            /**
             * @brief Provides mutable access to the current write halo
             * @return Collection of mutable pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, scalar_t> gGhost() noexcept
            {
                return {gGhost_.x0(), gGhost_.x1(), gGhost_.y0(), gGhost_.y1(), gGhost_.z0(), gGhost_.z1()};
            }

            /**
             * @brief Loads halo population data from neighboring blocks
             * @param[out] pop Array to store loaded population values
             * @param[in] fx0 Pointer to x-min face halo data
             * @param[in] fx1 Pointer to x-max face halo data
             * @param[in] fy0 Pointer to y-min face halo data
             * @param[in] fy1 Pointer to y-max face halo data
             * @param[in] fz0 Pointer to z-min face halo data
             * @param[in] fz1 Pointer to z-max face halo data
             *
             * This device function loads population values from neighboring blocks'
             * halo regions based on the current thread's position within its block.
             * It handles all 18 directions of the D3Q19 lattice model.
             **/
            __device__ static inline void load(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const scalar_t *const ptrRestrict fx0,
                const scalar_t *const ptrRestrict fx1,
                const scalar_t *const ptrRestrict fy0,
                const scalar_t *const ptrRestrict fy1,
                const scalar_t *const ptrRestrict fz0,
                const scalar_t *const ptrRestrict fz1) noexcept
            {
                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;

                const label_t txp1 = (tx + 1 + block::nx()) % block::nx();
                const label_t txm1 = (tx - 1 + block::nx()) % block::nx();

                const label_t typ1 = (ty + 1 + block::ny()) % block::ny();
                const label_t tym1 = (ty - 1 + block::ny()) % block::ny();

                const label_t tzp1 = (tz + 1 + block::nz()) % block::nz();
                const label_t tzm1 = (tz - 1 + block::nz()) % block::nz();

                const label_t bxm1 = (bx - 1 + device::NUM_BLOCK_X) % device::NUM_BLOCK_X;
                const label_t bxp1 = (bx + 1 + device::NUM_BLOCK_X) % device::NUM_BLOCK_X;

                const label_t bym1 = (by - 1 + device::NUM_BLOCK_Y) % device::NUM_BLOCK_Y;
                const label_t byp1 = (by + 1 + device::NUM_BLOCK_Y) % device::NUM_BLOCK_Y;

                const label_t bzm1 = (bz - 1 + device::NUM_BLOCK_Z) % device::NUM_BLOCK_Z;
                const label_t bzp1 = (bz + 1 + device::NUM_BLOCK_Z) % device::NUM_BLOCK_Z;

                if (tx == 0)
                { // w
                    pop(label_constant<1>()) = __ldg(&fx1[idxPopX<0, VelocitySet::QF()>(ty, tz, bxm1, by, bz)]);
                    pop(label_constant<7>()) = __ldg(&fx1[idxPopX<1, VelocitySet::QF()>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)]);
                    pop(label_constant<9>()) = __ldg(&fx1[idxPopX<2, VelocitySet::QF()>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))]);
                    pop(label_constant<13>()) = __ldg(&fx1[idxPopX<3, VelocitySet::QF()>(typ1, tz, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)]);
                    pop(label_constant<15>()) = __ldg(&fx1[idxPopX<4, VelocitySet::QF()>(ty, tzp1, bxm1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                }
                else if (tx == (block::nx() - 1))
                { // e
                    pop(label_constant<2>()) = __ldg(&fx0[idxPopX<0, VelocitySet::QF()>(ty, tz, bxp1, by, bz)]);
                    pop(label_constant<8>()) = __ldg(&fx0[idxPopX<1, VelocitySet::QF()>(typ1, tz, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)]);
                    pop(label_constant<10>()) = __ldg(&fx0[idxPopX<2, VelocitySet::QF()>(ty, tzp1, bxp1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    pop(label_constant<14>()) = __ldg(&fx0[idxPopX<3, VelocitySet::QF()>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)]);
                    pop(label_constant<16>()) = __ldg(&fx0[idxPopX<4, VelocitySet::QF()>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))]);
                }

                if (ty == 0)
                { // s
                    pop(label_constant<3>()) = __ldg(&fy1[idxPopY<0, VelocitySet::QF()>(tx, tz, bx, bym1, bz)]);
                    pop(label_constant<7>()) = __ldg(&fy1[idxPopY<1, VelocitySet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)]);
                    pop(label_constant<11>()) = __ldg(&fy1[idxPopY<2, VelocitySet::QF()>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))]);
                    pop(label_constant<14>()) = __ldg(&fy1[idxPopY<3, VelocitySet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, bz)]);
                    pop(label_constant<17>()) = __ldg(&fy1[idxPopY<4, VelocitySet::QF()>(tx, tzp1, bx, bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                }
                else if (ty == (block::ny() - 1))
                { // n
                    pop(label_constant<4>()) = __ldg(&fy0[idxPopY<0, VelocitySet::QF()>(tx, tz, bx, byp1, bz)]);
                    pop(label_constant<8>()) = __ldg(&fy0[idxPopY<1, VelocitySet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, bz)]);
                    pop(label_constant<12>()) = __ldg(&fy0[idxPopY<2, VelocitySet::QF()>(tx, tzp1, bx, byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    pop(label_constant<13>()) = __ldg(&fy0[idxPopY<3, VelocitySet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)]);
                    pop(label_constant<18>()) = __ldg(&fy0[idxPopY<4, VelocitySet::QF()>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))]);
                }

                if (tz == 0)
                { // b
                    pop(label_constant<5>()) = __ldg(&fz1[idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bzm1)]);
                    pop(label_constant<9>()) = __ldg(&fz1[idxPopZ<1, VelocitySet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)]);
                    pop(label_constant<11>()) = __ldg(&fz1[idxPopZ<2, VelocitySet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)]);
                    pop(label_constant<16>()) = __ldg(&fz1[idxPopZ<3, VelocitySet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzm1)]);
                    pop(label_constant<18>()) = __ldg(&fz1[idxPopZ<4, VelocitySet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)]);
                }
                else if (tz == (block::nz() - 1))
                { // f
                    pop(label_constant<6>()) = __ldg(&fz0[idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bzp1)]);
                    pop(label_constant<10>()) = __ldg(&fz0[idxPopZ<1, VelocitySet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzp1)]);
                    pop(label_constant<12>()) = __ldg(&fz0[idxPopZ<2, VelocitySet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)]);
                    pop(label_constant<15>()) = __ldg(&fz0[idxPopZ<3, VelocitySet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)]);
                    pop(label_constant<17>()) = __ldg(&fz0[idxPopZ<4, VelocitySet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)]);
                }
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @param[in] pop Array containing population values to save
             * @param[out] gx0 Pointer to x-min face halo data
             * @param[out] gx1 Pointer to x-max face halo data
             * @param[out] gy0 Pointer to y-min face halo data
             * @param[out] gy1 Pointer to y-max face halo data
             * @param[out] gz0 Pointer to z-min face halo data
             * @param[out] gz1 Pointer to z-max face halo data
             *
             * This device function saves population values to halo regions for
             * neighboring blocks to read. It handles all 18 directions of the D3Q19 lattice model.
             **/
            __device__ static inline void save(
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                scalar_t *const ptrRestrict gx0,
                scalar_t *const ptrRestrict gx1,
                scalar_t *const ptrRestrict gy0,
                scalar_t *const ptrRestrict gy1,
                scalar_t *const ptrRestrict gz0,
                scalar_t *const ptrRestrict gz1) noexcept
            {
                const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
                const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
                const label_t z = threadIdx.z + blockDim.z * blockIdx.z;

                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                // const label_t bx = blockIdx.x;
                // const label_t by = blockIdx.y;
                // const label_t bz = blockIdx.z;

                /* write to global pop **/
                if (West(x))
                { // w
                    gx0[idxPopX<0, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<2>());
                    gx0[idxPopX<1, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<8>());
                    gx0[idxPopX<2, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<10>());
                    gx0[idxPopX<3, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<14>());
                    gx0[idxPopX<4, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<16>());
                }
                if (East(x))
                { // e
                    gx1[idxPopX<0, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<1>());
                    gx1[idxPopX<1, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<7>());
                    gx1[idxPopX<2, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<9>());
                    gx1[idxPopX<3, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<13>());
                    gx1[idxPopX<4, VelocitySet::QF()>(ty, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<15>());
                }

                if (South(y))
                { // s
                    gy0[idxPopY<0, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<4>());
                    gy0[idxPopY<1, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<8>());
                    gy0[idxPopY<2, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<12>());
                    gy0[idxPopY<3, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<13>());
                    gy0[idxPopY<4, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<18>());
                }
                if (North(y))
                { // n
                    gy1[idxPopY<0, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<3>());
                    gy1[idxPopY<1, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<7>());
                    gy1[idxPopY<2, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<11>());
                    gy1[idxPopY<3, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<14>());
                    gy1[idxPopY<4, VelocitySet::QF()>(tx, tz, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<17>());
                }

                if (Back(z))
                { // b
                    gz0[idxPopZ<0, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<6>());
                    gz0[idxPopZ<1, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<10>());
                    gz0[idxPopZ<2, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<12>());
                    gz0[idxPopZ<3, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<15>());
                    gz0[idxPopZ<4, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<17>());
                }
                if (Front(z))
                {
                    gz1[idxPopZ<0, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<5>());
                    gz1[idxPopZ<1, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<9>());
                    gz1[idxPopZ<2, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<11>());
                    gz1[idxPopZ<3, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<16>());
                    gz1[idxPopZ<4, VelocitySet::QF()>(tx, ty, blockIdx.x, blockIdx.y, blockIdx.z)] = pop(label_constant<18>());
                }
            }

        private:
            /**
             * @brief The individual halo objects
             **/
            haloFace<VelocitySet> fGhost_;
            haloFace<VelocitySet> gGhost_;

            /**
             * @brief Check if current thread is at western block boundary
             * @param[in] x Global x-coordinate
             * @return True if at western boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool West(const label_t x) noexcept
            {
                return (threadIdx.x == 0 && x != 0);
            }

            /**
             * @brief Check if current thread is at eastern block boundary
             * @param[in] x Global x-coordinate
             * @return True if at eastern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool East(const label_t x) noexcept
            {
                return (threadIdx.x == (block::nx() - 1) && x != (device::nx - 1));
            }

            /**
             * @brief Check if current thread is at southern block boundary
             * @param[in] y Global y-coordinate
             * @return True if at southern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool South(const label_t y) noexcept
            {
                return (threadIdx.y == 0 && y != 0);
            }

            /**
             * @brief Check if current thread is at northern block boundary
             * @param[in] y Global y-coordinate
             * @return True if at northern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool North(const label_t y) noexcept
            {
                return (threadIdx.y == (block::ny() - 1) && y != (device::ny - 1));
            }

            /**
             * @brief Check if current thread is at back (z-min) block boundary
             * @param[in] z Global z-coordinate
             * @return True if at back boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool Back(const label_t z) noexcept
            {
                return (threadIdx.z == 0 && z != 0);
            }

            /**
             * @brief Check if current thread is at front (z-max) block boundary
             * @param[in] z Global z-coordinate
             * @return True if at front boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool Front(const label_t z) noexcept
            {
                return (threadIdx.z == (block::nz() - 1) && z != (device::nz - 1));
            }
        };
    }
}

#endif