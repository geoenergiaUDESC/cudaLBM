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
    A class handling the individual faces of the device halo.

Namespace
    LBM::device

SourceFiles
    haloFace.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALOFACE_CUH
#define __MBLBM_HALOFACE_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @class haloFace
         * @brief Manages individual halo faces for inter-block communication in CUDA LBM
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         *
         * This class handles the storage and management of distribution functions
         * at block boundaries for all six faces (x0, x1, y0, y1, z0, z1). It provides
         * both read-only and mutable access to halo data for efficient communication
         * between adjacent CUDA blocks during LBM simulations.
         */
        template <class VelocitySet>
        class haloFace
        {
        public:
            /**
             * @brief Constructs halo faces from moment data and mesh
             * @param[in] fMom Moment representation of distribution functions (10 interlaced moments)
             * @param[in] mesh Lattice mesh defining simulation domain
             * @post All six halo faces are allocated and initialized with population data
             */
            [[nodiscard]] haloFace(const std::vector<std::vector<scalar_t>> &fMom, const host::latticeMesh &mesh) noexcept
                : x0_(device::allocateArray(initialise_pop<device::haloFaces::x(), 0>(fMom, mesh))),
                  x1_(device::allocateArray(initialise_pop<device::haloFaces::x(), 1>(fMom, mesh))),
                  y0_(device::allocateArray(initialise_pop<device::haloFaces::y(), 0>(fMom, mesh))),
                  y1_(device::allocateArray(initialise_pop<device::haloFaces::y(), 1>(fMom, mesh))),
                  z0_(device::allocateArray(initialise_pop<device::haloFaces::z(), 0>(fMom, mesh))),
                  z1_(device::allocateArray(initialise_pop<device::haloFaces::z(), 1>(fMom, mesh))) {};

            /**
             * @brief Destructor - releases all allocated device memory
             */
            ~haloFace() noexcept
            {
                cudaFree(x0_);
                cudaFree(x1_);
                cudaFree(y0_);
                cudaFree(y1_);
                cudaFree(z0_);
                cudaFree(z1_);
            }

            /**
             * @name Read-only Accessors
             * @brief Provide const access to halo face data
             * @return Const pointer to halo face data
             */
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x0Const() const noexcept
            {
                return x0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x1Const() const noexcept
            {
                return x1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y0Const() const noexcept
            {
                return y0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y1Const() const noexcept
            {
                return y1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z0Const() const noexcept
            {
                return z0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z1Const() const noexcept
            {
                return z1_;
            }

            /**
             * @name Mutable Accessors
             * @brief Provide mutable access to halo face data
             * @return Pointer to halo face data
             */
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x0() noexcept
            {
                return x0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x1() noexcept
            {
                return x1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y0() noexcept
            {
                return y0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y1() noexcept
            {
                return y1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z0() noexcept
            {
                return z0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z1() noexcept
            {
                return z1_;
            }

            /**
             * @name Pointer Reference Accessors
             * @brief Provide reference to pointer for swapping operations
             * @return Reference to pointer (used for buffer swapping)
             * @note These methods are specifically for pointer swapping and should not be used elsewhere
             */
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &x0Ref() noexcept
            {
                return x0_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &x1Ref() noexcept
            {
                return x1_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &y0Ref() noexcept
            {
                return y0_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &y1Ref() noexcept
            {
                return y1_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &z0Ref() noexcept
            {
                return z0_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &z1Ref() noexcept
            {
                return z1_;
            }

        private:
            /**
             * @brief Halo faces pointers
             **/
            scalar_t *ptrRestrict x0_;
            scalar_t *ptrRestrict x1_;
            scalar_t *ptrRestrict y0_;
            scalar_t *ptrRestrict y1_;
            scalar_t *ptrRestrict z0_;
            scalar_t *ptrRestrict z1_;

            /**
             * @brief Calculate number of elements for a halo face
             * @tparam faceIndex Direction index (x, y, or z)
             * @param[in] mesh Lattice mesh for dimensioning
             * @return Number of elements in the specified halo face
             */
            template <const label_t faceIndex>
            __host__ [[nodiscard]] static inline constexpr label_t nFaces(const host::latticeMesh &mesh) noexcept
            {
                if constexpr (faceIndex == device::haloFaces::x())
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nx()) * VelocitySet::QF();
                }
                if constexpr (faceIndex == device::haloFaces::y())
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::ny()) * VelocitySet::QF();
                }
                if constexpr (faceIndex == device::haloFaces::z())
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nz()) * VelocitySet::QF();
                }

                return 0;
            }

            /**
             * @brief Initialize population data for a specific halo face
             * @tparam faceIndex Direction index (x, y, or z)
             * @tparam side Face side (0 for min, 1 for max)
             * @param[in] fMom Moment representation of distribution functions
             * @param[in] mesh Lattice mesh for dimensioning
             * @return Initialized population data for the specified halo face
             */
            template <const label_t faceIndex, const label_t side>
            __host__ [[nodiscard]] const std::vector<scalar_t> initialise_pop(const std::vector<std::vector<scalar_t>> &fMom, const host::latticeMesh &mesh) const noexcept
            {
                std::vector<scalar_t> face(nFaces<faceIndex>(mesh), 0);

                // Loop over all blocks and threads
                for (label_t bz = 0; bz < mesh.nzBlocks(); ++bz)
                {
                    for (label_t by = 0; by < mesh.nyBlocks(); ++by)
                    {
                        for (label_t bx = 0; bx < mesh.nxBlocks(); ++bx)
                        {
                            for (label_t tz = 0; tz < block::nz(); ++tz)
                            {
                                for (label_t ty = 0; ty < block::ny(); ++ty)
                                {
                                    for (label_t tx = 0; tx < block::nx(); ++tx)
                                    {

                                        // Skip out-of-bounds elements (equivalent to GPU version)
                                        if (tx >= mesh.nx() || ty >= mesh.ny() || tz >= mesh.nz())
                                        {
                                            continue;
                                        }

                                        const label_t base = host::idx(tx, ty, tz, bx, by, bz, mesh);

                                        // Contiguous moment access
                                        const std::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(
                                            std::array<scalar_t, 10>{rho0<scalar_t>() + fMom[0][base],
                                                                     fMom[1][base],
                                                                     fMom[2][base],
                                                                     fMom[3][base],
                                                                     fMom[4][base],
                                                                     fMom[5][base],
                                                                     fMom[6][base],
                                                                     fMom[7][base],
                                                                     fMom[8][base],
                                                                     fMom[9][base]});

                                        // Handle ghost cells (equivalent to threadIdx.x/y/z checks)
                                        handleGhostCells<faceIndex, side>(face, pop, tx, ty, tz, bx, by, bz, mesh);
                                    }
                                }
                            }
                        }
                    }
                }

                return face;
            }

            /**
             * @brief Populate halo face with population data from boundary cells
             * @tparam faceIndex Direction index (x, y, or z)
             * @tparam side Face side (0 for min, 1 for max)
             * @param[out] face Halo face data to populate
             * @param[in] pop Population density values for current cell
             * @param[in] tx, ty, tz Thread indices within block
             * @param[in] bx, by, bz Block indices
             * @param[in] mesh Lattice mesh for dimensioning
             *
             * This method handles the D3Q19 lattice model, storing appropriate
             * population components based on boundary position and direction.
             */
            template <const label_t faceIndex, const label_t side>
            __host__ void static handleGhostCells(
                std::vector<scalar_t> &face,
                const std::array<scalar_t, VelocitySet::Q()> &pop,
                const label_t tx, const label_t ty, const label_t tz,
                const label_t bx, const label_t by, const label_t bz,
                const host::latticeMesh &mesh) noexcept
            {
                if constexpr (faceIndex == device::haloFaces::x())
                {
                    if constexpr (side == 0)
                    {
                        if (tx == 0)
                        { // w
                            face[host::idxPopX<0, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[2];
                            face[host::idxPopX<1, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[8];
                            face[host::idxPopX<2, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[10];
                            face[host::idxPopX<3, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[14];
                            face[host::idxPopX<4, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[16];
                        }
                    }
                    if constexpr (side == 1)
                    {
                        if (tx == (block::nx() - 1))
                        {
                            face[host::idxPopX<0, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[1];
                            face[host::idxPopX<1, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[7];
                            face[host::idxPopX<2, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[9];
                            face[host::idxPopX<3, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[13];
                            face[host::idxPopX<4, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[15];
                        }
                    }
                }

                if constexpr (faceIndex == device::haloFaces::y())
                {
                    if constexpr (side == 0)
                    {
                        if (ty == 0)
                        { // s
                            face[host::idxPopY<0, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[4];
                            face[host::idxPopY<1, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[8];
                            face[host::idxPopY<2, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[12];
                            face[host::idxPopY<3, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[13];
                            face[host::idxPopY<4, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[18];
                        }
                    }
                    if constexpr (side == 1)
                    {
                        if (ty == (block::ny() - 1))
                        {
                            face[host::idxPopY<0, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[3];
                            face[host::idxPopY<1, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[7];
                            face[host::idxPopY<2, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[11];
                            face[host::idxPopY<3, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[14];
                            face[host::idxPopY<4, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[17];
                        }
                    }
                }

                if constexpr (faceIndex == device::haloFaces::z())
                {
                    if constexpr (side == 0)
                    {
                        if (tz == 0)
                        { // b
                            face[host::idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[6];
                            face[host::idxPopZ<1, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[10];
                            face[host::idxPopZ<2, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[12];
                            face[host::idxPopZ<3, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[15];
                            face[host::idxPopZ<4, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[17];
                        }
                    }
                    if constexpr (side == 1)
                    {
                        if (tz == (block::nz() - 1))
                        {
                            face[host::idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[5];
                            face[host::idxPopZ<1, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[9];
                            face[host::idxPopZ<2, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[11];
                            face[host::idxPopZ<3, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[16];
                            face[host::idxPopZ<4, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[18];
                        }
                    }
                }
            }
        };
    }
}

#endif