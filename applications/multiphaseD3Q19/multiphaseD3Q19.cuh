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
    Main kernels for the multiphase moment representation with the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    multiphaseD3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MULTIPHASED3Q19_CUH
#define __MBLBM_MULTIPHASED3Q19_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/streaming/streaming.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/functionObjects/objectRegistry.cuh"

namespace config
{
    constexpr bool periodicX = true;
    constexpr bool periodicY = true;
}

namespace LBM
{

    using VelocitySet = D3Q19;
    using PhaseVelocitySet = D3Q7;
    using Collision = secondOrder;

    using HydroHalo = device::halo<VelocitySet, config::periodicX, config::periodicY>;
    using PhaseHalo = device::halo<PhaseVelocitySet, config::periodicX, config::periodicY>;

    __device__ __host__ [[nodiscard]] inline consteval label_t smem_alloc_size() noexcept { return block::sharedMemoryBufferSize<VelocitySet, 11>(sizeof(scalar_t)); }

    __device__ __host__ [[nodiscard]] inline consteval bool out_of_bounds_check() noexcept
    {
#ifdef OOB_CHECK
        return true;
#else
        return false;
#endif
    }

    __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }
#define launchBoundsD3Q19 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

    /**
     * @brief Performs the streaming step of the lattice Boltzmann method using the multiphase moment representation (D3Q19 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param normx Pointer to x-comp1nt of the unit interface normal
     * @param normy Pointer to y-comp1nt of the unit interface normal
     * @param normz Pointer to z-comp1nt of the unit interface normal
     * @param fBlockHalo Object containing pointers to the block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q19 __global__ void multiphaseStream(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const scalar_t *const ptrRestrict normx,
        const scalar_t *const ptrRestrict normy,
        const scalar_t *const ptrRestrict normz,
        const device::ptrCollection<6, const scalar_t> fGhostHydro,
        const device::ptrCollection<6, scalar_t> gGhostHydro,
        const device::ptrCollection<6, const scalar_t> fGhostPhase,
        const device::ptrCollection<6, scalar_t> gGhostPhase)
    {
        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds())
            {
                return;
            }
        }

        const label_t idx = device::idx();

        const scalar_t normx_ = normx[idx];
        const scalar_t normy_ = normy[idx];
        const scalar_t normz_ = normz[idx];

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Declare shared memory
        extern __shared__ scalar_t shared_buffer[];
        __shared__ scalar_t shared_buffer_g[(PhaseVelocitySet::Q() - 1) * block::stride()];

        const label_t tid = device::idxBlock();

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS<true>()> moments;
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                const label_t ID = tid * m_i<NUMBER_MOMENTS<true>() + 1>() + m_i<moment>();
                shared_buffer[ID] = devPtrs.ptr<moment>()[idx];
                if constexpr (moment == index::rho())
                {
                    moments[moment] = shared_buffer[ID] + rho0<scalar_t>();
                }
                else
                {
                    moments[moment] = shared_buffer[ID];
                }
            });

        __syncthreads();

        // Reconstruct the populations from the moments
        thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g = PhaseVelocitySet::reconstruct(moments);

        // Gather current phase field state
        const scalar_t phi_ = moments[m_i<10>()];

        // Add sharpening (compressive term)
        PhaseVelocitySet::sharpen(pop_g, phi_, normx_, normy_, normz_);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            streaming::save<VelocitySet>(pop, shared_buffer, tid);
            streaming::save<PhaseVelocitySet>(pop_g, shared_buffer_g, tid);

            __syncthreads();

            // Pull from shared memory
            streaming::pull<VelocitySet>(pop, shared_buffer);
            streaming::phase_pull(pop_g, shared_buffer_g);
        }

        // Load hydro pop from global memory in cover nodes
        HydroHalo::load(pop, fGhostHydro);

        // Load phase pop from global memory in cover nodes
        PhaseHalo::load(pop_g, fGhostPhase);

        // Compute post-stream moments
        velocitySet::calculate_moments<VelocitySet>(pop, moments);
        PhaseVelocitySet::calculate_phi(pop_g, moments);
        {
            // Update the shared buffer with the refreshed moments
            device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
                [&](const auto moment)
                {
                    const label_t ID = tid * label_constant<NUMBER_MOMENTS<true>() + 1>() + label_constant<moment>();
                    shared_buffer[ID] = moments[moment];
                });
        }

        __syncthreads();

        // Calculate the moments at the boundary
        {
            const normalVector boundaryNormal;

            if (boundaryNormal.isBoundary())
            {
                boundaryConditions::calculate_moments<VelocitySet, PhaseVelocitySet>(pop, moments, boundaryNormal, shared_buffer);
            }
        }

        // Coalesced write to global memory
        moments[m_i<0>()] = moments[m_i<0>()] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments[moment];
            });
    }

    /**
     * @brief Compute phase field interface normals and indicator
     * @param phi Pointer to phase field scalar
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
     * @param ind Pointer to interface indicator
     **/
    launchBoundsD3Q19 __global__ void computeNormals(
        const scalar_t *const ptrRestrict phi,
        scalar_t *const ptrRestrict normx,
        scalar_t *const ptrRestrict normy,
        scalar_t *const ptrRestrict normz,
        scalar_t *const ptrRestrict ind)
    {
        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        if (x == 0 || x == device::nx - 1 ||
            y == 0 || y == device::ny - 1 ||
            z == 0 || z == device::nz - 1)
        {
            return;
        }

        // const label_t idx = device::idx();
        const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

        const scalar_t phi_xp1_yp1_z = phi[device::idxGlobalFromIdx(x + 1, y + 1, z)];
        const scalar_t phi_xp1_y_zp1 = phi[device::idxGlobalFromIdx(x + 1, y, z + 1)];
        const scalar_t phi_xp1_ym1_z = phi[device::idxGlobalFromIdx(x + 1, y - 1, z)];
        const scalar_t phi_xp1_y_zm1 = phi[device::idxGlobalFromIdx(x + 1, y, z - 1)];
        const scalar_t phi_xm1_ym1_z = phi[device::idxGlobalFromIdx(x - 1, y - 1, z)];
        const scalar_t phi_xm1_y_zm1 = phi[device::idxGlobalFromIdx(x - 1, y, z - 1)];
        const scalar_t phi_xm1_yp1_z = phi[device::idxGlobalFromIdx(x - 1, y + 1, z)];
        const scalar_t phi_xm1_y_zp1 = phi[device::idxGlobalFromIdx(x - 1, y, z + 1)];
        const scalar_t phi_x_yp1_zp1 = phi[device::idxGlobalFromIdx(x, y + 1, z + 1)];
        const scalar_t phi_x_yp1_zm1 = phi[device::idxGlobalFromIdx(x, y + 1, z - 1)];
        const scalar_t phi_x_ym1_zm1 = phi[device::idxGlobalFromIdx(x, y - 1, z - 1)];
        const scalar_t phi_x_ym1_zp1 = phi[device::idxGlobalFromIdx(x, y - 1, z + 1)];

        const scalar_t sgx = VelocitySet::w_1<scalar_t>() * (phi[device::idxGlobalFromIdx(x + 1, y, z)] - phi[device::idxGlobalFromIdx(x - 1, y, z)]) +
                             VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                             phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                             phi_xp1_ym1_z - phi_xm1_yp1_z +
                                                             phi_xp1_y_zm1 - phi_xm1_y_zp1);

        const scalar_t sgy = VelocitySet::w_1<scalar_t>() * (phi[device::idxGlobalFromIdx(x, y + 1, z)] - phi[device::idxGlobalFromIdx(x, y - 1, z)]) +
                             VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                             phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                             phi_xm1_yp1_z - phi_xp1_ym1_z +
                                                             phi_x_yp1_zm1 - phi_x_ym1_zp1);

        const scalar_t sgz = VelocitySet::w_1<scalar_t>() * (phi[device::idxGlobalFromIdx(x, y, z + 1)] - phi[device::idxGlobalFromIdx(x, y, z - 1)]) +
                             VelocitySet::w_2<scalar_t>() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                             phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                             phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                                             phi_x_ym1_zp1 - phi_x_yp1_zm1);

        const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
        const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
        const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

        const scalar_t ind_ = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind_ + static_cast<scalar_t>(1e-9));

        const scalar_t normx_ = gx * invInd;
        const scalar_t normy_ = gy * invInd;
        const scalar_t normz_ = gz * invInd;

        ind[idx] = ind_;
        normx[idx] = normx_;
        normy[idx] = normy_;
        normz[idx] = normz_;
    }

    /**
     * @brief Performs the collision step of the lattice Boltzmann method using the multiphase moment representation (D3Q19 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param fBlockHalo Object containing pointers to the block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q19 __global__ void multiphaseCollide(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const scalar_t *const ptrRestrict ffx,
        const scalar_t *const ptrRestrict ffy,
        const scalar_t *const ptrRestrict ffz,
        const scalar_t *const ptrRestrict normx,
        const scalar_t *const ptrRestrict normy,
        const scalar_t *const ptrRestrict normz,
        const device::ptrCollection<6, const scalar_t> fGhostHydro,
        const device::ptrCollection<6, scalar_t> gGhostHydro,
        const device::ptrCollection<6, const scalar_t> fGhostPhase,
        const device::ptrCollection<6, scalar_t> gGhostPhase)
    {
        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds())
            {
                return;
            }
        }

        const label_t idx = device::idx();

        scalar_t ffx_ = static_cast<scalar_t>(0);
        scalar_t ffy_ = static_cast<scalar_t>(0);
        scalar_t ffz_ = static_cast<scalar_t>(0);
        scalar_t normx_ = static_cast<scalar_t>(0);
        scalar_t normy_ = static_cast<scalar_t>(0);
        scalar_t normz_ = static_cast<scalar_t>(0);
        scalar_t ind_ = static_cast<scalar_t>(0);

        {
            __shared__ scalar_t sh_phi[block::nz() + 4][block::ny() + 4][block::nx() + 4];
            __shared__ scalar_t shared_normx[block::nz() + 2][block::ny() + 2][block::nx() + 2];
            __shared__ scalar_t shared_normy[block::nz() + 2][block::ny() + 2][block::nx() + 2];
            __shared__ scalar_t shared_normz[block::nz() + 2][block::ny() + 2][block::nx() + 2];

            const label_t x0 = blockIdx.x * block::nx();
            const label_t y0 = blockIdx.y * block::ny();
            const label_t z0 = blockIdx.z * block::nz();

            const scalar_t *const ptrRestrict phi = devPtrs.ptr<10>();

            // Strict bulk-only: we require the full phi tile [x0-2 .. x0+nx()+1] etc to be inside domain.
            const bool stBlock =
                (x0 > 1) && (y0 > 1) && (z0 > 1) &&
                (x0 + block::nx() < device::nx - 1) &&
                (y0 + block::ny() < device::ny - 1) &&
                (z0 + block::nz() < device::nz - 1);

            if (!stBlock)
            {
                ffx_ = static_cast<scalar_t>(0);
                ffy_ = static_cast<scalar_t>(0);
                ffz_ = static_cast<scalar_t>(0);
                normx_ = static_cast<scalar_t>(0);
                normy_ = static_cast<scalar_t>(0);
                normz_ = static_cast<scalar_t>(0);
                ind_ = static_cast<scalar_t>(0);
            }
            else
            {
                // ------------------------------------------------------------
                // Load phi into shared: sh_phi[pz][py][px] corresponds to (x0+px-2, y0+py-2, z0+pz-2)
                // ------------------------------------------------------------
                for (label_t pz = threadIdx.z; pz < block::nz() + 4; pz += block::nz())
                {
                    const label_t gz = z0 + pz - 2;

                    for (label_t py = threadIdx.y; py < block::ny() + 4; py += block::ny())
                    {
                        const label_t gy = y0 + py - 2;

                        for (label_t px = threadIdx.x; px < block::nx() + 4; px += block::nx())
                        {
                            const label_t gx = x0 + px - 2;

                            sh_phi[pz][py][px] = phi[device::idxGlobalFromIdx(gx, gy, gz)];
                        }
                    }
                }

                __syncthreads();

                // ------------------------------------------------------------
                // Compute normals (with halos) from shared phi using the isotropic stencil only.
                // No boundary handling (guaranteed safe by stBlock).
                // ------------------------------------------------------------
                for (label_t iz = threadIdx.z; iz < block::nz() + 2; iz += block::nz())
                {
                    const label_t pz = iz + 1;

                    for (label_t iy = threadIdx.y; iy < block::ny() + 2; iy += block::ny())
                    {
                        const label_t py = iy + 1;

                        for (label_t ix = threadIdx.x; ix < block::nx() + 2; ix += block::nx())
                        {
                            const label_t px = ix + 1;

                            const scalar_t sgx =
                                VelocitySet::w_1<scalar_t>() * (sh_phi[pz][py][px + 1] - sh_phi[pz][py][px - 1]) +
                                VelocitySet::w_2<scalar_t>() * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                                                sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                                                sh_phi[pz][py - 1][px + 1] - sh_phi[pz][py + 1][px - 1] +
                                                                sh_phi[pz - 1][py][px + 1] - sh_phi[pz + 1][py][px - 1]);

                            const scalar_t sgy =
                                VelocitySet::w_1<scalar_t>() * (sh_phi[pz][py + 1][px] - sh_phi[pz][py - 1][px]) +
                                VelocitySet::w_2<scalar_t>() * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                                                sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                                                sh_phi[pz][py + 1][px - 1] - sh_phi[pz][py - 1][px + 1] +
                                                                sh_phi[pz - 1][py + 1][px] - sh_phi[pz + 1][py - 1][px]);

                            const scalar_t sgz =
                                VelocitySet::w_1<scalar_t>() * (sh_phi[pz + 1][py][px] - sh_phi[pz - 1][py][px]) +
                                VelocitySet::w_2<scalar_t>() * (sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                                                sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                                                sh_phi[pz + 1][py][px - 1] - sh_phi[pz - 1][py][px + 1] +
                                                                sh_phi[pz + 1][py - 1][px] - sh_phi[pz - 1][py + 1][px]);

                            const scalar_t gxv = velocitySet::as2<scalar_t>() * sgx;
                            const scalar_t gyv = velocitySet::as2<scalar_t>() * sgy;
                            const scalar_t gzv = velocitySet::as2<scalar_t>() * sgz;

                            const scalar_t ind2 = gxv * gxv + gyv * gyv + gzv * gzv;
                            const scalar_t invInd = static_cast<scalar_t>(1) / (sqrtf(ind2) + static_cast<scalar_t>(1e-9));

                            shared_normx[iz][iy][ix] = gxv * invInd;
                            shared_normy[iz][iy][ix] = gyv * invInd;
                            shared_normz[iz][iy][ix] = gzv * invInd;
                        }
                    }
                }

                __syncthreads();

                // ------------------------------------------------------------
                // Curvature + force for this thread (all threads in this block are bulk by construction).
                // ind_ is scalar (not shared).
                // ------------------------------------------------------------
                {
                    const label_t ix = threadIdx.x + 1;
                    const label_t iy = threadIdx.y + 1;
                    const label_t iz = threadIdx.z + 1;

                    normx_ = shared_normx[iz][iy][ix];
                    normy_ = shared_normy[iz][iy][ix];
                    normz_ = shared_normz[iz][iy][ix];

                    const scalar_t scx =
                        VelocitySet::w_1<scalar_t>() * (shared_normx[iz][iy][ix + 1] - shared_normx[iz][iy][ix - 1]) +
                        VelocitySet::w_2<scalar_t>() * (shared_normx[iz][iy + 1][ix + 1] - shared_normx[iz][iy - 1][ix - 1] +
                                                        shared_normx[iz + 1][iy][ix + 1] - shared_normx[iz - 1][iy][ix - 1] +
                                                        shared_normx[iz][iy - 1][ix + 1] - shared_normx[iz][iy + 1][ix - 1] +
                                                        shared_normx[iz - 1][iy][ix + 1] - shared_normx[iz + 1][iy][ix - 1]);

                    const scalar_t scy =
                        VelocitySet::w_1<scalar_t>() * (shared_normy[iz][iy + 1][ix] - shared_normy[iz][iy - 1][ix]) +
                        VelocitySet::w_2<scalar_t>() * (shared_normy[iz][iy + 1][ix + 1] - shared_normy[iz][iy - 1][ix - 1] +
                                                        shared_normy[iz + 1][iy + 1][ix] - shared_normy[iz - 1][iy - 1][ix] +
                                                        shared_normy[iz][iy + 1][ix - 1] - shared_normy[iz][iy - 1][ix + 1] +
                                                        shared_normy[iz - 1][iy + 1][ix] - shared_normy[iz + 1][iy - 1][ix]);

                    const scalar_t scz =
                        VelocitySet::w_1<scalar_t>() * (shared_normz[iz + 1][iy][ix] - shared_normz[iz - 1][iy][ix]) +
                        VelocitySet::w_2<scalar_t>() * (shared_normz[iz + 1][iy][ix + 1] - shared_normz[iz - 1][iy][ix - 1] +
                                                        shared_normz[iz + 1][iy + 1][ix] - shared_normz[iz - 1][iy - 1][ix] +
                                                        shared_normz[iz + 1][iy][ix - 1] - shared_normz[iz - 1][iy][ix + 1] +
                                                        shared_normz[iz + 1][iy - 1][ix] - shared_normz[iz - 1][iy + 1][ix]);

                    const scalar_t curvature = velocitySet::as2<scalar_t>() * (scx + scy + scz);

                    // ind_ as scalar: compute |grad(phi)| at cell center from shared phi.
                    {
                        const label_t cx = threadIdx.x + 2;
                        const label_t cy = threadIdx.y + 2;
                        const label_t cz = threadIdx.z + 2;

                        const scalar_t sgx =
                            VelocitySet::w_1<scalar_t>() * (sh_phi[cz][cy][cx + 1] - sh_phi[cz][cy][cx - 1]) +
                            VelocitySet::w_2<scalar_t>() * (sh_phi[cz][cy + 1][cx + 1] - sh_phi[cz][cy - 1][cx - 1] +
                                                            sh_phi[cz + 1][cy][cx + 1] - sh_phi[cz - 1][cy][cx - 1] +
                                                            sh_phi[cz][cy - 1][cx + 1] - sh_phi[cz][cy + 1][cx - 1] +
                                                            sh_phi[cz - 1][cy][cx + 1] - sh_phi[cz + 1][cy][cx - 1]);

                        const scalar_t sgy =
                            VelocitySet::w_1<scalar_t>() * (sh_phi[cz][cy + 1][cx] - sh_phi[cz][cy - 1][cx]) +
                            VelocitySet::w_2<scalar_t>() * (sh_phi[cz][cy + 1][cx + 1] - sh_phi[cz][cy - 1][cx - 1] +
                                                            sh_phi[cz + 1][cy + 1][cx] - sh_phi[cz - 1][cy - 1][cx] +
                                                            sh_phi[cz][cy + 1][cx - 1] - sh_phi[cz][cy - 1][cx + 1] +
                                                            sh_phi[cz - 1][cy + 1][cx] - sh_phi[cz + 1][cy - 1][cx]);

                        const scalar_t sgz =
                            VelocitySet::w_1<scalar_t>() * (sh_phi[cz + 1][cy][cx] - sh_phi[cz - 1][cy][cx]) +
                            VelocitySet::w_2<scalar_t>() * (sh_phi[cz + 1][cy][cx + 1] - sh_phi[cz - 1][cy][cx - 1] +
                                                            sh_phi[cz + 1][cy + 1][cx] - sh_phi[cz - 1][cy - 1][cx] +
                                                            sh_phi[cz + 1][cy][cx - 1] - sh_phi[cz - 1][cy][cx + 1] +
                                                            sh_phi[cz + 1][cy - 1][cx] - sh_phi[cz - 1][cy + 1][cx]);

                        const scalar_t gxv = velocitySet::as2<scalar_t>() * sgx;
                        const scalar_t gyv = velocitySet::as2<scalar_t>() * sgy;
                        const scalar_t gzv = velocitySet::as2<scalar_t>() * sgz;

                        ind_ = sqrtf(gxv * gxv + gyv * gyv + gzv * gzv);
                    }

                    const scalar_t stCurv = -device::sigma * curvature * ind_;

                    ffx_ = stCurv * normx_;
                    ffy_ = stCurv * normy_;
                    ffz_ = stCurv * normz_;
                }
            }
        }

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS<true>()> moments;
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                if constexpr (moment == index::rho())
                {
                    moments[moment] = devPtrs.ptr<moment>()[idx] + rho0<scalar_t>();
                }
                else
                {
                    moments[moment] = devPtrs.ptr<moment>()[idx];
                }
            });

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments, ffx_, ffy_, ffz_);

        // Calculate post collision populations
        thread::array<scalar_t, VelocitySet::Q()> pop;
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g;
        VelocitySet::reconstruct(pop, moments);
        PhaseVelocitySet::reconstruct(pop_g, moments);

        // Gather current phase field state
        const scalar_t phi_ = moments[m_i<10>()];

        // Add sharpening (compressive term)
        PhaseVelocitySet::sharpen(pop_g, phi_, normx_, normy_, normz_);

        // Coalesced write to global memory
        moments[m_i<0>()] = moments[m_i<0>()] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments[moment];
            });

        // Save the hydro populations to the block halo
        HydroHalo::save(pop, gGhostHydro);

        // Save the phase populations to the block halo
        PhaseHalo::save(pop_g, gGhostPhase);
    }
}

#endif