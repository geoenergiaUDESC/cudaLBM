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

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
     * @param fBlockHalo Object containing pointers to the block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q19 __global__ void multiphaseStream(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
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

        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        const bool isInterior =
            (x > 0) & (x < device::nx - 1) &
            (y > 0) & (y < device::ny - 1) &
            (z > 0) & (z < device::nz - 1);

        const label_t idx = device::idx();

        scalar_t normx_ = static_cast<scalar_t>(0);
        scalar_t normy_ = static_cast<scalar_t>(0);
        scalar_t normz_ = static_cast<scalar_t>(0);

        const scalar_t *__restrict__ phi = devPtrs.ptr<10>();

        if (isInterior)
        {
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

            normx_ = gx * invInd;
            normy_ = gy * invInd;
            normz_ = gz * invInd;
        }

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

        // Add sharpening (compressive term) on g-populations
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
     * @brief Performs the collision step of the lattice Boltzmann method using the multiphase moment representation (D3Q19 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param fBlockHalo Object containing pointers to the block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q19 __global__ void multiphaseCollide(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
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
        // const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

        // // --- Load center normal for sharpening (ALWAYS) ---
        // const scalar_t normx_ = normx[idx];
        // const scalar_t normy_ = normy[idx];
        // const scalar_t normz_ = normz[idx];

        // // --- Bulk-only surface tension force from curvature (global normals) ---
        // scalar_t ffx_ = scalar_t(0);
        // scalar_t ffy_ = scalar_t(0);
        // scalar_t ffz_ = scalar_t(0);

        // {
        //     const int x = int(threadIdx.x) + int(blockIdx.x) * int(blockDim.x);
        //     const int y = int(threadIdx.y) + int(blockIdx.y) * int(blockDim.y);
        //     const int z = int(threadIdx.z) + int(blockIdx.z) * int(blockDim.z);

        //     // Bulk-only (exclude boundary layer). If grid overruns domain, also keep force zero.
        //     const bool bulkInterior =
        //         (x > 0 && x < device::nx - 1) &&
        //         (y > 0 && y < device::ny - 1) &&
        //         (z > 0 && z < device::nz - 1);

        //     if (bulkInterior)
        //     {
        //         // Neighbor indices (same stencil as computeForces)
        //         const label_t xp = device::idxGlobalFromIdx(x + 1, y, z);
        //         const label_t xm = device::idxGlobalFromIdx(x - 1, y, z);
        //         const label_t yp = device::idxGlobalFromIdx(x, y + 1, z);
        //         const label_t ym = device::idxGlobalFromIdx(x, y - 1, z);
        //         const label_t zp = device::idxGlobalFromIdx(x, y, z + 1);
        //         const label_t zm = device::idxGlobalFromIdx(x, y, z - 1);

        //         const label_t xp1_yp1_z = device::idxGlobalFromIdx(x + 1, y + 1, z);
        //         const label_t xp1_y_zp1 = device::idxGlobalFromIdx(x + 1, y, z + 1);
        //         const label_t xp1_ym1_z = device::idxGlobalFromIdx(x + 1, y - 1, z);
        //         const label_t xp1_y_zm1 = device::idxGlobalFromIdx(x + 1, y, z - 1);

        //         const label_t xm1_ym1_z = device::idxGlobalFromIdx(x - 1, y - 1, z);
        //         const label_t xm1_y_zm1 = device::idxGlobalFromIdx(x - 1, y, z - 1);
        //         const label_t xm1_yp1_z = device::idxGlobalFromIdx(x - 1, y + 1, z);
        //         const label_t xm1_y_zp1 = device::idxGlobalFromIdx(x - 1, y, z + 1);

        //         const label_t x_yp1_zp1 = device::idxGlobalFromIdx(x, y + 1, z + 1);
        //         const label_t x_yp1_zm1 = device::idxGlobalFromIdx(x, y + 1, z - 1);
        //         const label_t x_ym1_zm1 = device::idxGlobalFromIdx(x, y - 1, z - 1);
        //         const label_t x_ym1_zp1 = device::idxGlobalFromIdx(x, y - 1, z + 1);

        //         const scalar_t w1 = VelocitySet::w_1<scalar_t>();
        //         const scalar_t w2 = VelocitySet::w_2<scalar_t>();

        //         // Divergence of normals (exactly your computeForces stencil)
        //         const scalar_t scx =
        //             w1 * (normx[xp] - normx[xm]) +
        //             w2 * (normx[xp1_yp1_z] - normx[xm1_ym1_z] +
        //                   normx[xp1_y_zp1] - normx[xm1_y_zm1] +
        //                   normx[xp1_ym1_z] - normx[xm1_yp1_z] +
        //                   normx[xp1_y_zm1] - normx[xm1_y_zp1]);

        //         const scalar_t scy =
        //             w1 * (normy[yp] - normy[ym]) +
        //             w2 * (normy[xp1_yp1_z] - normy[xm1_ym1_z] +
        //                   normy[x_yp1_zp1] - normy[x_ym1_zm1] +
        //                   normy[xm1_yp1_z] - normy[xp1_ym1_z] +
        //                   normy[x_yp1_zm1] - normy[x_ym1_zp1]);

        //         const scalar_t scz =
        //             w1 * (normz[zp] - normz[zm]) +
        //             w2 * (normz[xp1_y_zp1] - normz[xm1_y_zm1] +
        //                   normz[x_yp1_zp1] - normz[x_ym1_zm1] +
        //                   normz[xm1_y_zp1] - normz[xp1_y_zm1] +
        //                   normz[x_ym1_zp1] - normz[x_yp1_zm1]);

        //         const scalar_t curvature = velocitySet::as2<scalar_t>() * (scx + scy + scz);

        //         // Your model uses ind at the center (must be precomputed globally)
        //         const scalar_t ind_c = ind[idx];

        //         const scalar_t stCurv = -device::sigma * curvature * ind_c;
        //         ffx_ = stCurv * normx_;
        //         ffy_ = stCurv * normy_;
        //         ffz_ = stCurv * normz_;
        //     }
        // }

        scalar_t ffx_ = scalar_t(0);
        scalar_t ffy_ = scalar_t(0);
        scalar_t ffz_ = scalar_t(0);
        scalar_t normx_ = scalar_t(0);
        scalar_t normy_ = scalar_t(0);
        scalar_t normz_ = scalar_t(0);

        {
            constexpr int BX = 8, BY = 8, BZ = 8;

            __shared__ scalar_t sh_phi[BZ + 4][BY + 4][BX + 4]; // halo-2
            __shared__ scalar_t sh_nx[BZ + 2][BY + 2][BX + 2];  // halo-1
            __shared__ scalar_t sh_ny[BZ + 2][BY + 2][BX + 2];
            __shared__ scalar_t sh_nz[BZ + 2][BY + 2][BX + 2];
            __shared__ scalar_t sh_ind[BZ + 2][BY + 2][BX + 2];

            const int tx = int(threadIdx.x);
            const int ty = int(threadIdx.y);
            const int tz = int(threadIdx.z);

            const int x = tx + int(blockIdx.x) * int(blockDim.x);
            const int y = ty + int(blockIdx.y) * int(blockDim.y);
            const int z = tz + int(blockIdx.z) * int(blockDim.z);

            const int NX = int(device::nx);
            const int NY = int(device::ny);
            const int NZ = int(device::nz);

            // Enforce compile-time tile dims == launch dims (uniform across block => safe)
            if ((int(blockDim.x) != BX) | (int(blockDim.y) != BY) | (int(blockDim.z) != BZ))
            {
                return; // safe because no __syncthreads() happened yet
            }

            const int x0 = int(blockIdx.x) * BX;
            const int y0 = int(blockIdx.y) * BY;
            const int z0 = int(blockIdx.z) * BZ;

            auto in_domain = [&](int gx, int gy, int gz) -> bool
            {
                return (unsigned)gx < (unsigned)NX &&
                       (unsigned)gy < (unsigned)NY &&
                       (unsigned)gz < (unsigned)NZ;
            };

            auto gidx = [&](int gx, int gy, int gz) -> label_t
            {
                return device::idxGlobalFromIdx(label_t(gx), label_t(gy), label_t(gz));
            };

            const scalar_t *__restrict__ phi = devPtrs.ptr<10>();

            // ---------------------- 1) Load phi tile with halo-2 (no OOB reads) ----------------------
            for (int pz = tz; pz < (BZ + 4); pz += BZ)
            {
                const int gz = z0 + (pz - 2);
                for (int py = ty; py < (BY + 4); py += BY)
                {
                    const int gy = y0 + (py - 2);
                    for (int px = tx; px < (BX + 4); px += BX)
                    {
                        const int gx = x0 + (px - 2);
                        sh_phi[pz][py][px] = in_domain(gx, gy, gz) ? phi[gidx(gx, gy, gz)] : scalar_t(0);
                    }
                }
            }

            __syncthreads();

            // Helpers: 2nd-order one-sided / centered derivatives from sh_phi
            auto ddx = [&](int gx, int pz, int py, int px) -> scalar_t
            {
                if (gx == 0)
                    return scalar_t(0.5) * (-3 * sh_phi[pz][py][px] + 4 * sh_phi[pz][py][px + 1] - sh_phi[pz][py][px + 2]);
                if (gx == NX - 1)
                    return scalar_t(0.5) * (3 * sh_phi[pz][py][px] - 4 * sh_phi[pz][py][px - 1] + sh_phi[pz][py][px - 2]);
                return scalar_t(0.5) * (sh_phi[pz][py][px + 1] - sh_phi[pz][py][px - 1]);
            };
            auto ddy = [&](int gy, int pz, int py, int px) -> scalar_t
            {
                if (gy == 0)
                    return scalar_t(0.5) * (-3 * sh_phi[pz][py][px] + 4 * sh_phi[pz][py + 1][px] - sh_phi[pz][py + 2][px]);
                if (gy == NY - 1)
                    return scalar_t(0.5) * (3 * sh_phi[pz][py][px] - 4 * sh_phi[pz][py - 1][px] + sh_phi[pz][py - 2][px]);
                return scalar_t(0.5) * (sh_phi[pz][py + 1][px] - sh_phi[pz][py - 1][px]);
            };
            auto ddz = [&](int gz, int pz, int py, int px) -> scalar_t
            {
                if (gz == 0)
                    return scalar_t(0.5) * (-3 * sh_phi[pz][py][px] + 4 * sh_phi[pz + 1][py][px] - sh_phi[pz + 2][py][px]);
                if (gz == NZ - 1)
                    return scalar_t(0.5) * (3 * sh_phi[pz][py][px] - 4 * sh_phi[pz - 1][py][px] + sh_phi[pz - 2][py][px]);
                return scalar_t(0.5) * (sh_phi[pz + 1][py][px] - sh_phi[pz - 1][py][px]);
            };

            // ---------------------- 2) Compute normals on (BX+2)^3 tile ----------------------
            const scalar_t w1 = VelocitySet::w_1<scalar_t>();
            const scalar_t w2 = VelocitySet::w_2<scalar_t>();
            constexpr scalar_t eps = scalar_t(1e-18);

            for (int iz = tz; iz < (BZ + 2); iz += BZ)
            {
                const int gz_n = z0 + (iz - 1);
                const int pz = iz + 1;

                for (int iy = ty; iy < (BY + 2); iy += BY)
                {
                    const int gy_n = y0 + (iy - 1);
                    const int py = iy + 1;

                    for (int ix = tx; ix < (BX + 2); ix += BX)
                    {
                        const int gx_n = x0 + (ix - 1);
                        const int px = ix + 1;

                        // Outside domain => set halo normals to zero
                        if (!in_domain(gx_n, gy_n, gz_n))
                        {
                            sh_ind[iz][iy][ix] = scalar_t(0);
                            sh_nx[iz][iy][ix] = scalar_t(0);
                            sh_ny[iz][iy][ix] = scalar_t(0);
                            sh_nz[iz][iy][ix] = scalar_t(0);
                            continue;
                        }

                        scalar_t gxv, gyv, gzv;

                        // Interior (distance >=1): use your isotropic LB gradient
                        const bool normalInterior1 =
                            (gx_n >= 1 && gx_n <= NX - 2) &&
                            (gy_n >= 1 && gy_n <= NY - 2) &&
                            (gz_n >= 1 && gz_n <= NZ - 2);

                        if (normalInterior1)
                        {
                            const scalar_t sgx =
                                w1 * (sh_phi[pz][py][px + 1] - sh_phi[pz][py][px - 1]) +
                                w2 * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                      sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                      sh_phi[pz][py - 1][px + 1] - sh_phi[pz][py + 1][px - 1] +
                                      sh_phi[pz - 1][py][px + 1] - sh_phi[pz + 1][py][px - 1]);

                            const scalar_t sgy =
                                w1 * (sh_phi[pz][py + 1][px] - sh_phi[pz][py - 1][px]) +
                                w2 * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                      sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                      sh_phi[pz][py + 1][px - 1] - sh_phi[pz][py - 1][px + 1] +
                                      sh_phi[pz - 1][py + 1][px] - sh_phi[pz + 1][py - 1][px]);

                            const scalar_t sgz =
                                w1 * (sh_phi[pz + 1][py][px] - sh_phi[pz - 1][py][px]) +
                                w2 * (sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                      sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                      sh_phi[pz + 1][py][px - 1] - sh_phi[pz - 1][py][px + 1] +
                                      sh_phi[pz + 1][py - 1][px] - sh_phi[pz - 1][py + 1][px]);

                            gxv = velocitySet::as2<scalar_t>() * sgx;
                            gyv = velocitySet::as2<scalar_t>() * sgy;
                            gzv = velocitySet::as2<scalar_t>() * sgz;
                        }
                        else
                        {
                            // Boundary normals (x==0/NX-1 etc.): one-sided/centered axis derivatives, in-domain only.
                            gxv = ddx(gx_n, pz, py, px);
                            gyv = ddy(gy_n, pz, py, px);
                            gzv = ddz(gz_n, pz, py, px);
                        }

                        const scalar_t ind2 = gxv * gxv + gyv * gyv + gzv * gzv;
                        const scalar_t ind = ::sqrt(ind2);
                        const scalar_t invInd = scalar_t(1) / (ind + eps);

                        sh_ind[iz][iy][ix] = ind;
                        sh_nx[iz][iy][ix] = gxv * invInd;
                        sh_ny[iz][iy][ix] = gyv * invInd;
                        sh_nz[iz][iy][ix] = gzv * invInd;
                    }
                }
            }

            __syncthreads();

            // ---------------------- 3) Curvature + forces (exclude ONLY true boundaries) ----------------------
            // Now boundary normals exist, so curvature is valid for distance>=1.
            const bool curvInterior1 =
                (x >= 1 && x <= NX - 2) &&
                (y >= 1 && y <= NY - 2) &&
                (z >= 1 && z <= NZ - 2);

            if (curvInterior1)
            {
                const int ix = tx + 1;
                const int iy = ty + 1;
                const int iz = tz + 1;

                // Center normals for sharpening (FIXED indexing)
                normx_ = sh_nx[iz][iy][ix];
                normy_ = sh_ny[iz][iy][ix];
                normz_ = sh_nz[iz][iy][ix];

                const scalar_t scx =
                    w1 * (sh_nx[iz][iy][ix + 1] - sh_nx[iz][iy][ix - 1]) +
                    w2 * (sh_nx[iz][iy + 1][ix + 1] - sh_nx[iz][iy - 1][ix - 1] +
                          sh_nx[iz + 1][iy][ix + 1] - sh_nx[iz - 1][iy][ix - 1] +
                          sh_nx[iz][iy - 1][ix + 1] - sh_nx[iz][iy + 1][ix - 1] +
                          sh_nx[iz - 1][iy][ix + 1] - sh_nx[iz + 1][iy][ix - 1]);

                const scalar_t scy =
                    w1 * (sh_ny[iz][iy + 1][ix] - sh_ny[iz][iy - 1][ix]) +
                    w2 * (sh_ny[iz][iy + 1][ix + 1] - sh_ny[iz][iy - 1][ix - 1] +
                          sh_ny[iz + 1][iy + 1][ix] - sh_ny[iz - 1][iy - 1][ix] +
                          sh_ny[iz][iy + 1][ix - 1] - sh_ny[iz][iy - 1][ix + 1] +
                          sh_ny[iz - 1][iy + 1][ix] - sh_ny[iz + 1][iy - 1][ix]);

                const scalar_t scz =
                    w1 * (sh_nz[iz + 1][iy][ix] - sh_nz[iz - 1][iy][ix]) +
                    w2 * (sh_nz[iz + 1][iy][ix + 1] - sh_nz[iz - 1][iy][ix - 1] +
                          sh_nz[iz + 1][iy + 1][ix] - sh_nz[iz - 1][iy - 1][ix] +
                          sh_nz[iz + 1][iy][ix - 1] - sh_nz[iz - 1][iy][ix + 1] +
                          sh_nz[iz + 1][iy - 1][ix] - sh_nz[iz - 1][iy + 1][ix]);

                const scalar_t curvature = velocitySet::as2<scalar_t>() * (scx + scy + scz);

                const scalar_t ind_c = sh_ind[iz][iy][ix];
                const scalar_t stCurv = -device::sigma * curvature * ind_c;

                ffx_ = stCurv * normx_;
                ffy_ = stCurv * normy_;
                ffz_ = stCurv * normz_;
            }
            else
            {
                // True boundary cells only
                normx_ = normy_ = normz_ = scalar_t(0);
                ffx_ = ffy_ = ffz_ = scalar_t(0);
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

        // Add sharpening (compressive term) on g-populations
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