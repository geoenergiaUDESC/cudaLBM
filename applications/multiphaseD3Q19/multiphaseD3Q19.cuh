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
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
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

        // Declare shared memory (flattened)
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
     * @brief Compute surface tension forces
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
     * @param ind Pointer to interface indicator
     * @param ffx Pointer to x-component of the surface tension force
     * @param ffy Pointer to y-component of the surface tension force
     * @param ffz Pointer to z-component of the surface tension force
     **/
    launchBoundsD3Q19 __global__ void computeForces(
        const scalar_t *const ptrRestrict normx,
        const scalar_t *const ptrRestrict normy,
        const scalar_t *const ptrRestrict normz,
        const scalar_t *const ptrRestrict ind,
        scalar_t *const ptrRestrict ffx,
        scalar_t *const ptrRestrict ffy,
        scalar_t *const ptrRestrict ffz)
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

        const label_t xp1_yp1_z = device::idxGlobalFromIdx(x + 1, y + 1, z);
        const label_t xp1_y_zp1 = device::idxGlobalFromIdx(x + 1, y, z + 1);
        const label_t xp1_ym1_z = device::idxGlobalFromIdx(x + 1, y - 1, z);
        const label_t xp1_y_zm1 = device::idxGlobalFromIdx(x + 1, y, z - 1);
        const label_t xm1_ym1_z = device::idxGlobalFromIdx(x - 1, y - 1, z);
        const label_t xm1_y_zm1 = device::idxGlobalFromIdx(x - 1, y, z - 1);
        const label_t xm1_yp1_z = device::idxGlobalFromIdx(x - 1, y + 1, z);
        const label_t xm1_y_zp1 = device::idxGlobalFromIdx(x - 1, y, z + 1);
        const label_t x_yp1_zp1 = device::idxGlobalFromIdx(x, y + 1, z + 1);
        const label_t x_yp1_zm1 = device::idxGlobalFromIdx(x, y + 1, z - 1);
        const label_t x_ym1_zm1 = device::idxGlobalFromIdx(x, y - 1, z - 1);
        const label_t x_ym1_zp1 = device::idxGlobalFromIdx(x, y - 1, z + 1);

        const scalar_t scx = VelocitySet::w_1<scalar_t>() * (normx[device::idxGlobalFromIdx(x + 1, y, z)] - normx[device::idxGlobalFromIdx(x - 1, y, z)]) +
                             VelocitySet::w_2<scalar_t>() * (normx[xp1_yp1_z] - normx[xm1_ym1_z] +
                                                             normx[xp1_y_zp1] - normx[xm1_y_zm1] +
                                                             normx[xp1_ym1_z] - normx[xm1_yp1_z] +
                                                             normx[xp1_y_zm1] - normx[xm1_y_zp1]);

        const scalar_t scy = VelocitySet::w_1<scalar_t>() * (normy[device::idxGlobalFromIdx(x, y + 1, z)] - normy[device::idxGlobalFromIdx(x, y - 1, z)]) +
                             VelocitySet::w_2<scalar_t>() * (normy[xp1_yp1_z] - normy[xm1_ym1_z] +
                                                             normy[x_yp1_zp1] - normy[x_ym1_zm1] +
                                                             normy[xm1_yp1_z] - normy[xp1_ym1_z] +
                                                             normy[x_yp1_zm1] - normy[x_ym1_zp1]);

        const scalar_t scz = VelocitySet::w_1<scalar_t>() * (normz[device::idxGlobalFromIdx(x, y, z + 1)] - normz[device::idxGlobalFromIdx(x, y, z - 1)]) +
                             VelocitySet::w_2<scalar_t>() * (normz[xp1_y_zp1] - normz[xm1_y_zm1] +
                                                             normz[x_yp1_zp1] - normz[x_ym1_zm1] +
                                                             normz[xm1_y_zp1] - normz[xp1_y_zm1] +
                                                             normz[x_ym1_zp1] - normz[x_yp1_zm1]);

        const scalar_t curvature = velocitySet::as2<scalar_t>() * (scx + scy + scz);

        const scalar_t stCurv = -device::sigma * curvature * ind[idx];
        ffx[idx] = stCurv * normx[idx];
        ffy[idx] = stCurv * normy[idx];
        ffz[idx] = stCurv * normz[idx];
    }

    /**
     * @brief Performs the collision step of the lattice Boltzmann method using the multiphase moment representation (D3Q19 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param ffx Pointer to x-component of the surface tension force
     * @param ffy Pointer to y-component of the surface tension force
     * @param ffz Pointer to z-component of the surface tension force
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
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
        // const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

        const scalar_t ffx_ = ffx[idx];
        const scalar_t ffy_ = ffy[idx];
        const scalar_t ffz_ = ffz[idx];
        const scalar_t normx_ = normx[idx];
        const scalar_t normy_ = normy[idx];
        const scalar_t normz_ = normz[idx];

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

    launchBoundsD3Q19 __global__ void multiphaseD3Q19(
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

        const scalar_t ffx_ = ffx[idx];
        const scalar_t ffy_ = ffy[idx];
        const scalar_t ffz_ = ffz[idx];
        const scalar_t normx_ = normx[idx];
        const scalar_t normy_ = normy[idx];
        const scalar_t normz_ = normz[idx];

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

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
        scalar_t phi_ = moments[m_i<10>()];

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

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments, ffx_, ffy_, ffz_);

        // Calculate post collision populations
        VelocitySet::reconstruct(pop, moments);
        PhaseVelocitySet::reconstruct(pop_g, moments);

        // Gather current phase field state
        phi_ = moments[m_i<10>()];

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