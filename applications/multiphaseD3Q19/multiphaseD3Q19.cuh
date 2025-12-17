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
    along with this program. If not, see <https://www.gnu.org/licenses/>.

Description
    Main kernel for the multiphase moment representation with the D3Q19 velocity set

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

    // Templated bool: isMultiphase
    using HydroHalo = device::halo<VelocitySet, config::periodicX, config::periodicY>;
    using PhaseHalo = device::halo<PhaseVelocitySet, config::periodicX, config::periodicY>;

    __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }
#define launchBoundsD3Q19 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param devPtrs Collection of NUMBER_MOMENTS<false>() pointers to device arrays on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBoundsD3Q19 __global__ void multiphaseD3Q19(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> fGhostHydro,
        const device::ptrCollection<6, scalar_t> gGhostHydro,
        const device::ptrCollection<6, const scalar_t> fGhostPhase,
        const device::ptrCollection<6, scalar_t> gGhostPhase)
    {
        // Always a multiple of 32, so no need to check this(I think)
        // if (device::out_of_bounds())
        // {
        //     return;
        // }

        // const label_t idx = device::idx();
        const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

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
        {
            // Read into shared
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
        }

        __syncthreads();

        // Phase-field snapshot at time level n stored in shared memory for stencil operations
        // __shared__ scalar_t shared_phi[block::stride()];
        // shared_phi[tid] = moments[m_i<10>()];

        // ======================================== LBM routines start below ======================================== //

        // Reconstruct the populations from the moments
        thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct<true>(moments);
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g = PhaseVelocitySet::reconstruct<true>(moments);

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
        HydroHalo::load(
            pop,
            fGhostHydro.ptr<0>(),
            fGhostHydro.ptr<1>(),
            fGhostHydro.ptr<2>(),
            fGhostHydro.ptr<3>(),
            fGhostHydro.ptr<4>(),
            fGhostHydro.ptr<5>());

        // Load phase pop from global memory in cover nodes
        PhaseHalo::load(
            pop_g,
            fGhostPhase.ptr<0>(),
            fGhostPhase.ptr<1>(),
            fGhostPhase.ptr<2>(),
            fGhostPhase.ptr<3>(),
            fGhostPhase.ptr<4>(),
            fGhostPhase.ptr<5>());

        // Compute post-stream moments
        VelocitySet::calculateMoments<true>(pop, moments);
        PhaseVelocitySet::calculateMoments<true>(pop_g, moments);
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

        // Calculate the moments either at the boundary or interior
        {
            const normalVector boundaryNormal;
            if (boundaryNormal.isBoundary())
            {
                boundaryConditions::calculateMoments<VelocitySet, PhaseVelocitySet>(pop, pop_g, moments, boundaryNormal, shared_buffer);
            }
            else
            {
                VelocitySet::calculateMoments<true>(pop, moments);
                PhaseVelocitySet::calculateMoments<true>(pop_g, moments);
            }
        }

        // Scale the moments correctly
        velocitySet::scale<true>(moments);

        // ======================================================================================================================== //
        // ======================================================================================================================== //
        // ======================================================================================================================== //

        // Collide
        Collision::collide<true>(moments);

        // Calculate post collision populations
        VelocitySet::reconstruct<true>(pop, moments);
        PhaseVelocitySet::reconstruct<true>(pop_g, moments);

        // Coalesced write to global memory
        moments[m_i<0>()] = moments[m_i<0>()] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments[moment];
            });

        // ======================================================================================================================== //
        // ======================================================================================================================== //
        // ======================================================================================================================== //

        // Save the hydro populations to the block halo
        HydroHalo::save(
            pop,
            gGhostHydro.ptr<0>(),
            gGhostHydro.ptr<1>(),
            gGhostHydro.ptr<2>(),
            gGhostHydro.ptr<3>(),
            gGhostHydro.ptr<4>(),
            gGhostHydro.ptr<5>());

        // Save the phase populations to the block halo
        PhaseHalo::save(
            pop_g,
            gGhostPhase.ptr<0>(),
            gGhostPhase.ptr<1>(),
            gGhostPhase.ptr<2>(),
            gGhostPhase.ptr<3>(),
            gGhostPhase.ptr<4>(),
            gGhostPhase.ptr<5>());

        // ============================================ LBM routines end ============================================ //
    }
}

#endif