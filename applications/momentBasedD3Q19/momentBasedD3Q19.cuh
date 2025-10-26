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
    Main kernel for the moment representation with the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    momentBasedD3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDD3Q19_CUH
#define __MBLBM_MOMENTBASEDD3Q19_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/streaming/streaming.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/functionObjects/objectRegistry.cuh"

namespace LBM
{

    using VelocitySet = D3Q19;
    using Collision = secondOrder;

    __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }
#define launchBoundsD3Q19 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param devPtrs Collection of 10 pointers to device arrays on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBoundsD3Q19 __global__ void momentBasedD3Q19(
        const device::ptrCollection<10, scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> fGhost,
        const device::ptrCollection<6, scalar_t> gGhost)
    {
        // Always a multiple of 32, so no need to check this(I think)
        // if (device::out_of_bounds())
        // {
        //     return;
        // }

        // const label_t idx = device::idx();
        const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Declare shared memory (flattened)
        // __shared__ thread::array<scalar_t, block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS()>()> shared_buffer;
        extern __shared__ scalar_t shared_buffer[];

        const label_t tid = device::idxBlock();

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS()> moments;
        {
            // Read into shared
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    const label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
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

        // Reconstruct the population from the moments
        thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            streaming::save<VelocitySet>(pop, shared_buffer, tid);

            __syncthreads();

            // Pull from shared memory
            streaming::pull<VelocitySet>(pop, shared_buffer);
        }

        // Load pop from global memory in cover nodes
        device::halo<VelocitySet>::load(
            pop,
            fGhost.ptr<0>(),
            fGhost.ptr<1>(),
            fGhost.ptr<2>(),
            fGhost.ptr<3>(),
            fGhost.ptr<4>(),
            fGhost.ptr<5>());

        /* =============================== BRENO: =============================== */
        /* Reconstruct post-stream moments into shared buffer for boundary access */
        
        // Compute post-stream/halo moments from populations
        VelocitySet::calculateMoments(pop, moments);
        {
            // Update the shared buffer with the refreshed moment fields 
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    const label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
                    shared_buffer[ID] = moments[moment]; 
                });
        }

        __syncthreads();

        /* ====================================================================== */  


        // Calculate the moments either at the boundary or interior
        {
            const normalVector boundaryNormal;
            if (boundaryNormal.isBoundary())
            {
                boundaryConditions::calculateMoments<VelocitySet>(pop, moments, boundaryNormal, shared_buffer);
            }
            else
            {
                VelocitySet::calculateMoments(pop, moments);
            }
        }

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments);

        // Calculate post collision populations
        VelocitySet::reconstruct(pop, moments);

        // Coalesced write to global memory
        moments[0] = moments[0] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments[moment];
            });

        // Save the populations to the block halo
        device::halo<VelocitySet>::save(
            pop,
            gGhost.ptr<0>(),
            gGhost.ptr<1>(),
            gGhost.ptr<2>(),
            gGhost.ptr<3>(),
            gGhost.ptr<4>(),
            gGhost.ptr<5>());
    }
}

#endif