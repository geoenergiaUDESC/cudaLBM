/**
Filename: momentBasedD3Q19.cuh
Contents: Main kernel for the moment representation with the D3Q19 velocity set
**/

#ifndef __MBLBM_MOMENTBASEDD319_CUH
#define __MBLBM_MOMENTBASEDD319_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/streaming/streaming.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"

namespace LBM
{

    using VelocitySet = D3Q19;
    using Collision = secondOrder;

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param devPtrs Collection of 10 pointers to device arrays on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBounds __global__ void momentBasedD3Q19(
        const device::ptrCollection<10, scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> fGhost,
        const device::ptrCollection<6, scalar_t> gGhost)
    {
        // Always a multiple of 32, so no need to check this(I think)
        // if (device::out_of_bounds())
        // {
        //     return;
        // }

        const label_t idx = device::idx();

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Declare shared memory (flattened)
        __shared__ thread::array<scalar_t, block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS()>()> shared_buffer;

        const label_t tid = device::idxBlock();

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS()> moments;
        {
            // Read into shared
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    shared_buffer[moment * block::stride() + tid] = devPtrs.ptr<moment>()[idx];
                    if constexpr (moment == index::rho())
                    {
                        moments[moment] = shared_buffer[moment * block::stride() + tid] + rho0<scalar_t>();
                    }
                    else
                    {
                        moments[moment] = shared_buffer[moment * block::stride() + tid];
                    }
                });
        }

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

        // Calculate the moments either at the boundary or interior
        {
            const normalVector boundaryNormal;
            if (boundaryNormal.isBoundary())
            {
                boundaryConditions::calculateMoments<VelocitySet>(pop, moments, boundaryNormal);
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