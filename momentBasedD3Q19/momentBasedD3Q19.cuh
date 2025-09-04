/**
Filename: momentBasedD3Q19.cuh
Contents: Main kernel for the moment representation with the D3Q19 velocity set
**/

#ifndef __MBLBM_MOMENTBASEDD319_CUH
#define __MBLBM_MOMENTBASEDD319_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"
#include "../collision/collision.cuh"
#include "../blockHalo/blockHalo.cuh"
#include "../fileIO/fileIO.cuh"
#include "../runTimeIO/runTimeIO.cuh"

namespace LBM
{

    using VSet = VelocitySet::D3Q19;
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

        // Prefetch L2
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Declare shared memory (flattened)
        __shared__ scalar_t shared_buffer[block::sharedMemoryBufferSize<VSet, NUMBER_MOMENTS()>()];

        const label_t tid = threadIdx.x + (threadIdx.y * block::nx()) + (threadIdx.z * block::nx() * block::ny());

        // Coalesced read from global memory
        threadArray<scalar_t, NUMBER_MOMENTS()> moments;
        {
            // Read into shared
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    shared_buffer[moment * block::stride() + tid] = devPtrs.ptr<moment>()[idx];
                    if constexpr (moment == index::rho())
                    {
                        moments.arr[moment] = shared_buffer[moment * block::stride() + tid] + rho0<scalar_t>();
                    }
                    else
                    {
                        moments.arr[moment] = shared_buffer[moment * block::stride() + tid];
                    }
                });
        }

        // Reconstruct the population from the moments
        threadArray<scalar_t, VSet::Q()> pop = VSet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            sharedMemory::save<VSet>(pop.arr, shared_buffer, tid);

            __syncthreads();

            // Pull from shared memory
            sharedMemory::pull<VSet>(pop.arr, shared_buffer);
        }

        // Load pop from global memory in cover nodes
        device::halo<VSet>::popLoad(
            pop.arr,
            fGhost.ptr<0>(),
            fGhost.ptr<1>(),
            fGhost.ptr<2>(),
            fGhost.ptr<3>(),
            fGhost.ptr<4>(),
            fGhost.ptr<5>());

        // Calculate the moments either at the boundary or interior
        {
            const normalVector b_n;
            if (b_n.isBoundary())
            {
                boundaryConditions::calculateMoments<VSet>(pop.arr, moments.arr, b_n);
            }
            else
            {
                VSet::calculateMoments(pop.arr, moments.arr);
            }
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments.arr);

        // Collide
        Collision::collide(moments.arr);

        // Calculate post collision populations
        VSet::reconstruct(pop.arr, moments.arr);

        // Coalesced write to global memory
        moments.arr[0] = moments.arr[0] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments.arr[moment];
            });

        // Save the populations to the block halo
        device::halo<VSet>::popSave(
            pop.arr,
            gGhost.ptr<0>(),
            gGhost.ptr<1>(),
            gGhost.ptr<2>(),
            gGhost.ptr<3>(),
            gGhost.ptr<4>(),
            gGhost.ptr<5>());
    }
}

#endif