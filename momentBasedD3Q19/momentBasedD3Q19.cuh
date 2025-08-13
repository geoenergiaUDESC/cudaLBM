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
#include "../moments/moments.cuh"
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
        device::halo blockHalo)
    {
        // Always a multiple of 32, so no need to check this(I think)
        if (device::out_of_bounds())
        {
            return;
        }

        // Coalesced read from global memory
        threadArray<scalar_t, NUMBER_MOMENTS()> moments;
        {
            __shared__ scalar_t s_mom[10][block::size()];

            const label_t thread_idx_in_block = threadIdx.x + (threadIdx.y * block::nx()) + (threadIdx.z * block::nx() * block::ny());

            // Read into shared
            device::constexpr_for<0, 10>(
                [&](const auto q_)
                {
                    s_mom[q_][thread_idx_in_block] = devPtrs.ptr<q_>()[device::idx()];
                });
            __syncthreads();

            // Pull into thread memory
            moments.arr[0] = s_mom[0][thread_idx_in_block] + rho0();
            moments.arr[1] = s_mom[1][thread_idx_in_block];
            moments.arr[2] = s_mom[2][thread_idx_in_block];
            moments.arr[3] = s_mom[3][thread_idx_in_block];
            moments.arr[4] = s_mom[4][thread_idx_in_block];
            moments.arr[5] = s_mom[5][thread_idx_in_block];
            moments.arr[6] = s_mom[6][thread_idx_in_block];
            moments.arr[7] = s_mom[7][thread_idx_in_block];
            moments.arr[8] = s_mom[8][thread_idx_in_block];
            moments.arr[9] = s_mom[9][thread_idx_in_block];
            // __syncthreads();
        }

        // Reconstruct the population from the moments
        threadArray<scalar_t, VSet::Q()> pop = VSet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Declare shared memory
            __shared__ sharedArray<scalar_t, VSet::Q() - 1, block::size()> s_pop;

            // Save populations in shared memory
            sharedMemory::save<VSet>(pop.arr, s_pop.arr);

            // Pull from shared memory
            sharedMemory::pull<VSet>(pop.arr, s_pop.arr);
        }

        // Load pop from global memory in cover nodes
        blockHalo.popLoad<VSet>(pop.arr);

        // Calculate the moments either at the boundary or interior
        const normalVector b_n;
        if (b_n.isBoundary())
        {
            boundaryConditions::calculateMoments<VSet>(pop.arr, moments.arr, b_n);
        }
        else
        {
            VSet::calculateMoments(pop.arr, moments.arr);
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments.arr);

        // Collide
        Collision::collide(moments.arr);

        // Calculate post collision populations
        VSet::reconstruct(pop.arr, moments.arr);

        // Coalesced write to global memory
        devPtrs.ptr<0>()[device::idx()] = moments.arr[0] - rho0();
        devPtrs.ptr<1>()[device::idx()] = moments.arr[1];
        devPtrs.ptr<2>()[device::idx()] = moments.arr[2];
        devPtrs.ptr<3>()[device::idx()] = moments.arr[3];
        devPtrs.ptr<4>()[device::idx()] = moments.arr[4];
        devPtrs.ptr<5>()[device::idx()] = moments.arr[5];
        devPtrs.ptr<6>()[device::idx()] = moments.arr[6];
        devPtrs.ptr<7>()[device::idx()] = moments.arr[7];
        devPtrs.ptr<8>()[device::idx()] = moments.arr[8];
        devPtrs.ptr<9>()[device::idx()] = moments.arr[9];

        // Save the populations to the block halo
        blockHalo.popSave<VSet>(pop.arr);
    }
}

#endif