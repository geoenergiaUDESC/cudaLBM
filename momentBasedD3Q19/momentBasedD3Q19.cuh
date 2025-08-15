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
        const scalar_t *const ptrRestrict fx0,
        const scalar_t *const ptrRestrict fx1,
        const scalar_t *const ptrRestrict fy0,
        const scalar_t *const ptrRestrict fy1,
        const scalar_t *const ptrRestrict fz0,
        const scalar_t *const ptrRestrict fz1,
        scalar_t *const ptrRestrict gx0,
        scalar_t *const ptrRestrict gx1,
        scalar_t *const ptrRestrict gy0,
        scalar_t *const ptrRestrict gy1,
        scalar_t *const ptrRestrict gz0,
        scalar_t *const ptrRestrict gz1)
    {
        // Always a multiple of 32, so no need to check this(I think)
        // if (device::out_of_bounds())
        // {
        //     return;
        // }

        // Declare shared memory (flattened)
        __shared__ scalar_t shared_buffer[block::sharedMemoryBufferSize<VelocitySet::D3Q19>()];

        const label_t tid = threadIdx.x + (threadIdx.y * block::nx()) + (threadIdx.z * block::nx() * block::ny());
        const label_t idx = device::idx();

        // Coalesced read from global memory
        threadArray<scalar_t, NUMBER_MOMENTS()> moments;
        {

            // Read into shared
            shared_buffer[0 * block::stride() + tid] = devPtrs.ptr<0>()[device::idx()];
            shared_buffer[1 * block::stride() + tid] = devPtrs.ptr<1>()[device::idx()];
            shared_buffer[2 * block::stride() + tid] = devPtrs.ptr<2>()[device::idx()];
            shared_buffer[3 * block::stride() + tid] = devPtrs.ptr<3>()[device::idx()];
            shared_buffer[4 * block::stride() + tid] = devPtrs.ptr<4>()[device::idx()];
            shared_buffer[5 * block::stride() + tid] = devPtrs.ptr<5>()[device::idx()];
            shared_buffer[6 * block::stride() + tid] = devPtrs.ptr<6>()[device::idx()];
            shared_buffer[7 * block::stride() + tid] = devPtrs.ptr<7>()[device::idx()];
            shared_buffer[8 * block::stride() + tid] = devPtrs.ptr<8>()[device::idx()];
            shared_buffer[9 * block::stride() + tid] = devPtrs.ptr<9>()[device::idx()];

            // Synchronise before pulling into thread
            __syncthreads();

            // Pull into thread memory
            moments.arr[0] = shared_buffer[0 * block::stride() + tid] + rho0();
            moments.arr[1] = shared_buffer[1 * block::stride() + tid];
            moments.arr[2] = shared_buffer[2 * block::stride() + tid];
            moments.arr[3] = shared_buffer[3 * block::stride() + tid];
            moments.arr[4] = shared_buffer[4 * block::stride() + tid];
            moments.arr[5] = shared_buffer[5 * block::stride() + tid];
            moments.arr[6] = shared_buffer[6 * block::stride() + tid];
            moments.arr[7] = shared_buffer[7 * block::stride() + tid];
            moments.arr[8] = shared_buffer[8 * block::stride() + tid];
            moments.arr[9] = shared_buffer[9 * block::stride() + tid];
        }

        // Reconstruct the population from the moments
        threadArray<scalar_t, VSet::Q()> pop = VSet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            sharedMemory::save<VelocitySet::D3Q19>(pop.arr, shared_buffer, tid);

            // Pull from shared memory
            sharedMemory::pull<VelocitySet::D3Q19>(pop.arr, shared_buffer);
        }

        // Load pop from global memory in cover nodes
        device::halo::popLoad<VelocitySet::D3Q19>(
            pop.arr,
            fx0, fx1,
            fy0, fy1,
            fz0, fz1);

        // Calculate the moments either at the boundary or interior
        {
            const normalVector b_n;
            if (b_n.isBoundary())
            {
                boundaryConditions::calculateMoments<VelocitySet::D3Q19>(pop.arr, moments.arr, b_n);
            }
            else
            {
                VelocitySet::D3Q19::calculateMoments(pop.arr, moments.arr);
            }
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments.arr);

        // Collide
        Collision::collide(moments.arr);

        // Calculate post collision populations
        VelocitySet::D3Q19::reconstruct(pop.arr, moments.arr);

        // __pipeline pipeline;

        // Coalesced write to global memory
        moments.arr[0] = moments.arr[0] - rho0();
        {
            devPtrs.ptr<0>()[idx] = moments.arr[0];
            devPtrs.ptr<1>()[idx] = moments.arr[1];
            devPtrs.ptr<2>()[idx] = moments.arr[2];
            devPtrs.ptr<3>()[idx] = moments.arr[3];
            devPtrs.ptr<4>()[idx] = moments.arr[4];
            devPtrs.ptr<5>()[idx] = moments.arr[5];
            devPtrs.ptr<6>()[idx] = moments.arr[6];
            devPtrs.ptr<7>()[idx] = moments.arr[7];
            devPtrs.ptr<8>()[idx] = moments.arr[8];
            devPtrs.ptr<9>()[idx] = moments.arr[9];
        }

        // Save the populations to the block halo
        device::halo::popSave<VelocitySet::D3Q19>(
            pop.arr,
            gx0, gx1,
            gy0, gy1,
            gz0, gz1);
    }
}

#endif