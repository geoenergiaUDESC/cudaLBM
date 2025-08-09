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
    using VSet = VelocitySet::D3Q27;
    using Collision = secondOrder;

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param fMom Pointer to the interleaved moment variables on the GPU
     * @param nodeType Pointer to the mesh node types on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBounds static __global__ void momentBasedD3Q19(
        scalar_t *const ptrRestrict fMom,
        device::halo<VSet> blockHalo)
    {
        // device::prefetch<device::cacheLevel::L1, device::evictionPolicy::first, 1>(fMom);

        // Always a multiple of 32, so no need to check this(I think)
        // if (device::out_of_bounds())
        // {
        //     return;
        // }

        threadArray<scalar_t, NUMBER_MOMENTS()> moments = {
            fMom[device::idxMom<0>()] + rho0(),
            fMom[device::idxMom<1>()],
            fMom[device::idxMom<2>()],
            fMom[device::idxMom<3>()],
            fMom[device::idxMom<4>()],
            fMom[device::idxMom<5>()],
            fMom[device::idxMom<6>()],
            fMom[device::idxMom<7>()],
            fMom[device::idxMom<8>()],
            fMom[device::idxMom<9>()]};

        // Reconstruct the population from the moments
        threadArray<scalar_t, VSet::Q()> pop = VSet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Declare shared memory
            __shared__ sharedArray<scalar_t, VSet::Q() - 1, block::size()> s_pop;

            // Save populations in shared memory
            sharedMemory::save(pop.arr, s_pop.arr);

            // Pull from shared memory
            sharedMemory::pull(pop.arr, s_pop.arr);
        }

        // Load pop from global memory in cover nodes
        blockHalo.popLoad(pop.arr);

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

        // Write to global memory
        // fMom[device::idxMom<0>()] = moments.arr[0] - rho0();
        // fMom[device::idxMom<1>()] = moments.arr[1];
        // fMom[device::idxMom<2>()] = moments.arr[2];
        // fMom[device::idxMom<3>()] = moments.arr[3];
        // fMom[device::idxMom<4>()] = moments.arr[4];
        // fMom[device::idxMom<5>()] = moments.arr[5];
        // fMom[device::idxMom<6>()] = moments.arr[6];
        // fMom[device::idxMom<7>()] = moments.arr[7];
        // fMom[device::idxMom<8>()] = moments.arr[8];
        // fMom[device::idxMom<9>()] = moments.arr[9];

        // Save the populations to the block halo
        // blockHalo.popSave(pop.arr);
    }
}

#endif