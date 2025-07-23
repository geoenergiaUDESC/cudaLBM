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
    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param fMom Pointer to the interleaved moment variables on the GPU
     * @param nodeType Pointer to the mesh node types on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBounds __global__ void momentBasedD3Q19(
        scalar_t *const ptrRestrict fMom,
        device::halo blockHalo)
    {
        device::prefetch<device::cacheLevel::L1, device::evictionPolicy::first, 1>(fMom);

        // Always a multiple of 32, so no need to check this(I think)
        if (device::out_of_bounds())
        {
            return;
        }

        threadArray<scalar_t, NUMBER_MOMENTS()> moments = {
            rho0() + fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::u()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::v()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::w()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)],
            fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)]};

        // Reconstruct the population from the moments
        threadArray<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Declare shared memory
            __shared__ sharedArray<scalar_t, VelocitySet::D3Q19::Q() - 1, block::size()> s_pop;

            // Save populations in shared memory
            sharedMemory::save<VelocitySet::D3Q19>(pop.arr, s_pop.arr);

            // Pull from shared memory
            sharedMemory::pull<VelocitySet::D3Q19>(pop.arr, s_pop.arr);
        }

        // Load pop from global memory in cover nodes
        blockHalo.popLoad<VelocitySet::D3Q19>(pop.arr);

        // Calculate the moments either at the boundary or interior
        const normalVector b_n;
        if (b_n.isBoundary())
        {
            boundaryConditions::calculateMoments<VelocitySet::D3Q19>(pop.arr, moments.arr, b_n);
        }
        else
        {
            VelocitySet::D3Q19::calculateMoments(pop.arr, moments.arr);
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments.arr);

        // Collide
        secondOrder::collide(moments.arr);

        // Calculate post collision populations
        VelocitySet::D3Q19::reconstruct(pop.arr, moments.arr);

        // Write to global memory
        fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)] = moments.arr[0] - rho0();
        fMom[device::idxMom<index::u()>(threadIdx, blockIdx)] = moments.arr[1];
        fMom[device::idxMom<index::v()>(threadIdx, blockIdx)] = moments.arr[2];
        fMom[device::idxMom<index::w()>(threadIdx, blockIdx)] = moments.arr[3];
        fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)] = moments.arr[4];
        fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)] = moments.arr[5];
        fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)] = moments.arr[6];
        fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)] = moments.arr[7];
        fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)] = moments.arr[8];
        fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)] = moments.arr[9];

        // Save the populations to the block halo
        blockHalo.popSave<VelocitySet::D3Q19>(pop.arr);
    }
}

#endif