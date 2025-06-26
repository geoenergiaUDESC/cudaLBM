/**
Filename: momentBasedD3Q19.cuh
Contents: Main kernel for the moment representation with the D3Q19 velocity set
**/

#ifndef __MBLBM_MOMENTBASEDD319_CUH
#define __MBLBM_MOMENTBASEDD319_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "array/array.cuh"
#include "collision.cuh"
#include "moments/moments.cuh"

namespace LBM
{
    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param fMom Pointer to the interleaved moment variables on the GPU
     * @param nodeType Pointer to the mesh node types on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBounds __global__ void momentBasedD3Q19(
        scalar_t *const fMom,
        const nodeType_t *const dNodeType,
        device::halo blockHalo)
    {
        const nodeType_t nodeType = dNodeType[device::idxScalarBlock(threadIdx, blockIdx)];

        if (device::out_of_bounds() || device::bad_node_type(nodeType))
        {
            return;
        }

        scalar_t pop[VelocitySet::D3Q19::Q()];
        __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];

        momentArray_t moments = {
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
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // Save populations in shared memory
        sharedMemory::save<VelocitySet::D3Q19>(pop, s_pop);

        // Pull from shared memory
        sharedMemory::pull<VelocitySet::D3Q19>(pop, s_pop);

        // Load pop from global memory in cover nodes
        blockHalo.popLoad<VelocitySet::D3Q19>(pop);

        // Calculate the moments either at the boundary or interior
        if (nodeType != BULK)
        {
            boundaryConditions::calculateMoments<VelocitySet::D3Q19>(pop, moments, nodeType);
        }
        else
        {
            VelocitySet::D3Q19::calculateMoments(pop, moments);
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments);

        // Collide
        collide(moments);

        // Calculate post collision populations
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // Write the moments to global memory
        fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)] = moments[0] - rho0();
        fMom[device::idxMom<index::u()>(threadIdx, blockIdx)] = moments[1];
        fMom[device::idxMom<index::v()>(threadIdx, blockIdx)] = moments[2];
        fMom[device::idxMom<index::w()>(threadIdx, blockIdx)] = moments[3];
        fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)] = moments[4];
        fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)] = moments[5];
        fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)] = moments[6];
        fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)] = moments[7];
        fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)] = moments[8];
        fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)] = moments[9];

        // Save the populations to the block halo
        blockHalo.popSave<VelocitySet::D3Q19>(pop);
    }
}

#endif