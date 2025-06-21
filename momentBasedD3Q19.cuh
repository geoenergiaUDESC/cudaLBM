/**
Filename: momentBasedD3Q19.cuh
Contents: Main kernel for the moment representation with the D3Q19 velocity set
**/

#ifndef __MBLBM_MOMENTBASEDD319_CUH
#define __MBLBM_MOMENTBASEDD319_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "velocitySet/velocitySet.cuh"
#include "boundaryConditions.cuh"
#include "globalFunctions.cuh"
#include "boundaryConditions.cuh"
#include "sharedMemory.cuh"

namespace LBM
{
    launchBounds __global__ void momentBasedD3Q19(
        scalar_t *const fMom,
        const nodeType_t *const dNodeType,
        device::halo blockHalo)
    {
        const nodeType_t nodeType = dNodeType[device::idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        if (device::out_of_bounds() || device::bad_node_type(nodeType))
        {
            return;
        }

        scalar_t pop[VelocitySet::D3Q19::Q()];
        __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];

        momentArray_t moments = {
            rho0() + fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)]};

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
        fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[0] - rho0();
        fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[1];
        fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[2];
        fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[3];
        fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[4];
        fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[5];
        fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[6];
        fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[7];
        fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[8];
        fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[9];

        // Save the populations to the block halo
        blockHalo.popSave<VelocitySet::D3Q19>(pop);
    }
}

#endif