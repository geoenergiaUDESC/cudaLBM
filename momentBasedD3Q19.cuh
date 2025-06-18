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

namespace LBM
{
    __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void momentBasedD3Q19(
        scalar_t *const fMom,
        const nodeType_t *const dNodeType,
        device::halo ghostInterface)
    {
        const nodeType_t nodeType = dNodeType[device::idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        if (device::out_of_bounds(d_nx, d_ny, d_nz) || device::bad_node_type(nodeType))
        {
            return;
        }

        scalar_t pop[VelocitySet::D3Q19::Q()];
        __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];

        momentArray_t moments = {
            RHO_0 + fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)]};

        VelocitySet::D3Q19::reconstruct(pop, moments);

        // save populations in shared memory
        {
            s_pop[device::idxPopBlock<0>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[1];
            s_pop[device::idxPopBlock<1>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[2];
            s_pop[device::idxPopBlock<2>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[3];
            s_pop[device::idxPopBlock<3>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[4];
            s_pop[device::idxPopBlock<4>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[5];
            s_pop[device::idxPopBlock<5>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[6];
            s_pop[device::idxPopBlock<6>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[7];
            s_pop[device::idxPopBlock<7>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[8];
            s_pop[device::idxPopBlock<8>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[9];
            s_pop[device::idxPopBlock<9>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[10];
            s_pop[device::idxPopBlock<10>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[11];
            s_pop[device::idxPopBlock<11>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[12];
            s_pop[device::idxPopBlock<12>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[13];
            s_pop[device::idxPopBlock<13>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[14];
            s_pop[device::idxPopBlock<14>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[15];
            s_pop[device::idxPopBlock<15>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[16];
            s_pop[device::idxPopBlock<16>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[17];
            s_pop[device::idxPopBlock<17>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[18];
        }

        // sync threads of the block so all populations are saved
        __syncthreads();

        // Pull from shared memory
        {
            const label_t xp1 = (threadIdx.x + 1 + block::nx()) % block::nx();
            const label_t xm1 = (threadIdx.x - 1 + block::nx()) % block::nx();

            const label_t yp1 = (threadIdx.y + 1 + block::ny()) % block::ny();
            const label_t ym1 = (threadIdx.y - 1 + block::ny()) % block::ny();

            const label_t zp1 = (threadIdx.z + 1 + block::nz()) % block::nz();
            const label_t zm1 = (threadIdx.z - 1 + block::nz()) % block::nz();

            pop[1] = s_pop[device::idxPopBlock<0>(xm1, threadIdx.y, threadIdx.z)];
            pop[2] = s_pop[device::idxPopBlock<1>(xp1, threadIdx.y, threadIdx.z)];
            pop[3] = s_pop[device::idxPopBlock<2>(threadIdx.x, ym1, threadIdx.z)];
            pop[4] = s_pop[device::idxPopBlock<3>(threadIdx.x, yp1, threadIdx.z)];
            pop[5] = s_pop[device::idxPopBlock<4>(threadIdx.x, threadIdx.y, zm1)];
            pop[6] = s_pop[device::idxPopBlock<5>(threadIdx.x, threadIdx.y, zp1)];
            pop[7] = s_pop[device::idxPopBlock<6>(xm1, ym1, threadIdx.z)];
            pop[8] = s_pop[device::idxPopBlock<7>(xp1, yp1, threadIdx.z)];
            pop[9] = s_pop[device::idxPopBlock<8>(xm1, threadIdx.y, zm1)];
            pop[10] = s_pop[device::idxPopBlock<9>(xp1, threadIdx.y, zp1)];
            pop[11] = s_pop[device::idxPopBlock<10>(threadIdx.x, ym1, zm1)];
            pop[12] = s_pop[device::idxPopBlock<11>(threadIdx.x, yp1, zp1)];
            pop[13] = s_pop[device::idxPopBlock<12>(xm1, yp1, threadIdx.z)];
            pop[14] = s_pop[device::idxPopBlock<13>(xp1, ym1, threadIdx.z)];
            pop[15] = s_pop[device::idxPopBlock<14>(xm1, threadIdx.y, zp1)];
            pop[16] = s_pop[device::idxPopBlock<15>(xp1, threadIdx.y, zm1)];
            pop[17] = s_pop[device::idxPopBlock<16>(threadIdx.x, ym1, zp1)];
            pop[18] = s_pop[device::idxPopBlock<17>(threadIdx.x, yp1, zm1)];
        }

        // Load pop from global memory in cover nodes
        ghostInterface.popLoad(pop);

        // Calculate the moments either at the boundary or interior
        if (nodeType != BULK)
        {
            boundaryConditions::calculateMoments(pop, moments, nodeType);
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
        fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[0] - RHO_0;
        fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[1];
        fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[2];
        fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[3];
        fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[4];
        fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[5];
        fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[6];
        fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[7];
        fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[8];
        fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[9];

        ghostInterface.popSave(pop);
    }
}

#endif