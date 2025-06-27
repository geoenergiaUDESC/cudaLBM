/**
Filename: momentBasedD3Q19.cuh
Contents: Main kernel for the moment representation with the D3Q19 velocity set
          (Optimized with Shared Memory for Moments and Halo Exchange)
**/

#ifndef __MBLBM_MOMENTBASEDD3Q19_CUH
#define __MBLBM_MOMENTBASEDD3Q19_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "array/array.cuh"
#include "velocitySet/velocitySet.cuh"
#include "boundaryConditions.cuh"
#include "globalFunctions.cuh"
#include "collision.cuh"
#include "moments/moments.cuh"
// CORRECTED PATH on the line below
#include "moments/halo.cuh" // Include our newly modified halo class

namespace LBM
{
    launchBounds __global__ void momentBasedD3Q19(
        scalar_t *const fMom,
        const nodeType_t *const dNodeType,
        device::halo blockHalo)
    {
        // --- KERNEL SETUP ---
        const label_t thread_id_in_block = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        const nodeType_t nodeType = dNodeType[device::idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        if (device::out_of_bounds() || device::bad_node_type(nodeType))
        {
            return;
        }

        // --- SHARED MEMORY DECLARATION ---
        __shared__ scalar_t s_buffer[block::size() * (VelocitySet::D3Q19::Q() - 1)];
        scalar_t *const s_moments = s_buffer;
        scalar_t pop[VelocitySet::D3Q19::Q()];

        // --- MOMENT LOADING ---
#pragma unroll
        for (int i = 0; i < NUMBER_MOMENTS(); ++i)
        {
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + i] = fMom[device::idxMom(i, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        }
        __syncthreads();

        momentArray_t moments = {
            rho0() + s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::rho()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::u()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::v()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::w()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::xx()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::xy()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::xz()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::yy()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::yz()],
            s_moments[thread_id_in_block * NUMBER_MOMENTS() + index::zz()]};

        // --- STREAMING & RECONSTRUCTION ---
        VelocitySet::D3Q19::reconstruct(pop, moments);
        // The shared memory buffer is reused here for streaming
        sharedMemory::save<VelocitySet::D3Q19>(pop, s_buffer);
        sharedMemory::pull<VelocitySet::D3Q19>(pop, s_buffer);

        // --- OPTIMIZATION: TWO-STAGE HALO LOAD ---
        // STAGE 1A: Global -> Shared. The s_buffer is now reused for the halo.
        blockHalo.halo_global_to_shared_load<VelocitySet::D3Q19>(s_buffer);
        __syncthreads(); // CRITICAL BARRIER

        // STAGE 1B: Shared -> Register
        blockHalo.popLoad_from_shared<VelocitySet::D3Q19>(pop, s_buffer);

        // --- POST-STREAMING COMPUTATION ---
        if (nodeType != BULK)
        {
            boundaryConditions::calculateMoments<VelocitySet::D3Q19>(pop, moments, nodeType);
        }
        else
        {
            VelocitySet::D3Q19::calculateMoments(pop, moments);
        }

        VelocitySet::velocitySet::scale(moments);
        collide(moments);
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // --- FINAL GLOBAL WRITE ---
        fMom[device::idxMom(index::rho(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[0] - rho0();
        fMom[device::idxMom(index::u(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[1];
        fMom[device::idxMom(index::v(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[2];
        fMom[device::idxMom(index::w(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[3];
        fMom[device::idxMom(index::xx(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[4];
        fMom[device::idxMom(index::xy(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[5];
        fMom[device::idxMom(index::xz(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[6];
        fMom[device::idxMom(index::yy(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[7];
        fMom[device::idxMom(index::yz(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[8];
        fMom[device::idxMom(index::zz(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[9];

        // --- OPTIMIZATION: TWO-STAGE HALO SAVE ---
        // STAGE 2A: Register -> Shared
        blockHalo.popSave_to_shared<VelocitySet::D3Q19>(pop, s_buffer);
        __syncthreads(); // CRITICAL BARRIER

        // STAGE 2B: Shared -> Global
        blockHalo.halo_shared_to_global_save<VelocitySet::D3Q19>(s_buffer);
    }
} // namespace LBM

#endif // __MBLBM_MOMENTBASEDD3Q19_CUH