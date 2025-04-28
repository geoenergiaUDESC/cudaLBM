/**
Filename: collision.cuh
Contents: Definition of the collision GPU kernel
**/

#ifndef __MBLBM_COLLISION_CUH
#define __MBLBM_COLLISION_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "velocitySet/velocitySet.cuh"
#include "scalarArray/scalarArray.cuh"

namespace mbLBM
{

    [[nodiscard]] inline consteval auto MAX_THREADS_PER_BLOCK() noexcept { return 1024; }
    [[nodiscard]] inline consteval auto MIN_BLOCKS_PER_MP() noexcept { return 8; }

#define MAX_THREADS_PER_BLOCK 1024
#define MIN_BLOCKS_PER_MP 16

    template <const label_t i>
    __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) do_nothing()
    {
        printf("Doing nothing with template arg %u\n", i);
    }

    template <class VelSet>
    __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) __global__ void kernel_collide(
        const device::moments &mom,
        const latticeMesh &mesh,
        const device::ghostInterface<VelSet> &interface)
    {
        constexpr const VelSet velSet;

        constexpr const scalar_t RHO_0 = 0.0;

        const scalar_t moments[10] = {
            RHO_0 + mom.rho()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.u()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.v()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.w()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.m_xx()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.m_xy()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.m_xz()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.m_yy()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.m_yz()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            mom.m_zz()[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)]};

#ifdef VERBOSE
        if ((!threadIdx.y == 0) || (!threadIdx.z == 0))
        {
            // printf("{%u %u %u} %0.15g %0.15g %0.15g %0.15g %0.15g %0.15g %0.15g %0.15g %0.15g %0.15g\n", threadIdx.x, threadIdx.y, threadIdx.z, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);
        }
#endif

        // Perform the reconstruction
        scalar_t pop[velSet.Q()];
        velSet.reconstruct(moments, pop);

        // Save the population in shared memory
        __shared__ scalar_t s_pop[block::size<std::size_t>() * (velSet.Q() - 1)];
        velSet.popSave(pop, s_pop);

        // Not 100% sure if this needs to be here or within popSave
        __syncthreads();

        // Pull population from shared memory
        velSet.popPull(s_pop, pop);

        // Load from shared memory
        velSet.popLoad(mesh, interface, pop);
    }

}

#endif