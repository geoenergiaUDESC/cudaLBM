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

    template <class VelSet>
    __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) void kernel_collide(
        const scalar_t *rho,
        const scalar_t *u,
        const scalar_t *v,
        const scalar_t *w,
        const scalar_t *m_xx,
        const scalar_t *m_xy,
        const scalar_t *m_xz,
        const scalar_t *m_yy,
        const scalar_t *m_yz,
        const scalar_t *m_zz,
        const latticeMesh &mesh,
        const device::ghostInterface<VelSet> &interface)
    {
        constexpr const VelSet velSet;

        constexpr const scalar_t RHO_0 = 0.0;

        const scalar_t moments[10] = {
            RHO_0 + rho[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            u[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            v[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            w[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            m_xx[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            m_xy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            m_xz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            m_yy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            m_yz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)],
            m_zz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, mesh)]};

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

        velSet.popLoad(mesh, interface, pop);
    }

    template <class VelSet>
    inline void collide(const device::moments &mom, const latticeMesh &mesh)
    {
        kernel_collide<VelSet><<<
            dim3{mesh.nx(), mesh.ny(), mesh.nz()},
            dim3{16, 16, 4},
            0,
            0>>>(
            mom.rho().get(),
            mom.u().get(),
            mom.v().get(),
            mom.w().get(),
            mom.m_xx().get(),
            mom.m_xy().get(),
            mom.m_xz().get(),
            mom.m_yy().get(),
            mom.m_yz().get(),
            mom.m_zz().get(),
            mesh.nx(),
            mesh.ny(),
            mesh.nz());
    }

}

#endif