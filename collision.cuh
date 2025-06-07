/**
Filename: collision.cuh
Contents: Definition of the collision GPU kernel
**/

#ifndef __MBLBM_COLLISION_CUH
#define __MBLBM_COLLISION_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "velocitySet/velocitySet.cuh"
#include "boundaryConditions.cuh"
#include "device_launch_parameters.h"

namespace mbLBM
{
    [[nodiscard]] inline consteval auto MAX_THREADS_PER_BLOCK() noexcept { return 512; }
    [[nodiscard]] inline consteval auto MIN_BLOCKS_PER_MP() noexcept { return 8; }

    __device__ inline void collide(scalar_t moments[10]) noexcept
    {
        const scalar_t t_omegaVar = 1.0 - d_omega;
        const scalar_t omegaVar_d2 = d_omega * 0.5;

        moments[4] = (t_omegaVar * moments[4] + omegaVar_d2 * moments[1] * moments[1]);
        moments[7] = (t_omegaVar * moments[7] + omegaVar_d2 * moments[2] * moments[2]);
        moments[9] = (t_omegaVar * moments[9] + omegaVar_d2 * moments[3] * moments[3]);

        moments[5] = (t_omegaVar * moments[5] + d_omega * moments[1] * moments[2]);
        moments[6] = (t_omegaVar * moments[6] + d_omega * moments[1] * moments[3]);
        moments[8] = (t_omegaVar * moments[8] + d_omega * moments[2] * moments[3]);
    }

    // __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void kernel_collide(
    //     scalar_t *const ptrRestrict rho,
    //     scalar_t *const ptrRestrict u,
    //     scalar_t *const ptrRestrict v,
    //     scalar_t *const ptrRestrict w,
    //     scalar_t *const ptrRestrict m_xx,
    //     scalar_t *const ptrRestrict m_xy,
    //     scalar_t *const ptrRestrict m_xz,
    //     scalar_t *const ptrRestrict m_yy,
    //     scalar_t *const ptrRestrict m_yz,
    //     scalar_t *const ptrRestrict m_zz,
    //     const nodeType::type *const ptrRestrict nodeTypes,
    //     scalar_t *const ptrRestrict f_x0,
    //     scalar_t *const ptrRestrict f_x1,
    //     scalar_t *const ptrRestrict f_y0,
    //     scalar_t *const ptrRestrict f_y1,
    //     scalar_t *const ptrRestrict f_z0,
    //     scalar_t *const ptrRestrict f_z1,
    //     scalar_t *const ptrRestrict g_x0,
    //     scalar_t *const ptrRestrict g_x1,
    //     scalar_t *const ptrRestrict g_y0,
    //     scalar_t *const ptrRestrict g_y1,
    //     scalar_t *const ptrRestrict g_z0,
    //     scalar_t *const ptrRestrict g_z1)
    // {
    //     // Definition of xyz is correct
    //     const label_t x = (threadIdx.x + blockDim.x * blockIdx.x);
    //     const label_t y = (threadIdx.y + blockDim.y * blockIdx.y);
    //     const label_t z = (threadIdx.z + blockDim.z * blockIdx.z);

    //     // Check if we should do an early return
    //     if ((x >= d_nx) || (y >= d_ny) || (z >= d_nz))
    //     {
    //         return;
    //     }

    //     constexpr const scalar_t RHO_0 = 1.0;

    //     // Definition of idxMom is correct
    //     scalar_t moments[10] = {
    //         rho[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] + RHO_0,
    //         u[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         v[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         w[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         m_xx[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         m_xy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         m_xz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         m_yy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         m_yz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
    //         m_zz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)]};

    //     // Perform the reconstruction
    //     scalar_t pop[VelocitySet::D3Q19::Q()];

    //     // Definition of reconstruction seems correct
    //     VelocitySet::D3Q19::reconstruct(pop, moments);

    //     // Save the population in shared memory
    //     __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];
    //     VelocitySet::D3Q19::popSave(pop, s_pop);

    //     // Pull population from shared memory
    //     VelocitySet::D3Q19::popPull(pop, s_pop);

    //     // Load from shared memory
    //     VelocitySet::D3Q19::popLoad(f_x0, f_x1, f_y0, f_y1, f_z0, f_z1, pop);

    //     // Perform the moment calculation
    //     const nodeType::type nodeType = nodeTypes[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

    //     if (nodeType != nodeType::BULK)
    //     {
    //         boundaryConditions::calculateMoments(pop, moments, nodeType);
    //     }
    //     else if (nodeType == nodeType::BULK)
    //     {
    //         VelocitySet::D3Q19::calculateMoments(pop, moments);
    //     }
    //     else
    //     {
    //         printf("Undefined node type\n");
    //     }

    //     // Multiply moments by as2 -- as4*0.5 -- as4 - add correction to m_alpha_beta
    //     // VelocitySet::velocitySet::scale(moments);

    //     // Collide and reconstruct
    //     collide(moments);
    //     VelocitySet::D3Q19::reconstruct(pop, moments);

    //     // Save the moments to global memory
    //     rho[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[0] - RHO_0;
    //     u[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[1];
    //     v[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[2];
    //     w[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[3];
    //     m_xx[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[4];
    //     m_xy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[5];
    //     m_xz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[6];
    //     m_yy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[7];
    //     m_yz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[8];
    //     m_zz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[9];

    //     // Save the interface to global memory
    //     device::ghostInterface::save(g_x0, g_x1, g_y0, g_y1, g_z0, g_z1, pop);
    // }

    __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void kernel_collide(
        scalar_t *const ptrRestrict d_mom,
        const nodeType::type *const ptrRestrict nodeTypes,
        scalar_t *const ptrRestrict f_x0,
        scalar_t *const ptrRestrict f_x1,
        scalar_t *const ptrRestrict f_y0,
        scalar_t *const ptrRestrict f_y1,
        scalar_t *const ptrRestrict f_z0,
        scalar_t *const ptrRestrict f_z1,
        scalar_t *const ptrRestrict g_x0,
        scalar_t *const ptrRestrict g_x1,
        scalar_t *const ptrRestrict g_y0,
        scalar_t *const ptrRestrict g_y1,
        scalar_t *const ptrRestrict g_z0,
        scalar_t *const ptrRestrict g_z1)
    {
        // Definition of xyz is correct
        const label_t x = (threadIdx.x + blockDim.x * blockIdx.x);
        const label_t y = (threadIdx.y + blockDim.y * blockIdx.y);
        const label_t z = (threadIdx.z + blockDim.z * blockIdx.z);

        // Check if we should do an early return
        if ((x >= d_nx) || (y >= d_ny) || (z >= d_nz))
        {
            return;
        }

        constexpr const scalar_t RHO_0 = 1.0;

        // Definition of idxMom is correct
        scalar_t moments[10] = {
            d_mom[idxMom__<0, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] + RHO_0,
            d_mom[idxMom__<1, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<2, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<3, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<4, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<5, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<6, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<7, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<8, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
            d_mom[idxMom__<9, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)]};

        // Perform the reconstruction
        scalar_t pop[VelocitySet::D3Q19::Q()];

        // Definition of reconstruction seems correct
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // Save the population in shared memory
        __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];
        VelocitySet::D3Q19::popSave(pop, s_pop);

        // Pull population from shared memory
        VelocitySet::D3Q19::popPull(pop, s_pop);

        // Load from shared memory
        VelocitySet::D3Q19::popLoad(f_x0, f_x1, f_y0, f_y1, f_z0, f_z1, pop);

        // Perform the moment calculation
        const nodeType::type nodeType = nodeTypes[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        if (nodeType != nodeType::BULK)
        {
            boundaryConditions::calculateMoments(pop, moments, nodeType);
        }
        else if (nodeType == nodeType::BULK)
        {
            VelocitySet::D3Q19::calculateMoments(pop, moments);
        }
        else
        {
            printf("Undefined node type\n");
        }

        // Multiply moments by as2 -- as4*0.5 -- as4 - add correction to m_alpha_beta
        // VelocitySet::velocitySet::scale(moments);

        // Collide and reconstruct
        collide(moments);
        VelocitySet::D3Q19::reconstruct(pop, moments);

        // Save the moments to global memory
        d_mom[idxMom__<0, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[0] - RHO_0;
        d_mom[idxMom__<1, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[1];
        d_mom[idxMom__<2, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[2];
        d_mom[idxMom__<3, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[3];
        d_mom[idxMom__<4, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[4];
        d_mom[idxMom__<5, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[5];
        d_mom[idxMom__<6, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[6];
        d_mom[idxMom__<7, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[7];
        d_mom[idxMom__<8, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[8];
        d_mom[idxMom__<9, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[9];

        // Save the interface to global memory
        device::ghostInterface::save(g_x0, g_x1, g_y0, g_y1, g_z0, g_z1, pop);
    }

}

#endif