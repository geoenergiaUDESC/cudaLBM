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
#include "globalFunctions.cuh"
#include "boundaryConditions.cuh"

namespace mbLBM
{

    [[nodiscard]] inline consteval auto MAX_THREADS_PER_BLOCK() noexcept { return 1024; }
    [[nodiscard]] inline consteval auto MIN_BLOCKS_PER_MP() noexcept { return 8; }

    __device__ static inline void collide(
        scalar_t *const rhoVar,
        scalar_t *const ux_t30,
        scalar_t *const uy_t30,
        scalar_t *const uz_t30,
        scalar_t *const m_xx_t45,
        scalar_t *const m_xy_t90,
        scalar_t *const m_xz_t90,
        scalar_t *const m_yy_t45,
        scalar_t *const m_yz_t90,
        scalar_t *const m_zz_t45) noexcept
    {
        const scalar_t omegaVar = OMEGA;
        const scalar_t t_omegaVar = 1 - omegaVar;
        const scalar_t omegaVar_d2 = omegaVar / 2;

        // Velocity updates are removed since force terms are zero
        // Diagonal moment updates (remove force terms)
        *m_xx_t45 = t_omegaVar * *m_xx_t45 + omegaVar_d2 * (*ux_t30) * (*ux_t30);
        *m_yy_t45 = t_omegaVar * *m_yy_t45 + omegaVar_d2 * (*uy_t30) * (*uy_t30);
        *m_zz_t45 = t_omegaVar * *m_zz_t45 + omegaVar_d2 * (*uz_t30) * (*uz_t30);

        // Off-diagonal moment updates (remove force terms)
        *m_xy_t90 = t_omegaVar * *m_xy_t90 + omegaVar * (*ux_t30) * (*uy_t30);
        *m_xz_t90 = t_omegaVar * *m_xz_t90 + omegaVar * (*ux_t30) * (*uz_t30);
        *m_yz_t90 = t_omegaVar * *m_yz_t90 + omegaVar * (*uy_t30) * (*uz_t30);
    }

    __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuMomCollisionStream(
        scalar_t *const fMom,
        const nodeType_t *const dNodeType,
        device::halo ghostInterface)
    {
        const nodeType_t nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        if (out_of_bounds(NX, NY, NZ) || bad_node_type(nodeType))
        {
            return;
        }

        scalar_t pop[Q];
        __shared__ scalar_t s_pop[block::size() * (Q - 1)];

        scalar_t rhoVar = RHO_0 + fMom[idxMom<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t ux_t30 = fMom[idxMom<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uy_t30 = fMom[idxMom<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uz_t30 = fMom[idxMom<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xx_t45 = fMom[idxMom<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xy_t90 = fMom[idxMom<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xz_t90 = fMom[idxMom<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yy_t45 = fMom[idxMom<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yz_t90 = fMom[idxMom<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_zz_t45 = fMom[idxMom<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        VelocitySet::D3Q19::reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

        const label_t xp1 = (threadIdx.x + 1 + block::nx()) % block::nx();
        const label_t xm1 = (threadIdx.x - 1 + block::nx()) % block::nx();

        const label_t yp1 = (threadIdx.y + 1 + block::ny()) % block::ny();
        const label_t ym1 = (threadIdx.y - 1 + block::ny()) % block::ny();

        const label_t zp1 = (threadIdx.z + 1 + block::nz()) % block::nz();
        const label_t zm1 = (threadIdx.z - 1 + block::nz()) % block::nz();

        // save populations in shared memory
        s_pop[idxPopBlock<0>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[1];
        s_pop[idxPopBlock<1>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[2];
        s_pop[idxPopBlock<2>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[3];
        s_pop[idxPopBlock<3>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[4];
        s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[5];
        s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[6];
        s_pop[idxPopBlock<6>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[7];
        s_pop[idxPopBlock<7>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[8];
        s_pop[idxPopBlock<8>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[9];
        s_pop[idxPopBlock<9>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[10];
        s_pop[idxPopBlock<10>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[11];
        s_pop[idxPopBlock<11>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[12];
        s_pop[idxPopBlock<12>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[13];
        s_pop[idxPopBlock<13>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[14];
        s_pop[idxPopBlock<14>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[15];
        s_pop[idxPopBlock<15>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[16];
        s_pop[idxPopBlock<16>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[17];
        s_pop[idxPopBlock<17>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[18];

        // sync threads of the block so all populations are saved
        __syncthreads();

        /* pull */

        pop[1] = s_pop[idxPopBlock<0>(xm1, threadIdx.y, threadIdx.z)];
        pop[2] = s_pop[idxPopBlock<1>(xp1, threadIdx.y, threadIdx.z)];
        pop[3] = s_pop[idxPopBlock<2>(threadIdx.x, ym1, threadIdx.z)];
        pop[4] = s_pop[idxPopBlock<3>(threadIdx.x, yp1, threadIdx.z)];
        pop[5] = s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, zm1)];
        pop[6] = s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, zp1)];
        pop[7] = s_pop[idxPopBlock<6>(xm1, ym1, threadIdx.z)];
        pop[8] = s_pop[idxPopBlock<7>(xp1, yp1, threadIdx.z)];
        pop[9] = s_pop[idxPopBlock<8>(xm1, threadIdx.y, zm1)];
        pop[10] = s_pop[idxPopBlock<9>(xp1, threadIdx.y, zp1)];
        pop[11] = s_pop[idxPopBlock<10>(threadIdx.x, ym1, zm1)];
        pop[12] = s_pop[idxPopBlock<11>(threadIdx.x, yp1, zp1)];
        pop[13] = s_pop[idxPopBlock<12>(xm1, yp1, threadIdx.z)];
        pop[14] = s_pop[idxPopBlock<13>(xp1, ym1, threadIdx.z)];
        pop[15] = s_pop[idxPopBlock<14>(xm1, threadIdx.y, zp1)];
        pop[16] = s_pop[idxPopBlock<15>(xp1, threadIdx.y, zm1)];
        pop[17] = s_pop[idxPopBlock<16>(threadIdx.x, ym1, zp1)];
        pop[18] = s_pop[idxPopBlock<17>(threadIdx.x, yp1, zm1)];

        /* load pop from global in cover nodes */
        ghostInterface.popLoad(pop);

        scalar_t invRho;
        if (nodeType != BULK)
        {
            boundaryConditions::calculateMoments(
                pop,
                &rhoVar,
                &ux_t30,
                &uy_t30,
                &uz_t30,
                &m_xx_t45,
                &m_xy_t90,
                &m_xz_t90,
                &m_yy_t45,
                &m_yz_t90,
                &m_zz_t45,
                nodeType);

            invRho = 1.0 / rhoVar;
        }
        else
        {
            // Calculate streaming moments

            // Equation 3
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
            invRho = 1 / rhoVar;

            // Equation 4 + force correction
            ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16])) * invRho;
            uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18])) * invRho;
            uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17])) * invRho;

            // Equation 5
            m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - VelocitySet::velocitySet::cs2();
            m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14]) * invRho;
            m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16]) * invRho;
            m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - VelocitySet::velocitySet::cs2();
            m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18]) * invRho;
            m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - VelocitySet::velocitySet::cs2();
        }

        // Multiply moments by as2 -- as4*0.5 -- as4 - add correction to m_alpha_beta
        ux_t30 = VelocitySet::velocitySet::F_M_I_SCALE() * ux_t30;
        uy_t30 = VelocitySet::velocitySet::F_M_I_SCALE() * uy_t30;
        uz_t30 = VelocitySet::velocitySet::F_M_I_SCALE() * uz_t30;

        m_xx_t45 = VelocitySet::velocitySet::F_M_II_SCALE() * (m_xx_t45);
        m_xy_t90 = VelocitySet::velocitySet::F_M_IJ_SCALE() * (m_xy_t90);
        m_xz_t90 = VelocitySet::velocitySet::F_M_IJ_SCALE() * (m_xz_t90);
        m_yy_t45 = VelocitySet::velocitySet::F_M_II_SCALE() * (m_yy_t45);
        m_yz_t90 = VelocitySet::velocitySet::F_M_IJ_SCALE() * (m_yz_t90);
        m_zz_t45 = VelocitySet::velocitySet::F_M_II_SCALE() * (m_zz_t45);

        // Collide
        collide(&rhoVar, &ux_t30, &uy_t30, &uz_t30, &m_xx_t45, &m_xy_t90, &m_xz_t90, &m_yy_t45, &m_yz_t90, &m_zz_t45);

        // Calculate post collision populations
        VelocitySet::D3Q19::reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

        /* write to global mom */

        fMom[idxMom<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar - RHO_0;
        fMom[idxMom<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30;
        fMom[idxMom<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30;
        fMom[idxMom<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30;
        fMom[idxMom<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xx_t45;
        fMom[idxMom<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xy_t90;
        fMom[idxMom<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xz_t90;
        fMom[idxMom<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yy_t45;
        fMom[idxMom<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yz_t90;
        fMom[idxMom<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_zz_t45;

        ghostInterface.populationSave(pop);
    }

}

#endif