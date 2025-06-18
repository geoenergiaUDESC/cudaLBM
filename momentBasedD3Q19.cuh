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

        // scalar_t moms[10] = {
        //     RHO_0 + fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        //     fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)]};

        scalar_t rhoVar = RHO_0 + fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t ux_t30 = fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uy_t30 = fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t uz_t30 = fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xx_t45 = fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xy_t90 = fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_xz_t90 = fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yy_t45 = fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_yz_t90 = fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
        scalar_t m_zz_t45 = fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

        VelocitySet::D3Q19::reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);
        // VelocitySet::D3Q19::reconstruct(pop, moms);

        const label_t xp1 = (threadIdx.x + 1 + block::nx()) % block::nx();
        const label_t xm1 = (threadIdx.x - 1 + block::nx()) % block::nx();

        const label_t yp1 = (threadIdx.y + 1 + block::ny()) % block::ny();
        const label_t ym1 = (threadIdx.y - 1 + block::ny()) % block::ny();

        const label_t zp1 = (threadIdx.z + 1 + block::nz()) % block::nz();
        const label_t zm1 = (threadIdx.z - 1 + block::nz()) % block::nz();

        // save populations in shared memory
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

        // sync threads of the block so all populations are saved
        __syncthreads();

        /* pull */

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

        /* load pop from global in cover nodes */
        ghostInterface.popLoad(pop);

        // scalar_t invRho;
        if (nodeType != BULK)
        {
            // boundaryConditions::calculateMoments(
            //     pop,
            //     moms,
            //     nodeType);

            boundaryConditions::calculateMoments(pop, &rhoVar, &ux_t30, &uy_t30, &uz_t30, &m_xx_t45, &m_xy_t90, &m_xz_t90, &m_yy_t45, &m_yz_t90, &m_zz_t45, nodeType);
        }
        else
        {
            // VelocitySet::D3Q19::calculateMoments(
            //     pop,
            //     moms);

            VelocitySet::D3Q19::calculateMoments(pop, &rhoVar, &ux_t30, &uy_t30, &uz_t30, &m_xx_t45, &m_xy_t90, &m_xz_t90, &m_yy_t45, &m_yz_t90, &m_zz_t45);
        }

        // Multiply moments by as2 -- as4*0.5 -- as4 - add correction to m_alpha_beta
        // moms[1] = VelocitySet::velocitySet::F_M_I_SCALE() * moms[1];
        // moms[2] = VelocitySet::velocitySet::F_M_I_SCALE() * moms[2];
        // moms[3] = VelocitySet::velocitySet::F_M_I_SCALE() * moms[3];
        // moms[4] = VelocitySet::velocitySet::F_M_II_SCALE() * (moms[4]);
        // moms[5] = VelocitySet::velocitySet::F_M_IJ_SCALE() * (moms[5]);
        // moms[6] = VelocitySet::velocitySet::F_M_IJ_SCALE() * (moms[6]);
        // moms[7] = VelocitySet::velocitySet::F_M_II_SCALE() * (moms[7]);
        // moms[8] = VelocitySet::velocitySet::F_M_IJ_SCALE() * (moms[8]);
        // moms[9] = VelocitySet::velocitySet::F_M_II_SCALE() * (moms[9]);

        ux_t30 = VelocitySet::velocitySet::F_M_I_SCALE() * ux_t30;
        uy_t30 = VelocitySet::velocitySet::F_M_I_SCALE() * uy_t30;
        uz_t30 = VelocitySet::velocitySet::F_M_I_SCALE() * uz_t30;
        m_xx_t45 = VelocitySet::velocitySet::F_M_II_SCALE() * m_xx_t45;
        m_xy_t90 = VelocitySet::velocitySet::F_M_IJ_SCALE() * m_xy_t90;
        m_xz_t90 = VelocitySet::velocitySet::F_M_IJ_SCALE() * m_xz_t90;
        m_yy_t45 = VelocitySet::velocitySet::F_M_II_SCALE() * m_yy_t45;
        m_yz_t90 = VelocitySet::velocitySet::F_M_IJ_SCALE() * m_yz_t90;
        m_zz_t45 = VelocitySet::velocitySet::F_M_II_SCALE() * m_zz_t45;

        // Collide
        // collide(moms);
        collide(&rhoVar, &ux_t30, &uy_t30, &uz_t30, &m_xx_t45, &m_xy_t90, &m_xz_t90, &m_yy_t45, &m_yz_t90, &m_zz_t45);

        // Calculate post collision populations
        // VelocitySet::D3Q19::reconstruct(pop, moms);
        VelocitySet::D3Q19::reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

        /* write to global mom */

        // fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[0];
        // fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[1];
        // fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[2];
        // fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[3];
        // fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[4];
        // fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[5];
        // fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[6];
        // fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[7];
        // fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[8];
        // fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moms[9];

        fMom[device::idxMom<index::rho()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar - RHO_0;
        fMom[device::idxMom<index::u()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30;
        fMom[device::idxMom<index::v()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30;
        fMom[device::idxMom<index::w()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30;
        fMom[device::idxMom<index::xx()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xx_t45;
        fMom[device::idxMom<index::xy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xy_t90;
        fMom[device::idxMom<index::xz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xz_t90;
        fMom[device::idxMom<index::yy()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yy_t45;
        fMom[device::idxMom<index::yz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yz_t90;
        fMom[device::idxMom<index::zz()>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_zz_t45;

        ghostInterface.popSave(pop);
    }
}

#endif