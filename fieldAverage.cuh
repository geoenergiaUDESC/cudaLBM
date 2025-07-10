/**
Filename: fieldAverage.cuh
Contents: Implements time averaging of the solution variables
**/

#ifndef __MBLBM_FIELDAVERAGE_CUH
#define __MBLBM_FIELDAVERAGE_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    namespace fieldAverage
    {
        /**
         * @brief Executes the time average of all variables within fMom
         * @param fMom The 10 solution variables
         * @param fMomMean The time averaged solution variables
         * @param nodeTypes The node types of the mesh
         **/
        launchBounds __global__ void calculate(
            const scalar_t *const ptrRestrict fMom,
            scalar_t *const ptrRestrict fMomMean,
            const label_t step)
        {
            if (device::out_of_bounds())
            {
                return;
            }

            const label_t meanCounter = step;

            const scalar_t invCount = static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(meanCounter) + static_cast<scalar_t>(1.0));

            const momentArray_t moments_0 = {
                fMom[device::idxMom<index::rho()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::u()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::v()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::w()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::xx()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::xy()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::xz()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::yy()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::yz()>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::zz()>(threadIdx, blockIdx)]};

            fMomMean[device::idxMom<index::rho()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::rho()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[0]) * invCount;
            fMomMean[device::idxMom<index::u()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::u()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[1]) * invCount;
            fMomMean[device::idxMom<index::v()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::v()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[2]) * invCount;
            fMomMean[device::idxMom<index::w()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::w()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[3]) * invCount;
            fMomMean[device::idxMom<index::xx()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::xx()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[4]) * invCount;
            fMomMean[device::idxMom<index::xy()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::xy()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[5]) * invCount;
            fMomMean[device::idxMom<index::xz()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::xz()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[6]) * invCount;
            fMomMean[device::idxMom<index::yy()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::yy()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[7]) * invCount;
            fMomMean[device::idxMom<index::yz()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::yz()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[8]) * invCount;
            fMomMean[device::idxMom<index::zz()>(threadIdx, blockIdx)] = ((fMomMean[device::idxMom<index::zz()>(threadIdx, blockIdx)] * static_cast<scalar_t>(meanCounter)) + moments_0[9]) * invCount;
        }
    }
}

#endif