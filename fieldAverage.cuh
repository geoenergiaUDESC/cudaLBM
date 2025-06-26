/**
Filename: fieldAverage.cuh
Contents: Implements time averaging of the solution variables (Optimized Version)
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
         * @brief Executes the time average of all variables within fMom (Optimized)
         * @param fMom The 10 solution variables
         * @param fMomMean The time averaged solution variables
         * @param nodeTypes The node types of the mesh
         * @param step The current step, used to calculate meanCounter
         * @param invCount The pre-calculated value of 1.0f / (step + 1.0f)
         **/
        // --- ADD `launchBounds` TO THE KERNEL DEFINITION ---
        launchBounds __global__ void calculate(
            const scalar_t *const ptrRestrict fMom,
            scalar_t *const ptrRestrict fMomMean,
            const label_t step,
            const scalar_t invCount,
            const scalar_t meanCounterScalar)
        {
            // Get the thread's unique spatial index once
            const label_t spatial_idx = device::idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
            // const nodeType_t nodeType = nodeTypes[spatial_idx];

            // if (device::out_of_bounds() || device::bad_node_type(nodeType))
            // {
            //     return;
            // }

            // const scalar_t meanCounterScalar = static_cast<scalar_t>(step);

            // This loop processes one moment at a time. This is the core optimization
            // that allows the GPU to "pipeline" the work, hiding the time it takes
            // to fetch data from memory behind useful math calculations.
            for (int i = 0; i < 10; ++i)
            {
                // Calculate the full index for the current moment using the new idxMom function
                const label_t momentIdx = device::idxMom(i, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

                // Perform the read, compute, and write for a single moment
                const scalar_t currentMoment = fMom[momentIdx];
                const scalar_t meanMoment = fMomMean[momentIdx];

                fMomMean[momentIdx] = ((meanMoment * meanCounterScalar) + currentMoment) * invCount;
            }
        }

        __device__ [[nodiscard]] inline constexpr label_t MOM_INDEX(
            const label_t baseIndex,
            const label_t m,
            const label_t momentStride) noexcept
        {
            return (baseIndex + (m * momentStride));
        }

        // --- ADD `launchBounds` TO THE KERNEL DEFINITION ---
        launchBounds __global__ void calculatePipelined(
            const scalar_t *__restrict__ fMom,
            scalar_t *__restrict__ fMomMean,
            const label_t step,
            const scalar_t invCount,
            const scalar_t meanCounterScalar)
        {
            // Precompute index components
            const label_t nx = block::nx();
            const label_t ny = block::ny();
            const label_t nz = block::nz();
            const label_t nMom = NUMBER_MOMENTS();

            // Precompute block offset
            const label_t blockOffset = blockIdx.x + d_NUM_BLOCK_X * (blockIdx.y + d_NUM_BLOCK_Y * blockIdx.z);

            // Precompute thread offset within block
            const label_t threadOffset = threadIdx.x + nx * (threadIdx.y + ny * threadIdx.z);

            // Precompute moment stride (cells per moment slice)
            const label_t momentStride = nx * ny * nz;

            // Base index without moment component
            const label_t baseIndex = threadOffset + momentStride * nMom * blockOffset;

            // Helper macro for moment-specific indexing
            // #define MOM_INDEX(m) (baseIndex + (m) * momentStride)

            // Pipeline stages
            const label_t idx_rho = MOM_INDEX(baseIndex, index::rho(), momentStride);
            const scalar_t c_rho = fMom[idx_rho];
            const scalar_t m_rho = fMomMean[idx_rho];

            const label_t idx_u = MOM_INDEX(baseIndex, index::u(), momentStride);
            const scalar_t c_u = fMom[idx_u];
            const scalar_t m_u = fMomMean[idx_u];

            // Compute rho, Fetch v
            const scalar_t n_rho = fma(m_rho, meanCounterScalar, c_rho) * invCount;
            const label_t idx_v = MOM_INDEX(baseIndex, index::v(), momentStride);
            const scalar_t c_v = fMom[idx_v];
            const scalar_t m_v = fMomMean[idx_v];

            // Compute u, Fetch w, Write rho
            const scalar_t n_u = fma(m_u, meanCounterScalar, c_u) * invCount;
            const label_t idx_w = MOM_INDEX(baseIndex, index::w(), momentStride);
            const scalar_t c_w = fMom[idx_w];
            const scalar_t m_w = fMomMean[idx_w];
            fMomMean[idx_rho] = n_rho;

            // Compute v, Fetch xx, Write u
            const scalar_t n_v = fma(m_v, meanCounterScalar, c_v) * invCount;
            const label_t idx_xx = MOM_INDEX(baseIndex, index::xx(), momentStride);
            const scalar_t c_xx = fMom[idx_xx];
            const scalar_t m_xx = fMomMean[idx_xx];
            fMomMean[idx_u] = n_u;

            // Compute w, Fetch xy, Write v
            const scalar_t n_w = fma(m_w, meanCounterScalar, c_w) * invCount;
            const label_t idx_xy = MOM_INDEX(baseIndex, index::xy(), momentStride);
            const scalar_t c_xy = fMom[idx_xy];
            const scalar_t m_xy = fMomMean[idx_xy];
            fMomMean[idx_v] = n_v;

            // Compute xx, Fetch xz, Write w
            const scalar_t n_xx = fma(m_xx, meanCounterScalar, c_xx) * invCount;
            const label_t idx_xz = MOM_INDEX(baseIndex, index::xz(), momentStride);
            const scalar_t c_xz = fMom[idx_xz];
            const scalar_t m_xz = fMomMean[idx_xz];
            fMomMean[idx_w] = n_w;

            // Compute xy, Fetch yy, Write xx
            const scalar_t n_xy = fma(m_xy, meanCounterScalar, c_xy) * invCount;
            const label_t idx_yy = MOM_INDEX(baseIndex, index::yy(), momentStride);
            const scalar_t c_yy = fMom[idx_yy];
            const scalar_t m_yy = fMomMean[idx_yy];
            fMomMean[idx_xx] = n_xx;

            // Compute xz, Fetch yz, Write xy
            const scalar_t n_xz = fma(m_xz, meanCounterScalar, c_xz) * invCount;
            const label_t idx_yz = MOM_INDEX(baseIndex, index::yz(), momentStride);
            const scalar_t c_yz = fMom[idx_yz];
            const scalar_t m_yz = fMomMean[idx_yz];
            fMomMean[idx_xy] = n_xy;

            // Compute yy, Fetch zz, Write xz
            const scalar_t n_yy = fma(m_yy, meanCounterScalar, c_yy) * invCount;
            const label_t idx_zz = MOM_INDEX(baseIndex, index::zz(), momentStride);
            const scalar_t c_zz = fMom[idx_zz];
            const scalar_t m_zz = fMomMean[idx_zz];
            fMomMean[idx_xz] = n_xz;

            // Final computations and writes
            const scalar_t n_yz = fma(m_yz, meanCounterScalar, c_yz) * invCount;
            fMomMean[idx_yy] = n_yy;

            const scalar_t n_zz = fma(m_zz, meanCounterScalar, c_zz) * invCount;
            fMomMean[idx_yz] = n_yz;

            fMomMean[idx_zz] = n_zz;

            // Cleanup macro
            // #undef MOM_INDEX
        }
    }
}

#endif