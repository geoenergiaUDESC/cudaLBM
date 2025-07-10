/**
Filename: sharedMemory.cuh
Contents: Handles the use of shared memory on the GPU
**/

#ifndef __MBLBM_SHAREDMEMORY_CUH
#define __MBLBM_SHAREDMEMORY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    class sharedMemory
    {
    public:
        /**
         * @brief Default constructor for the sharedMemory class
         **/
        [[nodiscard]] inline consteval sharedMemory() {};

        /**
         * @brief Saves the thread population density to the block shared memory
         * @param pop The population density at the current thread
         * @param s_pop The population density in shared memory
         **/
        template <class VSet>
        __device__ static inline void save(
            const scalar_t (&ptrRestrict pop)[VSet::Q()],
            scalar_t (&ptrRestrict s_pop)[block::size()][(VSet::Q() - 1)]) noexcept
        {
            constexpr label_t nx = block::nx();
            constexpr label_t ny = block::ny();
            const label_t linear_idx = threadIdx.x + nx * (threadIdx.y + ny * threadIdx.z);

            constexpr_for<0, 18>(                        //
                [&](const auto q_)                       //
                {                                        //
                    s_pop[linear_idx][q_] = pop[q_ + 1]; //
                });                                      //
            __syncthreads();

            // s_pop[0][linear_idx] = pop[1];
            // s_pop[1][linear_idx] = pop[2];
            // s_pop[2][linear_idx] = pop[3];
            // s_pop[3][linear_idx] = pop[4];
            // s_pop[4][linear_idx] = pop[5];
            // s_pop[5][linear_idx] = pop[6];
            // s_pop[6][linear_idx] = pop[7];
            // s_pop[7][linear_idx] = pop[8];
            // s_pop[8][linear_idx] = pop[9];
            // s_pop[9][linear_idx] = pop[10];
            // s_pop[10][linear_idx] = pop[11];
            // s_pop[11][linear_idx] = pop[12];
            // s_pop[12][linear_idx] = pop[13];
            // s_pop[13][linear_idx] = pop[14];
            // s_pop[14][linear_idx] = pop[15];
            // s_pop[15][linear_idx] = pop[16];
            // s_pop[16][linear_idx] = pop[17];
            // s_pop[17][linear_idx] = pop[18];
            // __syncthreads();

            // constexpr_for<0, 18>(                                            //
            //     [&](const auto q_)                                           //
            //     {                                                            //
            //         s_pop[device::idxPopBlock<q_>(threadIdx)] = pop[q_ + 1]; //
            //     });                                                          //

            // s_pop[device::idxPopBlock<0>(threadIdx)] = pop[1];
            // s_pop[device::idxPopBlock<1>(threadIdx)] = pop[2];
            // s_pop[device::idxPopBlock<2>(threadIdx)] = pop[3];
            // s_pop[device::idxPopBlock<3>(threadIdx)] = pop[4];
            // s_pop[device::idxPopBlock<4>(threadIdx)] = pop[5];
            // s_pop[device::idxPopBlock<5>(threadIdx)] = pop[6];
            // s_pop[device::idxPopBlock<6>(threadIdx)] = pop[7];
            // s_pop[device::idxPopBlock<7>(threadIdx)] = pop[8];
            // s_pop[device::idxPopBlock<8>(threadIdx)] = pop[9];
            // s_pop[device::idxPopBlock<9>(threadIdx)] = pop[10];
            // s_pop[device::idxPopBlock<10>(threadIdx)] = pop[11];
            // s_pop[device::idxPopBlock<11>(threadIdx)] = pop[12];
            // s_pop[device::idxPopBlock<12>(threadIdx)] = pop[13];
            // s_pop[device::idxPopBlock<13>(threadIdx)] = pop[14];
            // s_pop[device::idxPopBlock<14>(threadIdx)] = pop[15];
            // s_pop[device::idxPopBlock<15>(threadIdx)] = pop[16];
            // s_pop[device::idxPopBlock<16>(threadIdx)] = pop[17];
            // s_pop[device::idxPopBlock<17>(threadIdx)] = pop[18];

            // __syncthreads();
        }

        /**
         * @brief Pulls the population density from shared memory
         * @param pop The population density at the current thread
         * @param s_pop The population density in shared memory
         **/
        template <class VSet>
        __device__ static inline void pull(
            scalar_t (&ptrRestrict pop)[VSet::Q()],
            const scalar_t (&ptrRestrict s_pop)[(VSet::Q() - 1)][block::size()]) noexcept
        {
            constexpr label_t nx = block::nx();
            constexpr label_t ny = block::ny();
            constexpr label_t nz = block::nz();
            constexpr label_t nx_mask = nx - 1;
            constexpr label_t ny_mask = ny - 1;
            constexpr label_t nz_mask = nz - 1;

            const label_t x = threadIdx.x;
            const label_t y = threadIdx.y;
            const label_t z = threadIdx.z;

            const label_t xm1 = (x - 1) & nx_mask;
            const label_t xp1 = (x + 1) & nx_mask;
            const label_t ym1 = (y - 1) & ny_mask;
            const label_t yp1 = (y + 1) & ny_mask;
            const label_t zm1 = (z - 1) & nz_mask;
            const label_t zp1 = (z + 1) & nz_mask;

            auto idx = [nx, ny](label_t x, label_t y, label_t z)
            {
                return x + nx * (y + ny * z);
            };

            pop[1] = s_pop[0][idx(xm1, y, z)];
            pop[2] = s_pop[1][idx(xp1, y, z)];
            pop[3] = s_pop[2][idx(x, ym1, z)];
            pop[4] = s_pop[3][idx(x, yp1, z)];
            pop[5] = s_pop[4][idx(x, y, zm1)];
            pop[6] = s_pop[5][idx(x, y, zp1)];
            pop[7] = s_pop[6][idx(xm1, ym1, z)];
            pop[8] = s_pop[7][idx(xp1, yp1, z)];
            pop[9] = s_pop[8][idx(xm1, y, zm1)];
            pop[10] = s_pop[9][idx(xp1, y, zp1)];
            pop[11] = s_pop[10][idx(x, ym1, zm1)];
            pop[12] = s_pop[11][idx(x, yp1, zp1)];
            pop[13] = s_pop[12][idx(xm1, yp1, z)];
            pop[14] = s_pop[13][idx(xp1, ym1, z)];
            pop[15] = s_pop[14][idx(xm1, y, zp1)];
            pop[16] = s_pop[15][idx(xp1, y, zm1)];
            pop[17] = s_pop[16][idx(x, ym1, zp1)];
            pop[18] = s_pop[17][idx(x, yp1, zm1)];

            // const label_t xp1 = (threadIdx.x + 1 + block::nx()) % block::nx();
            // const label_t xm1 = (threadIdx.x - 1 + block::nx()) % block::nx();

            // const label_t yp1 = (threadIdx.y + 1 + block::ny()) % block::ny();
            // const label_t ym1 = (threadIdx.y - 1 + block::ny()) % block::ny();

            // const label_t zp1 = (threadIdx.z + 1 + block::nz()) % block::nz();
            // const label_t zm1 = (threadIdx.z - 1 + block::nz()) % block::nz();

            // pop[1] = s_pop[device::idxPopBlock<0>(xm1, threadIdx.y, threadIdx.z)];
            // pop[2] = s_pop[device::idxPopBlock<1>(xp1, threadIdx.y, threadIdx.z)];
            // pop[3] = s_pop[device::idxPopBlock<2>(threadIdx.x, ym1, threadIdx.z)];
            // pop[4] = s_pop[device::idxPopBlock<3>(threadIdx.x, yp1, threadIdx.z)];
            // pop[5] = s_pop[device::idxPopBlock<4>(threadIdx.x, threadIdx.y, zm1)];
            // pop[6] = s_pop[device::idxPopBlock<5>(threadIdx.x, threadIdx.y, zp1)];
            // pop[7] = s_pop[device::idxPopBlock<6>(xm1, ym1, threadIdx.z)];
            // pop[8] = s_pop[device::idxPopBlock<7>(xp1, yp1, threadIdx.z)];
            // pop[9] = s_pop[device::idxPopBlock<8>(xm1, threadIdx.y, zm1)];
            // pop[10] = s_pop[device::idxPopBlock<9>(xp1, threadIdx.y, zp1)];
            // pop[11] = s_pop[device::idxPopBlock<10>(threadIdx.x, ym1, zm1)];
            // pop[12] = s_pop[device::idxPopBlock<11>(threadIdx.x, yp1, zp1)];
            // pop[13] = s_pop[device::idxPopBlock<12>(xm1, yp1, threadIdx.z)];
            // pop[14] = s_pop[device::idxPopBlock<13>(xp1, ym1, threadIdx.z)];
            // pop[15] = s_pop[device::idxPopBlock<14>(xm1, threadIdx.y, zp1)];
            // pop[16] = s_pop[device::idxPopBlock<15>(xp1, threadIdx.y, zm1)];
            // pop[17] = s_pop[device::idxPopBlock<16>(threadIdx.x, ym1, zp1)];
            // pop[18] = s_pop[device::idxPopBlock<17>(threadIdx.x, yp1, zm1)];
        }

        template <class VSet>
        __device__ inline static void pull_v2(
            scalar_t (&ptrRestrict pop)[VSet::Q()],
            const scalar_t (&ptrRestrict s_pop)[block::size()][VSet::Q() - 1])
        {
            constexpr const label_t nx = block::nx();
            constexpr const label_t ny = block::ny();
            constexpr const label_t nz = block::nz();
            constexpr const label_t nx_mask = nx - 1;
            constexpr const label_t ny_mask = ny - 1;
            constexpr const label_t nz_mask = nz - 1;

            const label_t x = threadIdx.x;
            const label_t y = threadIdx.y;
            const label_t z = threadIdx.z;

            // Calculate periodic neighbors using bitwise AND (requires power-of-two dimensions)
            const label_t xm1 = (x - 1) & nx_mask;
            const label_t xp1 = (x + 1) & nx_mask;
            const label_t ym1 = (y - 1) & ny_mask;
            const label_t yp1 = (y + 1) & ny_mask;
            const label_t zm1 = (z - 1) & nz_mask;
            const label_t zp1 = (z + 1) & nz_mask;

            // Lambda for linear index calculation
            auto idx = [nx, ny](label_t x, label_t y, label_t z)
            {
                return x + nx * (y + ny * z);
            };

            // Full neighbor index array for D3Q19
            const label_t neighbor_idx[19] = {
                idx(x, y, z),
                idx(xm1, y, z),
                idx(xp1, y, z),
                idx(x, ym1, z),
                idx(x, yp1, z),
                idx(x, y, zm1),
                idx(x, y, zp1),
                idx(xm1, ym1, z),
                idx(xp1, yp1, z),
                idx(xm1, y, zm1),
                idx(xp1, y, zp1),
                idx(x, ym1, zm1),
                idx(x, yp1, zp1),
                idx(xm1, yp1, z),
                idx(xp1, ym1, z),
                idx(xm1, y, zp1),
                idx(xp1, y, zm1),
                idx(x, ym1, zp1),
                idx(x, yp1, zm1)};

            // #pragma unroll
            //             for (label_t i = 1; i < VSet::Q(); i++)
            //             {
            //                 pop[i] = s_pop[idx[i]][i - 1];
            //             }

            constexpr_for<1, VSet::Q() - 1>(                   //
                [&](const auto q_)                             //
                {                                              //
                    pop[q_] = s_pop[neighbor_idx[q_]][q_ - 1]; //
                });                                            //
        }

    private:
    };

}

#endif