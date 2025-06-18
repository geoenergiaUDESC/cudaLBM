/**
Filename: sharedMemory.cuh
Contents: Handles the use of shared memory on the GPU
**/

#ifndef __MBLBM_SHAREDMEMORY_CUH
#define __MBLBM_SHAREDMEMORY_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

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
        __device__ static inline void save(
            const scalar_t *const ptrRestrict pop,
            scalar_t *const ptrRestrict s_pop) noexcept
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

            __syncthreads();
        }

        /**
         * @brief Pulls the population density from shared memory
         * @param pop The population density at the current thread
         * @param s_pop The population density in shared memory
         **/
        __device__ static inline void pull(
            scalar_t *const ptrRestrict pop,
            const scalar_t *const ptrRestrict s_pop) noexcept
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

    private:
    };

}

#endif