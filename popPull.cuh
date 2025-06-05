#ifndef POPPULL_CUH
#define POPPULL_CUH

{
    const label_t xp1 = (threadIdx.x + 1 + block::nx()) % block::nx();
    const label_t xm1 = (threadIdx.x - 1 + block::nx()) % block::nx();
    const label_t yp1 = (threadIdx.y + 1 + block::ny()) % block::ny();
    const label_t ym1 = (threadIdx.y - 1 + block::ny()) % block::ny();
    const label_t zp1 = (threadIdx.z + 1 + block::nz()) % block::nz();
    const label_t zm1 = (threadIdx.z - 1 + block::nz()) % block::nz();

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
}

#endif