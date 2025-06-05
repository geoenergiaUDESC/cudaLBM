#ifndef POPSAVE_CUH
#define POPSAVE_CUH

{
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
}

__syncthreads();

#endif