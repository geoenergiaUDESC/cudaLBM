#ifndef __LBM_PRINTONCE_CUH
#define __LBM_PRINTONCE_CUH

#include <cuda_runtime.h>
#include <cstdio>

namespace LBM
{
    __device__ int printedFallback[64] = {false};

    __device__ inline void printOnce(const uint8_t caseId, const char* name)
    {
        if (atomicExch(&printedFallback[caseId], true) == false)
            printf("[fallback] %s applied\n", name);
    }
}

#endif
