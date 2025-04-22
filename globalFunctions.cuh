/**
Filename: globalFunctions.cuh
Contents: Functions used throughout the source code
**/

#ifndef __MBLBM_GLOBALFUNCTIONS_CUH
#define __MBLBM_GLOBALFUNCTIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    struct blockLabel_t
    {
        const label_t nx;
        const label_t ny;
        const label_t nz;
    };

    struct blockPartitionRange_t
    {
        const label_t begin;
        const label_t end;
    };

    struct blockRange_t
    {
        const blockPartitionRange_t xRange;
        const blockPartitionRange_t yRange;
        const blockPartitionRange_t zRange;
    };

    template <typename T>
    [[nodiscard]] inline constexpr T blockLabel(const T i, const T j, const T k, const blockLabel_t &blockLabel) noexcept
    {
        return (k * (blockLabel.nx * blockLabel.ny)) + (j * blockLabel.nx) + (i);
    }
}

#endif