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

    namespace block
    {
        template <typename T>
        [[nodiscard]] inline consteval T nx() noexcept { return 8; }
        template <typename T>
        [[nodiscard]] inline consteval T ny() noexcept { return 8; }
        template <typename T>
        [[nodiscard]] inline consteval T nz() noexcept { return 8; }
        template <typename T>
        [[nodiscard]] inline consteval T size() noexcept { return nx<T>() * ny<T>() * nz<T>(); }
    }

    template <typename T>
    [[nodiscard]] inline constexpr T blockLabel(const T i, const T j, const T k, const blockLabel_t &blockLabel) noexcept
    {
        return (k * (blockLabel.nx * blockLabel.ny)) + (j * blockLabel.nx) + (i);
    }
    [[nodiscard]] __host__ __device__ inline constexpr std::size_t idxMom(const std::size_t threadIDx, const std::size_t threadIDy, const std::size_t threadIDz, const std::size_t blockIDx, const std::size_t blockIDy, const std::size_t blockIDz, const std::size_t nx, const std::size_t ny, const std::size_t nz)
    {
        const std::size_t nxBlock = nx / block::nx<std::size_t>();
        const std::size_t nyBlock = ny / block::ny<std::size_t>();
        return threadIDx + block::nx<std::size_t>() * (threadIDy + block::ny<std::size_t>() * (threadIDz + block::nz<std::size_t>() * ((blockIDx + nxBlock * (blockIDy + nyBlock * (blockIDz))))));
    }
    [[nodiscard]] __host__ __device__ inline constexpr std::size_t idxPopBlock(const std::size_t threadIDx, const std::size_t threadIDy, const std::size_t threadIDz, const std::size_t pop)
    {
        return threadIDx + block::nx<std::size_t>() * (threadIDy + block::ny<std::size_t>() * (threadIDz + block::nz<std::size_t>() * (pop)));
    }
}

#endif