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
        __host__ __device__ [[nodiscard]] inline consteval T nx() noexcept { return 8; }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T ny() noexcept { return 8; }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T nz() noexcept { return 8; }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T size() noexcept { return nx<T>() * ny<T>() * nz<T>(); }

        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr T nxBlocks(const T n) noexcept { return n / nx<T>(); }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr T nyBlocks(const T n) noexcept { return n / ny<T>(); }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr T nzBlocks(const T n) noexcept { return n / nz<T>(); }

    }

    template <typename T>
    __host__ __device__ [[nodiscard]] inline constexpr T blockLabel(const T i, const T j, const T k, const blockLabel_t &blockLabel) noexcept
    {
        return (k * (blockLabel.nx * blockLabel.ny)) + (j * blockLabel.nx) + (i);
    }
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxMom(const std::size_t threadIDx, const std::size_t threadIDy, const std::size_t threadIDz, const std::size_t blockIDx, const std::size_t blockIDy, const std::size_t blockIDz, const std::size_t nx, const std::size_t ny, const std::size_t nz)
    {
        const std::size_t nxBlock = block::nxBlocks<std::size_t>(nx);
        const std::size_t nyBlock = block::nyBlocks<std::size_t>(ny);

        return threadIDx + block::nx<std::size_t>() * (threadIDy + block::ny<std::size_t>() * (threadIDz + block::nz<std::size_t>() * ((blockIDx + nxBlock * (blockIDy + nyBlock * (blockIDz))))));
    }
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxPopBlock(const std::size_t threadIDx, const std::size_t threadIDy, const std::size_t threadIDz, const std::size_t pop)
    {
        return threadIDx + block::nx<std::size_t>() * (threadIDy + block::ny<std::size_t>() * (threadIDz + block::nz<std::size_t>() * (pop)));
    }
}

#endif