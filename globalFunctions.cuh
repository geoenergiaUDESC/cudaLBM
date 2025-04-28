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
        __host__ __device__ [[nodiscard]] inline consteval T nx() noexcept
        {
            return 8;
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T ny() noexcept
        {
            return 8;
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T nz() noexcept
        {
            return 8;
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T size() noexcept
        {
            return nx<T>() * ny<T>() * nz<T>();
        }

        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr T nxBlocks(const T n) noexcept
        {
            return n / nx<T>();
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr T nyBlocks(const T n) noexcept
        {
            return n / ny<T>();
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr T nzBlocks(const T n) noexcept
        {
            return n / nz<T>();
        }
        template <typename T, class M>
        __host__ __device__ [[nodiscard]] inline constexpr T nBlocks(const M &mesh) noexcept
        {
            return (mesh.nx() / nx<T>()) * (mesh.ny() / ny<T>()) * (mesh.nz() / nz<T>());
        }

        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T West() noexcept
        {
            return 0;
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T East() noexcept
        {
            return block::nx<T>() - 1;
        }

        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T South() noexcept
        {
            return 0;
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T North() noexcept
        {
            return block::ny<T>() - 1;
        }

        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T Back() noexcept
        {
            return 0;
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline consteval T Front() noexcept
        {
            return block::nz<T>() - 1;
        }

    }

    template <typename T>
    __host__ __device__ [[nodiscard]] inline constexpr T blockLabel(const T i, const T j, const T k, const blockLabel_t &blockLabel) noexcept
    {
        return (k * (blockLabel.nx * blockLabel.ny)) + (j * blockLabel.nx) + (i);
    }

    template <class M>
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxMom(
        const std::size_t threadIDx, const std::size_t threadIDy, const std::size_t threadIDz,
        const std::size_t blockIDx, const std::size_t blockIDy, const std::size_t blockIDz,
        const M &mesh)
    {
        const std::size_t nxBlock = block::nxBlocks<std::size_t>(mesh.nx());
        const std::size_t nyBlock = block::nyBlocks<std::size_t>(mesh.ny());

        return threadIDx + block::nx<std::size_t>() * (threadIDy + block::ny<std::size_t>() * (threadIDz + block::nz<std::size_t>() * ((blockIDx + nxBlock * (blockIDy + nyBlock * (blockIDz))))));
    }

    template <const std::size_t pop>
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxPopBlock(const std::size_t threadIDx, const std::size_t threadIDy, const std::size_t threadIDz)
    {
        return threadIDx + block::nx<std::size_t>() * (threadIDy + block::ny<std::size_t>() * (threadIDz + block::nz<std::size_t>() * (pop)));
    }

    template <const label_t QF, const label_t pop, class M>
    __device__ [[nodiscard]] inline constexpr std::size_t idxPopX(
        const std::size_t ty,
        const std::size_t tz,
        const std::size_t bx,
        const std::size_t by,
        const std::size_t bz,
        const M &mesh)
    {

        /*idx //  D   pop //  D   pop
        D3Q19
        0   //  1   1   //  -1  2
        1   //  1   7   //  -1  8
        2   //  1   9   //  -1  10
        3   //  1   13  //  -1  14
        4   //  1   15  //  -1  16
        D3Q27
        6   //  1   19  //  -1  20
        7   //  1   21  //  -1  22
        8   //  1   23  //  -1  24
        9   //  1   26  //  -1  25
        */

        return ty + block::ny<std::size_t>() * (tz + block::nz<std::size_t>() * (pop + QF * (bx + block::nxBlocks<std::size_t>(mesh.nx()) * (by + block::nyBlocks<std::size_t>(mesh.ny()) * bz))));
    }

    template <const label_t QF, const label_t pop, class M>
    __device__ [[nodiscard]] inline constexpr std::size_t idxPopY(
        const std::size_t tx,
        const std::size_t tz,
        const std::size_t bx,
        const std::size_t by,
        const std::size_t bz,
        const M &mesh)
    {

        /*
        idx //  D   pop //  D   pop
        D3Q19
        0   //  1   3   //  -1  4
        1   //  1   7   //  -1  8
        2   //  1   11  //  -1  12
        3   //  1   14  //  -1  13
        4   //  1   17  //  -1  18
        D3Q27
        6   //  1   19  //  -1  20
        7   //  1   21  //  -1  22
        8   //  1   24  //  -1  23
        9   //  1   25  //  -1  26
        */
        return tx + block::nx<std::size_t>() * (tz + block::nz<std::size_t>() * (pop + QF * (bx + block::nxBlocks<std::size_t>(mesh.nx()) * (by + block::nyBlocks<std::size_t>(mesh.ny()) * bz))));
    }

    template <const label_t QF, const label_t pop, class M>
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxPopZ(
        const std::size_t tx,
        const std::size_t ty,
        const std::size_t bx,
        const std::size_t by,
        const std::size_t bz,
        const M &mesh)
    {

        /*
        idx //  D   pop //  D   pop
        D3Q19
        0   //  1   5   //  -1  6
        1   //  1   9   //  -1  10
        2   //  1   11  //  -1  12
        3   //  1   16  //  -1  15
        4   //  1   18  //  -1  17
        D3Q27
        6   //  1   19  //  -1  20
        7   //  1   22  //  -1  21
        8   //  1   23  //  -1  24
        9   //  1   25  //  -1  26
        */

        return tx + block::nx<std::size_t>() * (ty + block::ny<std::size_t>() * (pop + QF * (bx + block::nxBlocks<std::size_t>(mesh.nx()) * (by + block::nyBlocks<std::size_t>(mesh.ny()) * bz))));
    }
}

#endif