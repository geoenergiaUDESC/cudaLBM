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
    __host__ __device__ [[nodiscard]] inline constexpr scalar_t omega(const scalar_t Re, const scalar_t u_inf, const label_t nx) noexcept
    {
        return 1.0 / (0.5 + 3.0 * (u_inf * static_cast<scalar_t>(nx) / Re));
    }

    /**
     * @brief Struct holding the number of lattice elements in three dimensions
     **/
    struct blockLabel_t
    {
        const label_t nx;
        const label_t ny;
        const label_t nz;
    };

    /**
     * @brief Struct holding the first and last indices of a particular dimension of a block
     **/
    struct blockPartitionRange_t
    {
        const label_t begin;
        const label_t end;
    };

    /**
     * @brief Struct holding first and last indices of all dimensions of a block
     **/
    struct blockRange_t
    {
        const blockPartitionRange_t xRange;
        const blockPartitionRange_t yRange;
        const blockPartitionRange_t zRange;
    };

    namespace block
    {
        /**
         * @brief CUDA block size parameters
         * @return Dimensions of CUDA blocks
         **/
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

        /**
         * @brief Calculate the number of blocks in a given direction from a number of lattice points
         * @return Number of blocks in a given direction
         * @param n Number of lattice points in a given direction
         **/
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

        /**
         * @brief Provide the block-relative indices of block boundaries
         * @return The x, y or z index in a given direction corresponding to a block boundary
         **/
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

        /**
         * @brief Check whether a particular lattice lies along a block boundary
         * @return TRUE if n lies along the boundary, FALSE otherwise
         * @param n The lattice index to be checked
         **/
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr bool South(const T n) noexcept
        {
            return (n == South<T>());
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr bool West(const T n) noexcept
        {
            return (n == West<T>());
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr bool East(const T n) noexcept
        {
            return (n == East<T>());
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr bool North(const T n) noexcept
        {
            return (n == North<T>());
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr bool Back(const T n) noexcept
        {
            return (n == Back<T>());
        }
        template <typename T>
        __host__ __device__ [[nodiscard]] inline constexpr bool Front(const T n) noexcept
        {
            return (n == Front<T>());
        }
    }

    /**
     * @brief Returns the flattened block-relative index of a point expressed in 3 dimensions
     * @return The index of x, y, z expressed in flattened indices
     * @param x The x-component of the block
     * @param y The y-component of the block
     * @param z The z-component of the block
     * @param blockDimensions The physical dimensions of the block
     **/
    template <typename T>
    __host__ __device__ [[nodiscard]] inline constexpr T blockLabel(const T x, const T y, const T z, const blockLabel_t &blockDimensions) noexcept
    {
        return (z * (blockDimensions.nx * blockDimensions.ny)) + (y * blockDimensions.nx) + (x);
    }

    /**
     * @brief Returns the flattened block-relative index of a point expressed in 3 dimensions
     * @return The index of x, y, z expressed in flattened indices
     * @param x The x-component of the block
     * @param y The y-component of the block
     * @param z The z-component of the block
     * @param mesh The mesh
     **/
    template <typename T, class M>
    __host__ __device__ [[nodiscard]] inline constexpr T blockLabel(const T x, const T y, const T z, const M &mesh) noexcept
    {
        return (z * (mesh.nx() * mesh.ny())) + (y * mesh.nx()) + (x);
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

    __host__ __device__ [[nodiscard]] inline std::size_t idxScalarBlock(
        const std::size_t tx,
        const std::size_t ty,
        const std::size_t tz,
        const std::size_t bx,
        const std::size_t by,
        const std::size_t bz)
    {
        return tx + block::nx<std::size_t>() * (ty + block::ny<std::size_t>() * (tz + block::nz<std::size_t>() * (bx + block::nx<std::size_t>() * (by + block::ny<std::size_t>() * (bz)))));
    }

    template <const std::size_t pop>
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxPopBlock(
        const std::size_t threadIDx, const std::size_t threadIDy, const std::size_t threadIDz)
    {
        return threadIDx + block::nx<std::size_t>() * (threadIDy + block::ny<std::size_t>() * (threadIDz + block::nz<std::size_t>() * (pop)));
    }

    template <class M>
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxPopX(
        const std::size_t ty,
        const std::size_t tz,
        const std::size_t bx,
        const std::size_t by,
        const std::size_t bz,
        const std::size_t QF,
        const std::size_t pop,
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

    template <class M>
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxPopY(
        const std::size_t tx,
        const std::size_t tz,
        const std::size_t bx,
        const std::size_t by,
        const std::size_t bz,
        const std::size_t QF,
        const std::size_t pop,
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

    template <class M>
    __host__ __device__ [[nodiscard]] inline constexpr std::size_t idxPopZ(
        const std::size_t tx,
        const std::size_t ty,
        const std::size_t bx,
        const std::size_t by,
        const std::size_t bz,
        const std::size_t QF,
        const std::size_t pop,
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