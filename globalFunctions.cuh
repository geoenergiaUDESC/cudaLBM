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
    __host__ [[nodiscard]] inline constexpr scalar_t omega(const scalar_t Re, const scalar_t u_inf, const label_t nx) noexcept
    {
        return static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * (u_inf * static_cast<scalar_t>(nx) / Re));
    }

    // __device__ [[nodiscard]] inline scalar_t omega() noexcept
    // {
    //     return static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * (d_u_inf * static_cast<scalar_t>(d_nx) / d_Re));
    // }

    __device__ [[nodiscard]] inline scalar_t omega() noexcept
    {
        return static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * (0.05 * static_cast<scalar_t>(128.0) / 500.0));
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

        __host__ __device__ [[nodiscard]] inline consteval label_t nx() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t ny() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t nz() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t size() noexcept
        {
            return nx() * ny() * nz();
        }
        __host__ __device__ [[nodiscard]] inline consteval dim3 threadBlock() noexcept
        {
            return {nx(), ny(), nz()};
        }
        template <typename M>
        __host__ __device__ [[nodiscard]] inline constexpr dim3 gridBlock(const M &mesh) noexcept
        {
            return {static_cast<uint32_t>(mesh.nx() / nx()), static_cast<uint32_t>(mesh.ny() / ny()), static_cast<uint32_t>(mesh.nz() / nz())};
        }

        /**
         * @brief Calculate the number of blocks in a given direction from a number of lattice points
         * @return Number of blocks in a given direction
         * @param n Number of lattice points in a given direction
         **/

        // __host__ __device__ [[nodiscard]] inline constexpr label_t nxBlocks(const label_t n) noexcept
        // {
        //     return n / nx();
        // }

        // __host__ __device__ [[nodiscard]] inline constexpr label_t nyBlocks(const label_t n) noexcept
        // {
        //     return n / ny();
        // }

        // __host__ __device__ [[nodiscard]] inline constexpr label_t nzBlocks(const label_t n) noexcept
        // {
        //     return n / nz();
        // }

        /**
         * @brief Provide the block-relative indices of block boundaries
         * @return The x, y or z index in a given direction corresponding to a block boundary
         **/

        __host__ __device__ [[nodiscard]] inline consteval label_t West() noexcept
        {
            return 0;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t East() noexcept
        {
            return block::nx() - 1;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t South() noexcept
        {
            return 0;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t North() noexcept
        {
            return block::ny() - 1;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t Back() noexcept
        {
            return 0;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t Front() noexcept
        {
            return block::nz() - 1;
        }

        /**
         * @brief Check whether a particular lattice lies along a block boundary
         * @return TRUE if n lies along the boundary, FALSE otherwise
         * @param n The lattice index to be checked
         **/

        __host__ __device__ [[nodiscard]] inline constexpr bool South(const label_t n) noexcept
        {
            return (n == South());
        }

        __host__ __device__ [[nodiscard]] inline constexpr bool West(const label_t n) noexcept
        {
            return (n == West());
        }

        __host__ __device__ [[nodiscard]] inline constexpr bool East(const label_t n) noexcept
        {
            return (n == East());
        }

        __host__ __device__ [[nodiscard]] inline constexpr bool North(const label_t n) noexcept
        {
            return (n == North());
        }

        __host__ __device__ [[nodiscard]] inline constexpr bool Back(const label_t n) noexcept
        {
            return (n == Back());
        }

        __host__ __device__ [[nodiscard]] inline constexpr bool Front(const label_t n) noexcept
        {
            return (n == Front());
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

    __host__ __device__ [[nodiscard]] inline constexpr label_t blockLabel(const label_t x, const label_t y, const label_t z, const blockLabel_t &blockDimensions) noexcept
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
    template <class M>
    __host__ __device__ [[nodiscard]] inline constexpr label_t blockLabel(const label_t x, const label_t y, const label_t z, const M &mesh) noexcept
    {
        return (z * (mesh.nx() * mesh.ny())) + (y * mesh.nx()) + (x);
    }

    __host__ [[nodiscard]] inline label_t idxMom(
        const label_t tx, const label_t ty, const label_t tz,
        const label_t bx, const label_t by, const label_t bz,
        const label_t nx, const label_t ny)
    {
        const label_t nxBlock = nx / block::nx();
        const label_t nyBlock = ny / block::ny();

        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nxBlock * (by + nyBlock * (bz)))));
    }

    __device__ [[nodiscard]] inline label_t idxMom(
        const label_t tx,
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz)))));
    }

    template <const label_t mom, const label_t NUMBER_MOMENTS>
    __device__ [[nodiscard]] inline label_t idxMom__(
        const label_t tx, const label_t ty, const label_t tz,
        const label_t bx, const label_t by, const label_t bz)
    {
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (mom + NUMBER_MOMENTS * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz))))));
    }

    __device__ [[nodiscard]] inline label_t idxScalarBlock(
        const label_t tx,
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz)))));
    }

    template <const label_t pop>
    __host__ __device__ [[nodiscard]] inline constexpr label_t idxPopBlock(
        const label_t tx, const label_t ty, const label_t tz)
    {
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (pop)));
    }

    template <const label_t pop, const label_t QF>
    __device__ [[nodiscard]] inline label_t idxPopX(
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
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

        return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    template <const label_t pop, const label_t QF>
    __device__ [[nodiscard]] inline label_t idxPopY(
        const label_t tx,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
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
        return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    template <const label_t pop, const label_t QF>
    __device__ [[nodiscard]] inline label_t idxPopZ(
        const label_t tx,
        const label_t ty,
        const label_t bx,
        const label_t by,
        const label_t bz)
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

        return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }
}

#endif