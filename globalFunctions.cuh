/**
Filename: globalFunctions.cuh
Contents: Functions used throughout the source code
**/

#ifndef __MBLBM_GLOBALFUNCTIONS_CUH
#define __MBLBM_GLOBALFUNCTIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    [[nodiscard]] inline consteval label_t NUMBER_MOMENTS() { return 10; }

    constexpr const label_t BULK = (0b00000000000000000000000000000000);
    constexpr const label_t NORTH = (0b00000000000000000000000011001100);
    constexpr const label_t SOUTH = (0b00000000000000000000000000110011);
    constexpr const label_t WEST = (0b00000000000000000000000001010101);
    constexpr const label_t EAST = (0b00000000000000000000000010101010);
    constexpr const label_t FRONT = (0b00000000000000000000000011110000);
    constexpr const label_t BACK = (0b00000000000000000000000000001111);
    constexpr const label_t NORTH_WEST = (0b00000000000000000000000011011101);
    constexpr const label_t NORTH_EAST = (0b00000000000000000000000011101110);
    constexpr const label_t NORTH_FRONT = (0b00000000000000000000000011111100);
    constexpr const label_t NORTH_BACK = (0b00000000000000000000000011001111);
    constexpr const label_t SOUTH_WEST = (0b00000000000000000000000001110111);
    constexpr const label_t SOUTH_EAST = (0b00000000000000000000000010111011);
    constexpr const label_t SOUTH_FRONT = (0b00000000000000000000000011110011);
    constexpr const label_t SOUTH_BACK = (0b00000000000000000000000000111111);
    constexpr const label_t WEST_FRONT = (0b00000000000000000000000011110101);
    constexpr const label_t WEST_BACK = (0b00000000000000000000000001011111);
    constexpr const label_t EAST_FRONT = (0b00000000000000000000000011111010);
    constexpr const label_t EAST_BACK = (0b00000000000000000000000010101111);
    constexpr const label_t NORTH_WEST_FRONT = (0b00000000000000000000000011111101);
    constexpr const label_t NORTH_WEST_BACK = (0b00000000000000000000000011011111);
    constexpr const label_t NORTH_EAST_FRONT = (0b00000000000000000000000011111110);
    constexpr const label_t NORTH_EAST_BACK = (0b00000000000000000000000011101111);
    constexpr const label_t SOUTH_WEST_FRONT = (0b00000000000000000000000011110111);
    constexpr const label_t SOUTH_WEST_BACK = (0b00000000000000000000000001111111);
    constexpr const label_t SOUTH_EAST_FRONT = (0b00000000000000000000000011111011);
    constexpr const label_t SOUTH_EAST_BACK = (0b00000000000000000000000010111111);

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

        __device__ __host__ [[nodiscard]] inline consteval label_t nx() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        __device__ __host__ [[nodiscard]] inline consteval label_t ny() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        __device__ __host__ [[nodiscard]] inline consteval label_t nz() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        __device__ __host__ [[nodiscard]] inline consteval label_t size() noexcept
        {
            return nx() * ny() * nz();
        }
    }

    /**
     * @brief Host-side indexing functions
     **/
    namespace host
    {
        __host__ [[nodiscard]] inline bool out_of_bounds(
            const label_t nx, const label_t ny, const label_t nz,
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return ((tx + bx * bx >= nx) || (ty + by * by >= ny) || (tz + bz * bz >= nz));
        }

        template <const label_t mom>
        __host__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nBlockx, const label_t nBlocky)
        {
            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (mom + NUMBER_MOMENTS() * (bx + nBlockx * (by + nBlocky * (bz))))));
        }

        __host__ [[nodiscard]] inline label_t idxScalarGlobal(const label_t x, const label_t y, const label_t z, const label_t nx, const label_t ny)
        {
            return x + nx * (y + ny * (z));
        }

        __host__ [[nodiscard]] inline label_t idxScalarBlock(
            const label_t tx,
            const label_t ty,
            const label_t tz,
            const label_t bx,
            const label_t by,
            const label_t bz,
            const label_t nx,
            const label_t ny)
        {
            const label_t nBlockx = nx / block::nx();
            const label_t nBlocky = ny / block::ny();

            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nBlockx * (by + nBlocky * (bz)))));
        }

        template <const label_t pop>
        __host__ [[nodiscard]] inline label_t idxPopBlock(const label_t tx, const label_t ty, const label_t tz)
        {
            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (pop)));
        }

        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopX(
            const label_t ty,
            const label_t tz,
            const label_t bx,
            const label_t by,
            const label_t bz,
            const label_t nBlockx,
            const label_t nBlocky)
        {
            return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + nBlockx * (by + nBlocky * bz))));
        }

        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopY(
            const label_t tx,
            const label_t tz,
            const label_t bx,
            const label_t by,
            const label_t bz,
            const label_t nBlockx,
            const label_t nBlocky)
        {
            return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + nBlockx * (by + nBlocky * bz))));
        }

        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopZ(
            const label_t tx,
            const label_t ty,
            const label_t bx,
            const label_t by,
            const label_t bz,
            const label_t nBlockx,
            const label_t nBlocky)
        {
            return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + nBlockx * (by + nBlocky * bz))));
        }
    }

    /**
     * @brief Device-side indexing functions
     **/
    namespace device
    {
        __device__ [[nodiscard]] inline bool out_of_bounds() noexcept
        {
            return ((threadIdx.x + blockDim.x * blockIdx.x >= d_nx) || (threadIdx.y + blockDim.y * blockIdx.y >= d_ny) || (threadIdx.z + blockDim.z * blockIdx.z >= d_nz));
        }

        __device__ [[nodiscard]] inline bool bad_node_type(const nodeType_t nodeType) noexcept
        {
            return (nodeType == 0b11111111);
        }

        template <const label_t mom>
        __device__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz)
        {
            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (mom + NUMBER_MOMENTS() * (bx + d_NUM_BLOCK_X * (by + d_NUM_BLOCK_Y * (bz))))));
        }

        template <const label_t mom>
        __device__ [[nodiscard]] inline label_t idxMom(
            const dim3 &tx,
            const dim3 &bx)
        {
            return tx.x + block::nx() * (tx.y + block::ny() * (tx.z + block::nz() * (mom + NUMBER_MOMENTS() * (bx.x + d_NUM_BLOCK_X * (bx.y + d_NUM_BLOCK_Y * (bx.z))))));
        }

        __device__ [[nodiscard]] inline label_t idxScalarGlobal(const label_t x, const label_t y, const label_t z)
        {
            return x + d_nx * (y + d_ny * (z));
        }

        __device__ [[nodiscard]] inline label_t idxScalarBlock(
            const label_t tx,
            const label_t ty,
            const label_t tz,
            const label_t bx,
            const label_t by,
            const label_t bz)
        {
            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + d_NUM_BLOCK_X * (by + d_NUM_BLOCK_Y * (bz)))));
        }

        __device__ [[nodiscard]] inline label_t idxScalarBlock(
            const dim3 &tx,
            const dim3 &bx)
        {
            return tx.x + block::nx() * (tx.y + block::ny() * (tx.z + block::nz() * (bx.x + d_NUM_BLOCK_X * (bx.y + d_NUM_BLOCK_Y * (bx.z)))));
        }

        template <const label_t pop>
        __device__ [[nodiscard]] inline label_t idxPopBlock(const label_t tx, const label_t ty, const label_t tz)
        {
            return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (pop)));
        }

        template <const label_t pop>
        __device__ [[nodiscard]] inline label_t idxPopBlock(const dim3 &tx)
        {
            return tx.x + block::nx() * (tx.y + block::ny() * (tx.z + block::nz() * (pop)));
        }

        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopX(
            const label_t ty,
            const label_t tz,
            const label_t bx,
            const label_t by,
            const label_t bz)
        {
            return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + d_NUM_BLOCK_X * (by + d_NUM_BLOCK_Y * bz))));
        }

        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopX(
            const label_t ty,
            const label_t tz,
            const dim3 &bx)
        {
            return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx.x + d_NUM_BLOCK_X * (bx.y + d_NUM_BLOCK_Y * bx.z))));
        }

        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopY(
            const label_t tx,
            const label_t tz,
            const label_t bx,
            const label_t by,
            const label_t bz)
        {
            return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + d_NUM_BLOCK_X * (by + d_NUM_BLOCK_Y * bz))));
        }

        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopY(
            const label_t tx,
            const label_t tz,
            const dim3 &bx)
        {
            return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx.x + d_NUM_BLOCK_X * (bx.y + d_NUM_BLOCK_Y * bx.z))));
        }

        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopZ(
            const label_t tx,
            const label_t ty,
            const label_t bx,
            const label_t by,
            const label_t bz)
        {
            return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + d_NUM_BLOCK_X * (by + d_NUM_BLOCK_Y * bz))));
        }

        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopZ(
            const label_t tx,
            const label_t ty,
            const dim3 &bx)
        {
            return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx.x + d_NUM_BLOCK_X * (bx.y + d_NUM_BLOCK_Y * bx.z))));
        }
    }

    /**
     * @brief Variable indices
     **/
    namespace index
    {
        __device__ __host__ [[nodiscard]] inline consteval label_t rho() { return 0; }
        __device__ __host__ [[nodiscard]] inline consteval label_t u() { return 1; }
        __device__ __host__ [[nodiscard]] inline consteval label_t v() { return 2; }
        __device__ __host__ [[nodiscard]] inline consteval label_t w() { return 3; }
        __device__ __host__ [[nodiscard]] inline consteval label_t xx() { return 4; }
        __device__ __host__ [[nodiscard]] inline consteval label_t xy() { return 5; }
        __device__ __host__ [[nodiscard]] inline consteval label_t xz() { return 6; }
        __device__ __host__ [[nodiscard]] inline consteval label_t yy() { return 7; }
        __device__ __host__ [[nodiscard]] inline consteval label_t yz() { return 8; }
        __device__ __host__ [[nodiscard]] inline consteval label_t zz() { return 9; }
    }

    __device__ __host__ [[nodiscard]] inline consteval scalar_t rho0() noexcept
    {
        return 1.0;
    }
}

#endif