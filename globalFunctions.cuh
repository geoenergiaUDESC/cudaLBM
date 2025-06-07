/**
Filename: globalFunctions.cuh
Contents: Functions used throughout the source code
**/

#ifndef __MBLBM_GLOBALFUNCTIONS_CUH
#define __MBLBM_GLOBALFUNCTIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "globalDefines.cuh"

namespace mbLBM
{
    __device__ [[nodiscard]] inline bool out_of_bounds(const label_t nx, const label_t ny, const label_t nz) noexcept
    {
        return ((threadIdx.x + blockDim.x * blockIdx.x >= nx) || (threadIdx.y + blockDim.y * blockIdx.y >= ny) || (threadIdx.z + blockDim.z * blockIdx.z >= nz));
    }

    __host__ [[nodiscard]] inline bool out_of_bounds(
        const label_t nx, const label_t ny, const label_t nz,
        const label_t tx, const label_t ty, const label_t tz,
        const label_t bx, const label_t by, const label_t bz) noexcept
    {
        return ((tx + bx * bx >= nx) || (ty + by * by >= ny) || (tz + bz * bz >= nz));
    }

    __device__ [[nodiscard]] inline bool bad_node_type(const nodeType_t nodeType) noexcept
    {
        return (nodeType == 0b11111111);
    }

    __host__ [[nodiscard]] inline constexpr scalar_t omega(const scalar_t Re, const scalar_t u_inf, const label_t nx) noexcept
    {
        return static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * (u_inf * static_cast<scalar_t>(nx) / Re));
    }

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
            return BLOCK_NX - 1;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t South() noexcept
        {
            return 0;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t North() noexcept
        {
            return BLOCK_NY - 1;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t Back() noexcept
        {
            return 0;
        }

        __host__ __device__ [[nodiscard]] inline consteval label_t Front() noexcept
        {
            return BLOCK_NZ - 1;
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

    template <const label_t mom>
    __device__ __host__ [[nodiscard]] inline label_t idxMom(
        const label_t tx, const label_t ty, const label_t tz,
        const label_t bx, const label_t by, const label_t bz)
    {
        return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (mom + NUMBER_MOMENTS * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz))))));
    }

    __host__ __device__ [[nodiscard]] inline scalar_t gpu_f_eq(const scalar_t rhow, const scalar_t uc3, const scalar_t p1_muu)
    {
        return (rhow * (p1_muu + uc3 * (static_cast<scalar_t>(1.0) + uc3 * static_cast<scalar_t>(0.5))));
    }

    __host__ __device__ [[nodiscard]] inline label_t idxScalarGlobal(const label_t x, const label_t y, const label_t z)
    {
        return x + NX * (y + NY * (z));
    }

    __host__ __device__ [[nodiscard]] inline label_t idxScalarBlock(
        const label_t tx,
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz)))));
    }

    template <const label_t pop>
    __host__ __device__ [[nodiscard]] inline label_t idxPopBlock(const label_t tx, const label_t ty, const label_t tz)
    {
        return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (pop)));
    }

    template <const label_t pop>
    __device__ [[nodiscard]] inline label_t idxPopX(
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return ty + BLOCK_NY * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    template <const label_t pop>
    __device__ [[nodiscard]] inline label_t idxPopY(
        const label_t tx,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return tx + BLOCK_NX * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    template <const label_t pop>
    __device__ [[nodiscard]] inline label_t idxPopZ(
        const label_t tx,
        const label_t ty,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return tx + BLOCK_NX * (ty + BLOCK_NY * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    __device__ [[nodiscard]] inline bool INTERFACE_BC_WEST(const label_t x)
    {
        return (threadIdx.x == 0 && x != 0);
    }
    __device__ [[nodiscard]] inline bool INTERFACE_BC_EAST(const label_t x)
    {
        return (threadIdx.x == (BLOCK_NX - 1) && x != (NX - 1));
    }

    __device__ [[nodiscard]] inline bool INTERFACE_BC_SOUTH(const label_t y)
    {
        return (threadIdx.y == 0 && y != 0);
    }
    __device__ [[nodiscard]] inline bool INTERFACE_BC_NORTH(const label_t y)
    {
        return (threadIdx.y == (BLOCK_NY - 1) && y != (NY - 1));
    }

    __device__ [[nodiscard]] inline bool INTERFACE_BC_BACK(const label_t z)
    {
        return (threadIdx.z == 0 && z != 0);
    }
    __device__ [[nodiscard]] inline bool INTERFACE_BC_FRONT(const label_t z)
    {
        return (threadIdx.z == (BLOCK_NZ - 1) && z != (NZ - 1));
    }
}

#endif