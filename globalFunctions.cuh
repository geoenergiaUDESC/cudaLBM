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
    constexpr const label_t Q = 19;
    constexpr const label_t QF = 5;

    constexpr const label_t NUMBER_MOMENTS = 10;

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

    constexpr const scalar_t RE = 500;

    constexpr const label_t NX = 128;
    constexpr const label_t NY = 128;
    constexpr const label_t NZ = 128;

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
        // __host__ __device__ [[nodiscard]] inline consteval dim3 threadBlock() noexcept
        // {
        //     return {nx(), ny(), nz()};
        // }
        // template <typename M>
        // __host__ __device__ [[nodiscard]] inline constexpr dim3 gridBlock(const M &mesh) noexcept
        // {
        //     return {static_cast<uint32_t>(mesh.nx() / nx()), static_cast<uint32_t>(mesh.ny() / ny()), static_cast<uint32_t>(mesh.nz() / nz())};
        // }

        // /**
        //  * @brief Calculate the number of blocks in a given direction from a number of lattice points
        //  * @return Number of blocks in a given direction
        //  * @param n Number of lattice points in a given direction
        //  **/

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

        // /**
        //  * @brief Provide the block-relative indices of block boundaries
        //  * @return The x, y or z index in a given direction corresponding to a block boundary
        //  **/

        // __host__ __device__ [[nodiscard]] inline consteval label_t West() noexcept
        // {
        //     return 0;
        // }

        // __host__ __device__ [[nodiscard]] inline consteval label_t East() noexcept
        // {
        //     return block::nx() - 1;
        // }

        // __host__ __device__ [[nodiscard]] inline consteval label_t South() noexcept
        // {
        //     return 0;
        // }

        // __host__ __device__ [[nodiscard]] inline consteval label_t North() noexcept
        // {
        //     return block::ny() - 1;
        // }

        // __host__ __device__ [[nodiscard]] inline consteval label_t Back() noexcept
        // {
        //     return 0;
        // }

        // __host__ __device__ [[nodiscard]] inline consteval label_t Front() noexcept
        // {
        //     return block::nz() - 1;
        // }

        // /**
        //  * @brief Check whether a particular lattice lies along a block boundary
        //  * @return TRUE if n lies along the boundary, FALSE otherwise
        //  * @param n The lattice index to be checked
        //  **/

        // __host__ __device__ [[nodiscard]] inline constexpr bool South(const label_t n) noexcept
        // {
        //     return (n == South());
        // }

        // __host__ __device__ [[nodiscard]] inline constexpr bool West(const label_t n) noexcept
        // {
        //     return (n == West());
        // }

        // __host__ __device__ [[nodiscard]] inline constexpr bool East(const label_t n) noexcept
        // {
        //     return (n == East());
        // }

        // __host__ __device__ [[nodiscard]] inline constexpr bool North(const label_t n) noexcept
        // {
        //     return (n == North());
        // }

        // __host__ __device__ [[nodiscard]] inline constexpr bool Back(const label_t n) noexcept
        // {
        //     return (n == Back());
        // }

        // __host__ __device__ [[nodiscard]] inline constexpr bool Front(const label_t n) noexcept
        // {
        //     return (n == Front());
        // }
    }

    constexpr const label_t NUM_BLOCK_X = NX / block::nx();
    constexpr const label_t NUM_BLOCK_Y = NY / block::ny();
    constexpr const label_t NUM_BLOCK_Z = NZ / block::nz();

    // /**
    //  * @brief Returns the flattened block-relative index of a point expressed in 3 dimensions
    //  * @return The index of x, y, z expressed in flattened indices
    //  * @param x The x-component of the block
    //  * @param y The y-component of the block
    //  * @param z The z-component of the block
    //  * @param blockDimensions The physical dimensions of the block
    //  **/

    // __host__ __device__ [[nodiscard]] inline constexpr label_t blockLabel(const label_t x, const label_t y, const label_t z, const blockLabel_t &blockDimensions) noexcept
    // {
    //     return (z * (blockDimensions.nx * blockDimensions.ny)) + (y * blockDimensions.nx) + (x);
    // }

    // /**
    //  * @brief Returns the flattened block-relative index of a point expressed in 3 dimensions
    //  * @return The index of x, y, z expressed in flattened indices
    //  * @param x The x-component of the block
    //  * @param y The y-component of the block
    //  * @param z The z-component of the block
    //  * @param mesh The mesh
    //  **/
    // template <class M>
    // __host__ __device__ [[nodiscard]] inline constexpr label_t blockLabel(const label_t x, const label_t y, const label_t z, const M &mesh) noexcept
    // {
    //     return (z * (mesh.nx() * mesh.ny())) + (y * mesh.nx()) + (x);
    // }

    template <const label_t mom>
    __device__ __host__ [[nodiscard]] inline label_t idxMom(
        const label_t tx, const label_t ty, const label_t tz,
        const label_t bx, const label_t by, const label_t bz)
    {
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (mom + NUMBER_MOMENTS * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz))))));
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
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz)))));
    }

    template <const label_t pop>
    __host__ __device__ [[nodiscard]] inline label_t idxPopBlock(const label_t tx, const label_t ty, const label_t tz)
    {
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (pop)));
    }

    template <const label_t pop>
    __device__ __host__ [[nodiscard]] inline label_t idxPopX(
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    template <const label_t pop>
    __device__ __host__ [[nodiscard]] inline label_t idxPopY(
        const label_t tx,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    template <const label_t pop>
    __device__ __host__ [[nodiscard]] inline label_t idxPopZ(
        const label_t tx,
        const label_t ty,
        const label_t bx,
        const label_t by,
        const label_t bz)
    {
        return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
    }

    constexpr const scalar_t U_MAX = static_cast<scalar_t>(0.05);

    constexpr const scalar_t VISC = U_MAX * static_cast<scalar_t>(NX) / RE;
    constexpr const scalar_t TAU = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * VISC; // relaxation time
    constexpr const scalar_t MACH_NUMBER = U_MAX / static_cast<scalar_t>(0.57735026918962);

    constexpr const scalar_t OMEGA = static_cast<scalar_t>(1.0) / TAU;                                   // (tau)^-1
    constexpr const scalar_t OMEGAd2 = OMEGA / static_cast<scalar_t>(2.0);                               // OMEGA/2
    constexpr const scalar_t OMEGAd9 = OMEGA / static_cast<scalar_t>(9.0);                               // OMEGA/9
    constexpr const scalar_t T_OMEGA = static_cast<scalar_t>(1.0) - OMEGA;                               // 1-OMEGA
    constexpr const scalar_t TT_OMEGA = static_cast<scalar_t>(1.0) - static_cast<scalar_t>(0.5) * OMEGA; // 1.0 - OMEGA/2
    constexpr const scalar_t OMEGA_P1 = static_cast<scalar_t>(1.0) + OMEGA;                              // 1+ OMEGA
    constexpr const scalar_t TT_OMEGA_T3 = TT_OMEGA * static_cast<scalar_t>(3.0);                        // 3*(1-0.5*OMEGA)

    constexpr const scalar_t RHO_0 = 1.0;

    constexpr const label_t M_RHO_INDEX = 0;
    constexpr const label_t M_UX_INDEX = 1;
    constexpr const label_t M_UY_INDEX = 2;
    constexpr const label_t M_UZ_INDEX = 3;
    constexpr const label_t M_MXX_INDEX = 4;
    constexpr const label_t M_MXY_INDEX = 5;
    constexpr const label_t M_MXZ_INDEX = 6;
    constexpr const label_t M_MYY_INDEX = 7;
    constexpr const label_t M_MYZ_INDEX = 8;
    constexpr const label_t M_MZZ_INDEX = 9;

    constexpr const label_t NUMBER_GHOST_FACE_XY = block::nx() * block::ny() * NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;
    constexpr const label_t NUMBER_GHOST_FACE_XZ = block::nx() * block::nz() * NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;
    constexpr const label_t NUMBER_GHOST_FACE_YZ = block::ny() * block::nz() * NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;

    constexpr const bool console_flush = false;
    constexpr const label_t GPU_INDEX = 1;

    constexpr const dim3 threadBlock(block::nx(), block::ny(), block::nz());
    constexpr const dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    constexpr const label_t INI_STEP = 0;
    constexpr const label_t N_STEPS = 1001;
}

#endif