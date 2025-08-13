/**
 * @file globalFunctions.cuh
 * @brief Global utility functions and structures for LBM simulations
 *
 * Contains core functionalities used throughout the LBM implementation including:
 * - Compile-time loop unrolling
 * - Memory indexing utilities
 * - Block/thread dimension constants
 * - Physical constant definitions
 * - Node type handling
 **/

#ifndef __MBLBM_GLOBALFUNCTIONS_CUH
#define __MBLBM_GLOBALFUNCTIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    /**
     * @brief Compile-time recursive loop unroller
     * @tparam Start Starting index (inclusive)
     * @tparam End Ending index (exclusive)
     * @tparam F Callable type accepting integral_constant<label_t>
     * @param f Function object to execute per iteration
     *
     * @note Equivalent to runtime loop: `for(label_t i=Start; i<End; ++i)`
     * @note Enables `if constexpr` usage in loop bodies
     * @warning Recursion depth limited by compiler constraints
     *
     * Example usage:
     * @code
     * constexpr_for<0, 5>([](auto i) {
     *     // i is integral_constant<label_t, N>
     *     if constexpr (i.value % 2 == 0) { ... }
     * });
     * @endcode
     **/
    namespace host
    {
        template <const label_t Start, const label_t End, typename F>
        __host__ inline constexpr void constexpr_for(F &&f)
        {
            if constexpr (Start < End)
            {
                f(std::integral_constant<label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    namespace device
    {
        template <const label_t Start, const label_t End, typename F>
        __device__ inline constexpr void constexpr_for(F &&f)
        {
            if constexpr (Start < End)
            {
                f(integralConstant<label_t, Start>());
                if constexpr (Start + 1 < End)
                {
                    constexpr_for<Start + 1, End>(std::forward<F>(f));
                }
            }
        }
    }

    /**
     * @brief Number of hydrodynamic moments
     **/
    [[nodiscard]] inline consteval label_t NUMBER_MOMENTS() { return 10; }

    /**
     * @brief Block dimensions descriptor
     * @details Stores lattice dimensions in 3D space
     **/
    struct blockLabel_t
    {
        const label_t nx; // < Lattice points in x-direction
        const label_t ny; // < Lattice points in y-direction
        const label_t nz; // < Lattice points in z-direction
    };

    /**
     * @brief Point indices descriptor
     * @details Stores point indices in 3D space
     **/
    struct pointLabel_t
    {
        const label_t x; // < Lattice point in x-direction
        const label_t y; // < Lattice point in y-direction
        const label_t z; // < Lattice point in z-direction
    };

    /**
     * @brief 1D range descriptor [begin, end)
     **/
    struct blockPartitionRange_t
    {
        const label_t begin; // < Inclusive start index
        const label_t end;   // < Exclusive end index
    };

    /**
     * @brief 3D block range descriptor
     * @details Defines a rectangular region in lattice space
     **/
    struct blockRange_t
    {
        const blockPartitionRange_t xRange; // < X-dimension range
        const blockPartitionRange_t yRange; // < Y-dimension range
        const blockPartitionRange_t zRange; // < Z-dimension range
    };

    /**
     * @brief CUDA block dimension configuration
     * @details Compile-time constants defining thread block dimensions
     **/
    namespace block
    {
        /**
         * @brief Threads per block in x-dimension (compile-time constant)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t nx() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        /**
         * @brief Threads per block in y-dimension (compile-time constant)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t ny() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 8;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        /**
         * @brief Threads per block in z-dimension (compile-time constant)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t nz() noexcept
        {
#ifdef SCALAR_PRECISION_32
            return 4;
#elif SCALAR_PRECISION_64
            return 4;
#endif
        }

        /**
         * @brief Total threads per block (nx * ny * nz)
         **/
        __device__ __host__ [[nodiscard]] inline consteval label_t size() noexcept
        {
            return nx() * ny() * nz();
        }
    }

    /**
     * @brief Host-side indexing operations
     **/
    namespace host
    {
        /**
         * @brief Compute moment memory index
         * @tparam mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nBlockx Number of blocks in x-direction
         * @param nBlocky Number of blocks in y-direction
         * @return Linearized index in moment array
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        template <const label_t mom>
        __host__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nBlockx, const label_t nBlocky) noexcept
        {
            return mom + NUMBER_MOMENTS() * (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nBlockx * (by + nBlocky * bz)))));
        }

        /**
         * @overload Compute moment memory index
         * @tparam mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nBlockx Number of blocks in x-direction
         * @param nBlocky Number of blocks in y-direction
         * @return Linearized index in moment array
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        template <const label_t mom, class M>
        __host__ [[nodiscard]] inline label_t idxMom(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const M &mesh) noexcept
        {
            return idxMom<mom>(tx, ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks());
        }

        /**
         * @brief Memory index (host version)
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param nBlockx,nBlocky Number of blocks in the x and y directions
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        __host__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nBlockx, const label_t nBlocky) noexcept
        {
            return (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nBlockx * (by + nBlocky * bz)))));
        }

        /**
         * @brief Memory index (host version)
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @param mesh The mesh
         * @return Linearized index using mesh constants
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        template <class M>
        __host__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const M &mesh) noexcept
        {
            return idx(tx, ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks());
        }

        /**
         * @brief Global scalar field index (collapsed 3D)
         * @param x,y,z Global coordinates
         * @param nx,ny Global dimensions
         * @return Linearized index: x + nx*(y + ny*z)
         **/
        __host__ [[nodiscard]] inline label_t idxScalarGlobal(
            const label_t x, const label_t y, const label_t z,
            const label_t nx, const label_t ny) noexcept
        {
            return x + (nx * (y + (ny * z)));
        }

        __host__ [[nodiscard]] inline label_t idxScalarGlobal(
            const pointLabel_t point,
            const label_t nx, const label_t ny) noexcept
        {
            return point.x + (nx * (point.y + (ny * point.z)));
        }

        /**
         * @brief Index for X-aligned population arrays
         * @tparam pop Population index
         * @tparam QF Number of populations
         * @param ty,tz Thread-local y/z coordinates
         * @param bx,by,bz Block indices
         * @param nBlockx Number of blocks in x-direction
         * @param nBlocky Number of blocks in y-direction
         * @return Linearized index: ty + block::ny()*(tz + block::nz()*(pop + QF*(bx + nBlockx*(by + nBlocky*bz)))
         * @note Optimized for coalesced memory access in X-direction
         **/
        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopX(
            const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nBlockx, const label_t nBlocky) noexcept
        {
            return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + nBlockx * (by + nBlocky * bz))));
        }

        /**
         * @brief Index for Y-aligned population arrays
         * @copydetails idxPopX
         * @param tx,tz Thread-local x/z coordinates
         * @return Linearized index: tx + block::nx()*(tz + block::nz()*(pop + QF*(bx + nBlockx*(by + nBlocky*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopY(
            const label_t tx, const label_t tz,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nBlockx, const label_t nBlocky) noexcept
        {
            return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + nBlockx * (by + nBlocky * bz))));
        }

        /**
         * @brief Index for Z-aligned population arrays
         * @copydetails idxPopX
         * @param tx,ty Thread-local x/y coordinates
         * @return Linearized index: tx + block::nx()*(ty + block::ny()*(pop + QF*(bx + nBlockx*(by + nBlocky*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __host__ [[nodiscard]] inline label_t idxPopZ(
            const label_t tx, const label_t ty,
            const label_t bx, const label_t by, const label_t bz,
            const label_t nBlockx, const label_t nBlocky) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + nBlockx * (by + nBlocky * bz))));
        }

        [[nodiscard]] const std::array<cudaStream_t, 1> createCudaStream() noexcept
        {
            std::array<cudaStream_t, 1> streamsLBM;

            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
            checkCudaErrors(cudaDeviceSynchronize());

            return streamsLBM;
        }
    }

    /**
     * @brief Device-side indexing operations
     **/
    namespace device
    {
        /**
         * @brief Check if current thread exceeds global bounds
         * @note Uses device constants device::nx, device::ny, device::nz
         * @return True if thread is outside domain boundaries
         **/
        __device__ [[nodiscard]] inline bool out_of_bounds() noexcept
        {
            return ((threadIdx.x + blockDim.x * blockIdx.x >= device::nx) || (threadIdx.y + blockDim.y * blockIdx.y >= device::ny) || (threadIdx.z + blockDim.z * blockIdx.z >= device::nz));
        }

        /**
         * @brief Moment memory index (device version)
         * @tparam mom Moment index [0, NUMBER_MOMENTS())
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @return Linearized index using device constants device::NUM_BLOCK_X/Y
         *
         * Layout: [bx][by][bz][tz][ty][tx][mom] (mom fastest varying)
         **/
        template <const label_t mom>
        __device__ [[nodiscard]] inline label_t idxMom(const label_t tx, const label_t ty, const label_t tz, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return mom + NUMBER_MOMENTS() * (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz)))));
        }

        /**
         * @overload
         * @param tx Thread coordinates (dim3)
         * @param bx Block indices (dim3)
         **/
        template <const label_t mom>
        __device__ [[nodiscard]] inline label_t idxMom(const dim3 &tx, const dim3 &bx) noexcept
        {
            return idxMom<mom>(tx.x, tx.y, tx.z, bx.x, bx.y, bx.z);
        }

        /**
         * @brief Memory index (device version)
         * @param tx,ty,tz Thread-local coordinates
         * @param bx,by,bz Block indices
         * @return Linearized index using device constants device::NUM_BLOCK_X/Y
         *
         * Layout: [bx][by][bz][tz][ty][tx] (tx fastest varying)
         **/
        __device__ [[nodiscard]] inline label_t idx(
            const label_t tx, const label_t ty, const label_t tz,
            const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return (tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz)))));
        }

        /**
         * @overload
         * @param tx Thread coordinates (dim3)
         * @param bx Block indices (dim3)
         **/
        __device__ [[nodiscard]] inline label_t idx(const dim3 tx, const dim3 bx) noexcept
        {
            return idx(tx.x, tx.y, tx.z, bx.x, bx.y, bx.z);
        }

        /**
         * @overload
         **/
        __device__ [[nodiscard]] inline label_t idx() noexcept
        {
            return idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
        }

        /**
         * @brief Population index for X-aligned arrays (device version)
         * @tparam pop Population index
         * @tparam QF Number of populations
         * @param ty,tz Thread-local y/z coordinates
         * @param bx,by,bz Block indices
         * @return Linearized index: ty + block::ny()*(tz + block::nz()*(pop + QF*(bx + device::NUM_BLOCK_X*(by + device::NUM_BLOCK_Y*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopX(const label_t ty, const label_t tz, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return ty + block::ny() * (tz + block::nz() * (pop + QF * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz))));
        }

        /**
         * @overload
         * @param ty,tz Thread-local y/z coordinates
         * @param bx Block indices (dim3)
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopX(const label_t ty, const label_t tz, const dim3 &bx) noexcept
        {
            return idxPopX<pop, QF>(ty, tz, bx.x, bx.y, bx.z);
        }

        /**
         * @brief Population index for Y-aligned arrays (device version)
         * @copydetails idxPopX
         * @param tx,tz Thread-local x/z coordinates
         * @return Linearized index: tx + block::nx()*(tz + block::nz()*(pop + QF*(bx + device::NUM_BLOCK_X*(by + device::NUM_BLOCK_Y*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopY(const label_t tx, const label_t tz, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return tx + block::nx() * (tz + block::nz() * (pop + QF * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz))));
        }

        /**
         * @overload
         * @param tx,tz Thread-local x/z coordinates
         * @param bx Block indices (dim3)
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopY(const label_t tx, const label_t tz, const dim3 &bx) noexcept
        {
            return idxPopY<pop, QF>(tx, tz, bx.x, bx.y, bx.z);
        }

        /**
         * @brief Population index for Z-aligned arrays (device version)
         * @copydetails idxPopX
         * @param tx,ty Thread-local x/y coordinates
         * @return Linearized index: tx + block::nx()*(ty + block::ny()*(pop + QF*(bx + device::NUM_BLOCK_X*(by + device::NUM_BLOCK_Y*bz)))
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopZ(const label_t tx, const label_t ty, const label_t bx, const label_t by, const label_t bz) noexcept
        {
            return tx + block::nx() * (ty + block::ny() * (pop + QF * (bx + device::NUM_BLOCK_X * (by + device::NUM_BLOCK_Y * bz))));
        }

        /**
         * @overload
         * @param tx,ty Thread-local x/y coordinates
         * @param bx Block indices (dim3)
         **/
        template <const label_t pop, const label_t QF>
        __device__ [[nodiscard]] inline label_t idxPopZ(const label_t tx, const label_t ty, const dim3 &bx) noexcept
        {
            return idxPopZ<pop, QF>(tx, ty, bx.x, bx.y, bx.z);
        }
    }

    /**
     * @brief Moment indices for hydrodynamic variables
     **/
    namespace index
    {
        __device__ __host__ [[nodiscard]] inline consteval label_t rho() { return 0; } // < Density
        __device__ __host__ [[nodiscard]] inline consteval label_t u() { return 1; }   // < X-velocity
        __device__ __host__ [[nodiscard]] inline consteval label_t v() { return 2; }   // < Y-velocity
        __device__ __host__ [[nodiscard]] inline consteval label_t w() { return 3; }   // < Z-velocity
        __device__ __host__ [[nodiscard]] inline consteval label_t xx() { return 4; }  // < XX-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t xy() { return 5; }  // < XY-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t xz() { return 6; }  // < XZ-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t yy() { return 7; }  // < YY-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t yz() { return 8; }  // < YZ-stress component
        __device__ __host__ [[nodiscard]] inline consteval label_t zz() { return 9; }  // < ZZ-stress component
    }

    /**
     * @brief Reference density 1.0
     **/
    __device__ __host__ [[nodiscard]] inline consteval scalar_t rho0() noexcept
    {
        return 1.0;
    }
}

#endif