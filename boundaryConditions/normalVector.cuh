/**
Filename: normalVector.cuh
Contents: A class used to compute the normal vector of a boundary lattice
**/

#ifndef __MBLBM_NORMALVECTOR_CUH
#define __MBLBM_NORMALVECTOR_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    struct normalVector
    {
    public:
        /**
         * @brief Constructs a normalVector from the current threadIdx
         * @return A normalVector for the current threadIdx
         **/
        __device__ [[nodiscard]] inline normalVector() noexcept
            : bitmask_(computeBitmask()){};

        /**
         * @brief Constructs a normalVector from the current threadIdx
         * @return A normalVector for the current threadIdx
         **/
        __device__ [[nodiscard]] inline normalVector(const label_t x, const label_t y, const label_t z) noexcept
            : bitmask_(computeBitmask(x, y, z)){};

        /**
         * @brief Basic boundary flags
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t WEST() noexcept { return 0x01; }  // 1 << 0
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t EAST() noexcept { return 0x02; }  // 1 << 1
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH() noexcept { return 0x04; } // 1 << 2
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH() noexcept { return 0x08; } // 1 << 3
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t BACK() noexcept { return 0x10; }  // 1 << 4
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t FRONT() noexcept { return 0x20; } // 1 << 5

        /**
         * @brief Corner boundary types (8)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST_BACK() noexcept { return SOUTH() | WEST() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST_FRONT() noexcept { return SOUTH() | WEST() | FRONT(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST_BACK() noexcept { return SOUTH() | EAST() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST_FRONT() noexcept { return SOUTH() | EAST() | FRONT(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST_BACK() noexcept { return NORTH() | WEST() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST_FRONT() noexcept { return NORTH() | WEST() | FRONT(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST_BACK() noexcept { return NORTH() | EAST() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST_FRONT() noexcept { return NORTH() | EAST() | FRONT(); }

        /**
         * @brief Edge boundary types (12)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST() noexcept { return SOUTH() | WEST(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST() noexcept { return SOUTH() | EAST(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST() noexcept { return NORTH() | WEST(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST() noexcept { return NORTH() | EAST(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t WEST_BACK() noexcept { return WEST() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t WEST_FRONT() noexcept { return WEST() | FRONT(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t EAST_BACK() noexcept { return EAST() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t EAST_FRONT() noexcept { return EAST() | FRONT(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_BACK() noexcept { return SOUTH() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_FRONT() noexcept { return SOUTH() | FRONT(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_BACK() noexcept { return NORTH() | BACK(); }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_FRONT() noexcept { return NORTH() | FRONT(); }

        /**
         * @brief Special types
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t INTERIOR() noexcept { return 0x00; } // No boundaries

        /**
         * @brief Determine whether or not the current threadIdx lies on a certain boundary
         * @return Boolean true if it lies on the boundary, boolean false otherwise
         **/
        __device__ [[nodiscard]] inline bool isWest() const noexcept { return bitmask_ & WEST(); }
        __device__ [[nodiscard]] inline bool isEast() const noexcept { return bitmask_ & EAST(); }
        __device__ [[nodiscard]] inline bool isSouth() const noexcept { return bitmask_ & SOUTH(); }
        __device__ [[nodiscard]] inline bool isNorth() const noexcept { return bitmask_ & NORTH(); }
        __device__ [[nodiscard]] inline bool isBack() const noexcept { return bitmask_ & BACK(); }
        __device__ [[nodiscard]] inline bool isFront() const noexcept { return bitmask_ & FRONT(); }
        __device__ [[nodiscard]] inline bool isBoundary() const noexcept { return bitmask_ & 0x40; }
        __device__ [[nodiscard]] inline bool isInterior() const noexcept { return !isBoundary(); }

        /**
         * @brief Return the bit mask, i.e. the lattice node type
         * @return The lattice node type
         **/
        __device__ [[nodiscard]] inline uint8_t nodeType() const noexcept { return bitmask_ & 0x3F; }

    private:
        /**
         * @brief The underlying bit mask
         **/
        const uint8_t bitmask_;

        /**
         * @brief Compute the bit mask from the current threadIdx
         * @return The bit mask
         **/
        __device__ [[nodiscard]] inline static uint8_t computeBitmask() noexcept
        {
            return computeBitmask(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y, threadIdx.z + blockDim.z * blockIdx.z);
        }
        __device__ [[nodiscard]] inline static uint8_t computeBitmask(const label_t x, const label_t y, const label_t z) noexcept
        {
            return static_cast<uint8_t>(
                (x == 0) << 0 |                      // West (bit0)
                (x == device::nx - 1) << 1 |         // East (bit1)
                (y == 0) << 2 |                      // South (bit2)
                (y == device::ny - 1) << 3 |         // North (bit3)
                (z == 0) << 4 |                      // Back (bit4)
                (z == device::nz - 1) << 5 |         // Front (bit5)
                (!!(x == 0 || x == device::nx - 1 || //
                    y == 0 || y == device::ny - 1 || //
                    z == 0 || z == device::nz - 1))  //
                    << 6);                           // Any boundary (bit6)
        }
    };
}

#endif