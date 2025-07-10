/**
Filename: boundaryConditions.cuh
Contents: A class applying boundary conditions to the lid driven cavity case
**/

#ifndef __MBLBM_BOUNDARYCONDITIONS_CUH
#define __MBLBM_BOUNDARYCONDITIONS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

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
            : bitmask_(computeBitmask(x, y, z)) {}

        /**
         * @brief Basic boundary flags
         **/
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t WEST() noexcept { return 0x01; }  // 1 << 0
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t EAST() noexcept { return 0x02; }  // 1 << 1
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH() noexcept { return 0x04; } // 1 << 2
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH() noexcept { return 0x08; } // 1 << 3
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t BACK() noexcept { return 0x10; }  // 1 << 4
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t FRONT() noexcept { return 0x20; } // 1 << 5

        /**
         * @brief Corner boundary types (8)
         **/
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST_BACK() noexcept { return SOUTH() | WEST() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST_FRONT() noexcept { return SOUTH() | WEST() | FRONT(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST_BACK() noexcept { return SOUTH() | EAST() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST_FRONT() noexcept { return SOUTH() | EAST() | FRONT(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST_BACK() noexcept { return NORTH() | WEST() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST_FRONT() noexcept { return NORTH() | WEST() | FRONT(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST_BACK() noexcept { return NORTH() | EAST() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST_FRONT() noexcept { return NORTH() | EAST() | FRONT(); }

        /**
         * @brief Edge boundary types (12)
         **/
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST() noexcept { return SOUTH() | WEST(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST() noexcept { return SOUTH() | EAST(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST() noexcept { return NORTH() | WEST(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST() noexcept { return NORTH() | EAST(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t WEST_BACK() noexcept { return WEST() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t WEST_FRONT() noexcept { return WEST() | FRONT(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t EAST_BACK() noexcept { return EAST() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t EAST_FRONT() noexcept { return EAST() | FRONT(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_BACK() noexcept { return SOUTH() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t SOUTH_FRONT() noexcept { return SOUTH() | FRONT(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_BACK() noexcept { return NORTH() | BACK(); }
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t NORTH_FRONT() noexcept { return NORTH() | FRONT(); }

        /**
         * @brief Special types
         **/
        __host__ __device__ [[nodiscard]] static inline consteval uint8_t INTERIOR() noexcept { return 0x00; } // No boundaries

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
                (x == 0) << 0 |                // West (bit0)
                (x == d_nx - 1) << 1 |         // East (bit1)
                (y == 0) << 2 |                // South (bit2)
                (y == d_ny - 1) << 3 |         // North (bit3)
                (z == 0) << 4 |                // Back (bit4)
                (z == d_nz - 1) << 5 |         // Front (bit5)
                (!!(x == 0 || x == d_nx - 1 || //
                    y == 0 || y == d_ny - 1 || //
                    z == 0 || z == d_nz - 1))  //
                    << 6);                     // Any boundary (bit6)
        }
    };

    class boundaryConditions
    {
    public:
        [[nodiscard]] inline consteval boundaryConditions() {};

        /**
         * @brief Determine whether or not a component of the velocity set is incoming or outgoing based upon the boundary normal
         * @return 1 as type T if it is an incoming velocity component, 0 otherwise
         * @param q The index of the velocity set
         * @param b_n The boundary normal vector at the current lattice node
         **/
        template <class VSet, typename T, const label_t q_>
        __device__ [[nodiscard]] static inline constexpr T incomingSwitch(const lattice_constant<q_> q, const normalVector &b_n) noexcept
        {
            // b_n.x > 0  => EAST boundary
            // b_n.x < 0  => WEST boundary
            const bool cond_x = (b_n.isEast() & VSet::nxNeg(q)) | (b_n.isWest() & VSet::nxPos(q));

            // b_n.y > 0  => NORTH boundary
            // b_n.y < 0  => SOUTH boundary
            const bool cond_y = (b_n.isNorth() & VSet::nyNeg(q)) | (b_n.isSouth() & VSet::nyPos(q));

            // b_n.z > 0  => FRONT boundary
            // b_n.z < 0  => BACK boundary
            const bool cond_z = (b_n.isFront() & VSet::nzNeg(q)) | (b_n.isBack() & VSet::nzPos(q));

            return static_cast<T>(!(cond_x | cond_y | cond_z));
        }

        /**
         * @brief Calculate the moment variables at the boundary
         * @param pop The population density at the current lattice node
         * @param moments The moment variables at the current lattice node
         * @param b_n The boundary normal vector at the current lattice node
         **/
        template <class VSet>
        __device__ static inline constexpr void calculateMoments(
            const scalar_t pop[VSet::Q()],
            scalar_t (&ptrRestrict moments)[10],
            const normalVector &b_n) noexcept
        {
            const scalar_t rho_I =
                ((incomingSwitch<VSet, scalar_t>(lattice_constant<0>(), b_n) * pop[0]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<1>(), b_n) * pop[1]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<2>(), b_n) * pop[2]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<3>(), b_n) * pop[3]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<4>(), b_n) * pop[4]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<5>(), b_n) * pop[5]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<6>(), b_n) * pop[6]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<7>(), b_n) * pop[7]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<8>(), b_n) * pop[8]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<9>(), b_n) * pop[9]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<10>(), b_n) * pop[10]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<11>(), b_n) * pop[11]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<12>(), b_n) * pop[12]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<13>(), b_n) * pop[13]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<14>(), b_n) * pop[14]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<15>(), b_n) * pop[15]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<16>(), b_n) * pop[16]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<17>(), b_n) * pop[17]) +
                 (incomingSwitch<VSet, scalar_t>(lattice_constant<18>(), b_n) * pop[18]));
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            switch (b_n.nodeType())
            {
            // Static boundaries
            case normalVector::SOUTH_WEST_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_WEST_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_EAST_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_EAST_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(12) * rho_I) / static_cast<scalar_t>(7);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_WEST():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * pop[8];
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                // moments[5] = (static_cast<scalar_t>(25) * m_xy_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1)));
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_EAST():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (-pop[13]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xy_I - d_omega * m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                // moments[5] = (static_cast<scalar_t>(25) * m_xy_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xy_I - d_omega * m_xy_I + static_cast<scalar_t>(1)));
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::WEST_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[10]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::WEST_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (-pop[16]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::EAST_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (-pop[15]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_xz_I - d_omega * m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::EAST_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[9]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(25) * m_xz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_xz_I - m_xz_I + static_cast<scalar_t>(1)));
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (pop[12]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (static_cast<scalar_t>(25) * m_yz_I - static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (d_omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (-pop[18]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24));
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (static_cast<scalar_t>(25) * m_yz_I + static_cast<scalar_t>(1)) / (static_cast<scalar_t>(9) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::WEST():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[14]);
                const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[16]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::EAST():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[13]);
                const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[15]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::SOUTH():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[13]);
                const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[18]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[15]);
                const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[16]);
                const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[18]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = static_cast<scalar_t>(0);
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = (static_cast<scalar_t>(5) * m_xz_I) / static_cast<scalar_t>(3);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }

            // Lid boundaries
            case normalVector::NORTH():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[14]);
                const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(6) * rho_I) / static_cast<scalar_t>(5);
                moments[0] = rho;
                moments[1] = d_u_inf;
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                moments[4] = (static_cast<scalar_t>(6) * d_u_inf * d_u_inf * rho_I) / static_cast<scalar_t>(5);
                moments[5] = (static_cast<scalar_t>(5) * m_xy_I) / static_cast<scalar_t>(3) - d_u_inf / static_cast<scalar_t>(3);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = (static_cast<scalar_t>(5) * m_yz_I) / static_cast<scalar_t>(3);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_WEST_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_WEST_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_EAST_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_EAST_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t rho = (static_cast<scalar_t>(24) * rho_I) / (static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * d_u_inf - static_cast<scalar_t>(9) * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0); // At the corner, all moments vanish
                moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                moments[8] = static_cast<scalar_t>(0);
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_BACK():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (-pop[17]);
                const scalar_t rho = (static_cast<scalar_t>(72) * rho_I * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1))) / (static_cast<scalar_t>(2) * d_omega + static_cast<scalar_t>(48) - static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0);                           // At the intersection of front and back, only z derivative exists
                moments[5] = -VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of front and back, only z derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_FRONT():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_yz_I = inv_rho_I * (pop[11]);
                const scalar_t rho = (static_cast<scalar_t>(72) * rho_I * (d_omega * m_yz_I - m_yz_I + static_cast<scalar_t>(1))) / (static_cast<scalar_t>(2) * d_omega + static_cast<scalar_t>(48) - static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[4] = static_cast<scalar_t>(0);                          // At the intersection of front and back, only z derivative exists
                moments[5] = VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of front and back, only z derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_EAST():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * d_u_inf - static_cast<scalar_t>(18) * d_u_inf * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[5] = -static_cast<scalar_t>(2) * VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of East and West, only x derivative exists
                moments[4] = static_cast<scalar_t>(0);                                                      // At the intersection of East and West, only x derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            case normalVector::NORTH_WEST():
            {
                // const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                // const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;
                const scalar_t m_xy_I = inv_rho_I * (pop[7]);
                const scalar_t rho = (static_cast<scalar_t>(36) * rho_I * (d_omega * m_xy_I - m_xy_I + static_cast<scalar_t>(1))) / (d_omega + static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * d_u_inf - static_cast<scalar_t>(18) * d_u_inf * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf + static_cast<scalar_t>(3) * d_omega * d_u_inf * d_u_inf);
                moments[0] = rho;
                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = static_cast<scalar_t>(0);
                // moments[4] = d_u_inf * d_u_inf * rho;
                moments[5] = static_cast<scalar_t>(2) * VelocitySet::velocitySet::cs2() * d_tau * d_u_inf; // At the intersection of East and West, only x derivative exists
                moments[4] = static_cast<scalar_t>(0);                                                     // At the intersection of East and West, only x derivative exists
                // moments[5] = static_cast<scalar_t>(0);
                moments[6] = static_cast<scalar_t>(0);
                moments[7] = static_cast<scalar_t>(0);
                // moments[8] = (m_yz_I * (static_cast<scalar_t>(50) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf) - static_cast<scalar_t>(3) * d_u_inf * d_u_inf + static_cast<scalar_t>(2)) / (static_cast<scalar_t>(18) * (m_yz_I - d_omega * m_yz_I + static_cast<scalar_t>(1)));
                moments[8] = static_cast<scalar_t>(0); // At the lip, this moment vanishes
                moments[9] = static_cast<scalar_t>(0);

                return;
            }
            }
        }

    private:
    };
}

#endif