/**
Filename: halo.cuh
Contents: A class handling the device halo
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_HALO_CUH
#define __MBLBM_HALO_CUH

#include "haloFace.cuh"

namespace LBM
{
    namespace host
    {

    }

    namespace device
    {
        class halo
        {
        public:
            /**
             * @brief Constructs the block halo from a host array and a latticeMesh object
             * @param f The std::vector to be allocated on the device
             * @return An array object constructed from f
             **/
            [[nodiscard]] halo(const std::vector<scalar_t> &fMom, const host::latticeMesh &mesh) noexcept
                : fGhost_(haloFace(fMom, mesh)),
                  gGhost_(haloFace(fMom, mesh)) {};

            /**
             * @brief Default desructor for the halo class
             **/
            ~halo() {};

            /**
             * @brief Swaps the halo pointers
             **/
            __host__ inline void swap() noexcept
            {
                checkCudaErrorsInline(cudaDeviceSynchronize());
                std::swap(fGhost_.x0Ref(), gGhost_.x0Ref());
                std::swap(fGhost_.x1Ref(), gGhost_.x1Ref());
                std::swap(fGhost_.y0Ref(), gGhost_.y0Ref());
                std::swap(fGhost_.y1Ref(), gGhost_.y1Ref());
                std::swap(fGhost_.z0Ref(), gGhost_.z0Ref());
                std::swap(fGhost_.z1Ref(), gGhost_.z1Ref());
            }

            /**
             * @brief Provides read-only access to the halos
             * @return A const-qualified reference to the halos
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const haloFace &fGhost() const noexcept
            {
                return fGhost_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const haloFace &gGhost() const noexcept
            {
                return gGhost_;
            }

            /**
             * @brief Provides mutable access to the halos
             * @return A reference to the halos
             **/
            __device__ __host__ [[nodiscard]] inline constexpr haloFace &fGhost() noexcept
            {
                return fGhost_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr haloFace &gGhost() noexcept
            {
                return gGhost_;
            }

            /**
             * @brief Loads the populations at the halo points into pop
             * @param pop The population density array into which the halo points are to be loaded
             **/
            template <class VSet>
            __device__ inline void popLoad(scalar_t (&ptrRestrict pop)[VSet::Q()]) const noexcept
            {
                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;

                const label_t txp1 = (tx + 1 + block::nx()) % block::nx();
                const label_t txm1 = (tx - 1 + block::nx()) % block::nx();

                const label_t typ1 = (ty + 1 + block::ny()) % block::ny();
                const label_t tym1 = (ty - 1 + block::ny()) % block::ny();

                const label_t tzp1 = (tz + 1 + block::nz()) % block::nz();
                const label_t tzm1 = (tz - 1 + block::nz()) % block::nz();

                const label_t bxm1 = (bx - 1 + device::NUM_BLOCK_X) % device::NUM_BLOCK_X;
                const label_t bxp1 = (bx + 1 + device::NUM_BLOCK_X) % device::NUM_BLOCK_X;

                const label_t bym1 = (by - 1 + device::NUM_BLOCK_Y) % device::NUM_BLOCK_Y;
                const label_t byp1 = (by + 1 + device::NUM_BLOCK_Y) % device::NUM_BLOCK_Y;

                const label_t bzm1 = (bz - 1 + device::NUM_BLOCK_Z) % device::NUM_BLOCK_Z;
                const label_t bzp1 = (bz + 1 + device::NUM_BLOCK_Z) % device::NUM_BLOCK_Z;

                if (tx == 0)
                { // w
                    pop[1] = fGhost_.x1()[idxPopX<0, VSet::QF()>(ty, tz, bxm1, by, bz)];
                    pop[7] = fGhost_.x1()[idxPopX<1, VSet::QF()>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)];
                    pop[9] = fGhost_.x1()[idxPopX<2, VSet::QF()>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))];
                    pop[13] = fGhost_.x1()[idxPopX<3, VSet::QF()>(typ1, tz, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)];
                    pop[15] = fGhost_.x1()[idxPopX<4, VSet::QF()>(ty, tzp1, bxm1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
                }
                else if (tx == (block::nx() - 1))
                { // e
                    pop[2] = fGhost_.x0()[idxPopX<0, VSet::QF()>(ty, tz, bxp1, by, bz)];
                    pop[8] = fGhost_.x0()[idxPopX<1, VSet::QF()>(typ1, tz, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)];
                    pop[10] = fGhost_.x0()[idxPopX<2, VSet::QF()>(ty, tzp1, bxp1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
                    pop[14] = fGhost_.x0()[idxPopX<3, VSet::QF()>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)];
                    pop[16] = fGhost_.x0()[idxPopX<4, VSet::QF()>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))];
                }

                if (ty == 0)
                { // s
                    pop[3] = fGhost_.y1()[idxPopY<0, VSet::QF()>(tx, tz, bx, bym1, bz)];
                    pop[7] = fGhost_.y1()[idxPopY<1, VSet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)];
                    pop[11] = fGhost_.y1()[idxPopY<2, VSet::QF()>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))];
                    pop[14] = fGhost_.y1()[idxPopY<3, VSet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, bz)];
                    pop[17] = fGhost_.y1()[idxPopY<4, VSet::QF()>(tx, tzp1, bx, bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
                }
                else if (ty == (block::ny() - 1))
                { // n
                    pop[4] = fGhost_.y0()[idxPopY<0, VSet::QF()>(tx, tz, bx, byp1, bz)];
                    pop[8] = fGhost_.y0()[idxPopY<1, VSet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, bz)];
                    pop[12] = fGhost_.y0()[idxPopY<2, VSet::QF()>(tx, tzp1, bx, byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
                    pop[13] = fGhost_.y0()[idxPopY<3, VSet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)];
                    pop[18] = fGhost_.y0()[idxPopY<4, VSet::QF()>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))];
                }

                if (tz == 0)
                { // b
                    pop[5] = fGhost_.z1()[idxPopZ<0, VSet::QF()>(tx, ty, bx, by, bzm1)];
                    pop[9] = fGhost_.z1()[idxPopZ<1, VSet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)];
                    pop[11] = fGhost_.z1()[idxPopZ<2, VSet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)];
                    pop[16] = fGhost_.z1()[idxPopZ<3, VSet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzm1)];
                    pop[18] = fGhost_.z1()[idxPopZ<4, VSet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)];
                }
                else if (tz == (block::nz() - 1))
                { // f
                    pop[6] = fGhost_.z0()[idxPopZ<0, VSet::QF()>(tx, ty, bx, by, bzp1)];
                    pop[10] = fGhost_.z0()[idxPopZ<1, VSet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzp1)];
                    pop[12] = fGhost_.z0()[idxPopZ<2, VSet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)];
                    pop[15] = fGhost_.z0()[idxPopZ<3, VSet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)];
                    pop[17] = fGhost_.z0()[idxPopZ<4, VSet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)];
                }
            }

            /**
             * @brief Saves the populations in pop to the halo
             * @param pop The population density array from which the halo points are to be saved
             **/
            template <class VSet>
            __device__ inline void popSave(const scalar_t (&ptrRestrict pop)[VSet::Q()]) noexcept
            {
                const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
                const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
                const label_t z = threadIdx.z + blockDim.z * blockIdx.z;

                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                // const label_t bx = blockIdx.x;
                // const label_t by = blockIdx.y;
                // const label_t bz = blockIdx.z;

                /* write to global pop **/
                if (West(x))
                { // w
                    gGhost_.x0()[idxPopX<0, VSet::QF()>(ty, tz, blockIdx)] = pop[2];
                    gGhost_.x0()[idxPopX<1, VSet::QF()>(ty, tz, blockIdx)] = pop[8];
                    gGhost_.x0()[idxPopX<2, VSet::QF()>(ty, tz, blockIdx)] = pop[10];
                    gGhost_.x0()[idxPopX<3, VSet::QF()>(ty, tz, blockIdx)] = pop[14];
                    gGhost_.x0()[idxPopX<4, VSet::QF()>(ty, tz, blockIdx)] = pop[16];
                }
                if (East(x))
                { // e
                    gGhost_.x1()[idxPopX<0, VSet::QF()>(ty, tz, blockIdx)] = pop[1];
                    gGhost_.x1()[idxPopX<1, VSet::QF()>(ty, tz, blockIdx)] = pop[7];
                    gGhost_.x1()[idxPopX<2, VSet::QF()>(ty, tz, blockIdx)] = pop[9];
                    gGhost_.x1()[idxPopX<3, VSet::QF()>(ty, tz, blockIdx)] = pop[13];
                    gGhost_.x1()[idxPopX<4, VSet::QF()>(ty, tz, blockIdx)] = pop[15];
                }

                if (South(y))
                { // s
                    gGhost_.y0()[idxPopY<0, VSet::QF()>(tx, tz, blockIdx)] = pop[4];
                    gGhost_.y0()[idxPopY<1, VSet::QF()>(tx, tz, blockIdx)] = pop[8];
                    gGhost_.y0()[idxPopY<2, VSet::QF()>(tx, tz, blockIdx)] = pop[12];
                    gGhost_.y0()[idxPopY<3, VSet::QF()>(tx, tz, blockIdx)] = pop[13];
                    gGhost_.y0()[idxPopY<4, VSet::QF()>(tx, tz, blockIdx)] = pop[18];
                }
                if (North(y))
                { // n
                    gGhost_.y1()[idxPopY<0, VSet::QF()>(tx, tz, blockIdx)] = pop[3];
                    gGhost_.y1()[idxPopY<1, VSet::QF()>(tx, tz, blockIdx)] = pop[7];
                    gGhost_.y1()[idxPopY<2, VSet::QF()>(tx, tz, blockIdx)] = pop[11];
                    gGhost_.y1()[idxPopY<3, VSet::QF()>(tx, tz, blockIdx)] = pop[14];
                    gGhost_.y1()[idxPopY<4, VSet::QF()>(tx, tz, blockIdx)] = pop[17];
                }

                if (Back(z))
                { // b
                    gGhost_.z0()[idxPopZ<0, VSet::QF()>(tx, ty, blockIdx)] = pop[6];
                    gGhost_.z0()[idxPopZ<1, VSet::QF()>(tx, ty, blockIdx)] = pop[10];
                    gGhost_.z0()[idxPopZ<2, VSet::QF()>(tx, ty, blockIdx)] = pop[12];
                    gGhost_.z0()[idxPopZ<3, VSet::QF()>(tx, ty, blockIdx)] = pop[15];
                    gGhost_.z0()[idxPopZ<4, VSet::QF()>(tx, ty, blockIdx)] = pop[17];
                }
                if (Front(z))
                {
                    gGhost_.z1()[idxPopZ<0, VSet::QF()>(tx, ty, blockIdx)] = pop[5];
                    gGhost_.z1()[idxPopZ<1, VSet::QF()>(tx, ty, blockIdx)] = pop[9];
                    gGhost_.z1()[idxPopZ<2, VSet::QF()>(tx, ty, blockIdx)] = pop[11];
                    gGhost_.z1()[idxPopZ<3, VSet::QF()>(tx, ty, blockIdx)] = pop[16];
                    gGhost_.z1()[idxPopZ<4, VSet::QF()>(tx, ty, blockIdx)] = pop[18];
                }
            }

        private:
            /**
             * @brief The individual halo objects
             **/
            haloFace fGhost_;
            haloFace gGhost_;

            /**
             * @brief Check whether the current x, y or z index is at a block boundary
             * @param xyz The coordinate in the x, y or z directions
             * @return True if x, y or z is at a block boundary, false otherwise
             **/
            __device__ [[nodiscard]] inline bool West(const label_t x) const noexcept
            {
                return (threadIdx.x == 0 && x != 0);
            }
            __device__ [[nodiscard]] inline bool East(const label_t x) const noexcept
            {
                return (threadIdx.x == (block::nx() - 1) && x != (device::nx - 1));
            }
            __device__ [[nodiscard]] inline bool South(const label_t y) const noexcept
            {
                return (threadIdx.y == 0 && y != 0);
            }
            __device__ [[nodiscard]] inline bool North(const label_t y) const noexcept
            {
                return (threadIdx.y == (block::ny() - 1) && y != (device::ny - 1));
            }
            __device__ [[nodiscard]] inline bool Back(const label_t z) const noexcept
            {
                return (threadIdx.z == 0 && z != 0);
            }
            __device__ [[nodiscard]] inline bool Front(const label_t z) const noexcept
            {
                return (threadIdx.z == (block::nz() - 1) && z != (device::nz - 1));
            }
        };
    }
}

#endif