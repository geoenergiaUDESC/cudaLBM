/**
Filename: halo.cuh
Contents: A class handling the device halo
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_HALO_CUH
#define __MBLBM_HALO_CUH

#include "haloFace.cuh"

namespace mbLBM
{
    namespace host
    {

    }

    namespace device
    {
        class halo
        {
        public:
            [[nodiscard]] halo(
                const label_t nx,
                const label_t ny,
                const label_t nz) noexcept
                : fGhost_(haloFace(nx, ny, nz)),
                  gGhost_(haloFace(nx, ny, nz)) {};

            __host__ inline void swap() noexcept
            {
                interfaceSwap(fGhost().x0Ref(), gGhost().x0Ref());
                interfaceSwap(fGhost().x1Ref(), gGhost().x1Ref());
                interfaceSwap(fGhost().y0Ref(), gGhost().y0Ref());
                interfaceSwap(fGhost().y1Ref(), gGhost().y1Ref());
                interfaceSwap(fGhost().z0Ref(), gGhost().z0Ref());
                interfaceSwap(fGhost().z1Ref(), gGhost().z1Ref());
            }

            __host__ void interfaceSwap(scalar_t *&pt1, scalar_t *&pt2) const noexcept
            {
                scalar_t *temp = pt1;
                pt1 = pt2;
                pt2 = temp;
            }

            __device__ __host__ [[nodiscard]] inline constexpr const haloFace &fGhost() const noexcept
            {
                return fGhost_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const haloFace &gGhost() const noexcept
            {
                return gGhost_;
            }

            __device__ __host__ [[nodiscard]] inline constexpr haloFace &fGhost() noexcept
            {
                return fGhost_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr haloFace &gGhost() noexcept
            {
                return gGhost_;
            }

            __device__ inline void popLoad(scalar_t pop[19]) const noexcept
            {
                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;

                const label_t txp1 = (tx + 1 + BLOCK_NX) % BLOCK_NX;
                const label_t txm1 = (tx - 1 + BLOCK_NX) % BLOCK_NX;

                const label_t typ1 = (ty + 1 + BLOCK_NY) % BLOCK_NY;
                const label_t tym1 = (ty - 1 + BLOCK_NY) % BLOCK_NY;

                const label_t tzp1 = (tz + 1 + BLOCK_NZ) % BLOCK_NZ;
                const label_t tzm1 = (tz - 1 + BLOCK_NZ) % BLOCK_NZ;

                const label_t bxm1 = (bx - 1 + NUM_BLOCK_X) % NUM_BLOCK_X;
                const label_t bxp1 = (bx + 1 + NUM_BLOCK_X) % NUM_BLOCK_X;

                const label_t bym1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
                const label_t byp1 = (by + 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;

                const label_t bzm1 = (bz - 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;
                const label_t bzp1 = (bz + 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;

                if (tx == 0)
                { // w
                    pop[1] = fGhost_.x1()[idxPopX<0>(ty, tz, bxm1, by, bz)];
                    pop[7] = fGhost_.x1()[idxPopX<1>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)];
                    pop[9] = fGhost_.x1()[idxPopX<2>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))];
                    pop[13] = fGhost_.x1()[idxPopX<3>(typ1, tz, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bz)];
                    pop[15] = fGhost_.x1()[idxPopX<4>(ty, tzp1, bxm1, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                }
                else if (tx == (BLOCK_NX - 1))
                { // e
                    pop[2] = fGhost_.x0()[idxPopX<0>(ty, tz, bxp1, by, bz)];
                    pop[8] = fGhost_.x0()[idxPopX<1>(typ1, tz, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bz)];
                    pop[10] = fGhost_.x0()[idxPopX<2>(ty, tzp1, bxp1, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                    pop[14] = fGhost_.x0()[idxPopX<3>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)];
                    pop[16] = fGhost_.x0()[idxPopX<4>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))];
                }

                if (ty == 0)
                { // s
                    pop[3] = fGhost_.y1()[idxPopY<0>(tx, tz, bx, bym1, bz)];
                    pop[7] = fGhost_.y1()[idxPopY<1>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)];
                    pop[11] = fGhost_.y1()[idxPopY<2>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))];
                    pop[14] = fGhost_.y1()[idxPopY<3>(txp1, tz, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), bym1, bz)];
                    pop[17] = fGhost_.y1()[idxPopY<4>(tx, tzp1, bx, bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                }
                else if (ty == (BLOCK_NY - 1))
                { // n
                    pop[4] = fGhost_.y0()[idxPopY<0>(tx, tz, bx, byp1, bz)];
                    pop[8] = fGhost_.y0()[idxPopY<1>(txp1, tz, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), byp1, bz)];
                    pop[12] = fGhost_.y0()[idxPopY<2>(tx, tzp1, bx, byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
                    pop[13] = fGhost_.y0()[idxPopY<3>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)];
                    pop[18] = fGhost_.y0()[idxPopY<4>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))];
                }

                if (tz == 0)
                { // b
                    pop[5] = fGhost_.z1()[idxPopZ<0>(tx, ty, bx, by, bzm1)];
                    pop[9] = fGhost_.z1()[idxPopZ<1>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)];
                    pop[11] = fGhost_.z1()[idxPopZ<2>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)];
                    pop[16] = fGhost_.z1()[idxPopZ<3>(txp1, ty, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bzm1)];
                    pop[18] = fGhost_.z1()[idxPopZ<4>(tx, typ1, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzm1)];
                }
                else if (tz == (BLOCK_NZ - 1))
                { // f
                    pop[6] = fGhost_.z0()[idxPopZ<0>(tx, ty, bx, by, bzp1)];
                    pop[10] = fGhost_.z0()[idxPopZ<1>(txp1, ty, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bzp1)];
                    pop[12] = fGhost_.z0()[idxPopZ<2>(tx, typ1, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzp1)];
                    pop[15] = fGhost_.z0()[idxPopZ<3>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)];
                    pop[17] = fGhost_.z0()[idxPopZ<4>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)];
                }
            }

            __device__ inline void populationSave(const scalar_t pop[19]) noexcept
            {
                const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
                const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
                const label_t z = threadIdx.z + blockDim.z * blockIdx.z;

                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;

                /* write to global pop */
                if (INTERFACE_BC_WEST(x))
                { // w
                    gGhost_.x0()[idxPopX<0>(ty, tz, bx, by, bz)] = pop[2];
                    gGhost_.x0()[idxPopX<1>(ty, tz, bx, by, bz)] = pop[8];
                    gGhost_.x0()[idxPopX<2>(ty, tz, bx, by, bz)] = pop[10];
                    gGhost_.x0()[idxPopX<3>(ty, tz, bx, by, bz)] = pop[14];
                    gGhost_.x0()[idxPopX<4>(ty, tz, bx, by, bz)] = pop[16];
                }
                if (INTERFACE_BC_EAST(x))
                { // e
                    gGhost_.x1()[idxPopX<0>(ty, tz, bx, by, bz)] = pop[1];
                    gGhost_.x1()[idxPopX<1>(ty, tz, bx, by, bz)] = pop[7];
                    gGhost_.x1()[idxPopX<2>(ty, tz, bx, by, bz)] = pop[9];
                    gGhost_.x1()[idxPopX<3>(ty, tz, bx, by, bz)] = pop[13];
                    gGhost_.x1()[idxPopX<4>(ty, tz, bx, by, bz)] = pop[15];
                }

                if (INTERFACE_BC_SOUTH(y))
                { // s
                    gGhost_.y0()[idxPopY<0>(tx, tz, bx, by, bz)] = pop[4];
                    gGhost_.y0()[idxPopY<1>(tx, tz, bx, by, bz)] = pop[8];
                    gGhost_.y0()[idxPopY<2>(tx, tz, bx, by, bz)] = pop[12];
                    gGhost_.y0()[idxPopY<3>(tx, tz, bx, by, bz)] = pop[13];
                    gGhost_.y0()[idxPopY<4>(tx, tz, bx, by, bz)] = pop[18];
                }
                if (INTERFACE_BC_NORTH(y))
                { // n
                    gGhost_.y1()[idxPopY<0>(tx, tz, bx, by, bz)] = pop[3];
                    gGhost_.y1()[idxPopY<1>(tx, tz, bx, by, bz)] = pop[7];
                    gGhost_.y1()[idxPopY<2>(tx, tz, bx, by, bz)] = pop[11];
                    gGhost_.y1()[idxPopY<3>(tx, tz, bx, by, bz)] = pop[14];
                    gGhost_.y1()[idxPopY<4>(tx, tz, bx, by, bz)] = pop[17];
                }

                if (INTERFACE_BC_BACK(z))
                { // b
                    gGhost_.z0()[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[6];
                    gGhost_.z0()[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[10];
                    gGhost_.z0()[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[12];
                    gGhost_.z0()[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[15];
                    gGhost_.z0()[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[17];
                }
                if (INTERFACE_BC_FRONT(z))
                {
                    gGhost_.z1()[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[5];
                    gGhost_.z1()[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[9];
                    gGhost_.z1()[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[11];
                    gGhost_.z1()[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[16];
                    gGhost_.z1()[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[18];
                }
            }

        private:
            haloFace fGhost_;
            haloFace gGhost_;
        };
    }
}

#endif