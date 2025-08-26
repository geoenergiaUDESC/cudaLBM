/**
Filename: halo.cuh
Contents: A class handling the device halo
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_HALO_CUH
#define __MBLBM_HALO_CUH

namespace LBM
{
    namespace device
    {
        template <class VSet>
        class halo
        {
        public:
            /**
             * @brief Constructs the block halo from a host array and a latticeMesh object
             * @param f The std::vector to be allocated on the device
             * @return An array object constructed from f
             **/
            // [[nodiscard]] halo(const std::vector<scalar_t> &fMom, const host::latticeMesh &mesh) noexcept
            //     : fGhost_(haloFace<VSet>(fMom, mesh)),
            //       gGhost_(haloFace<VSet>(fMom, mesh)) {};

            [[nodiscard]] halo(const std::vector<std::vector<scalar_t>> &fMom, const host::latticeMesh &mesh) noexcept
                : fGhost_(haloFace<VSet>(fMom, mesh)),
                  gGhost_(haloFace<VSet>(fMom, mesh)) {};

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
             * @brief Provides access to the read halo
             * @return A collection of const-qualified pointers to the read halo
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> fGhost() const noexcept
            {
                return {fGhost_.x0Const(), fGhost_.x1Const(), fGhost_.y0Const(), fGhost_.y1Const(), fGhost_.z0Const(), fGhost_.z1Const()};
            }

            /**
             * @brief Provides access to the read halo
             * @return A collection of mutable pointers to the read halo
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, scalar_t> gGhost() noexcept
            {
                return {gGhost_.x0(), gGhost_.x1(), gGhost_.y0(), gGhost_.y1(), gGhost_.z0(), gGhost_.z1()};
            }

            /**
             * @brief Loads the populations at the halo points into pop
             * @param pop The population density array into which the halo points are to be loaded
             **/
            __device__ static inline void popLoad(
                scalar_t (&ptrRestrict pop)[VSet::Q()],
                const scalar_t *const ptrRestrict fx0,
                const scalar_t *const ptrRestrict fx1,
                const scalar_t *const ptrRestrict fy0,
                const scalar_t *const ptrRestrict fy1,
                const scalar_t *const ptrRestrict fz0,
                const scalar_t *const ptrRestrict fz1) noexcept
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
                    pop[1] = __ldg(&fx1[idxPopX<0, VSet::QF()>(ty, tz, bxm1, by, bz)]);
                    pop[7] = __ldg(&fx1[idxPopX<1, VSet::QF()>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)]);
                    pop[9] = __ldg(&fx1[idxPopX<2, VSet::QF()>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))]);
                    pop[13] = __ldg(&fx1[idxPopX<3, VSet::QF()>(typ1, tz, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)]);
                    pop[15] = __ldg(&fx1[idxPopX<4, VSet::QF()>(ty, tzp1, bxm1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                }
                else if (tx == (block::nx() - 1))
                { // e
                    pop[2] = __ldg(&fx0[idxPopX<0, VSet::QF()>(ty, tz, bxp1, by, bz)]);
                    pop[8] = __ldg(&fx0[idxPopX<1, VSet::QF()>(typ1, tz, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)]);
                    pop[10] = __ldg(&fx0[idxPopX<2, VSet::QF()>(ty, tzp1, bxp1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    pop[14] = __ldg(&fx0[idxPopX<3, VSet::QF()>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)]);
                    pop[16] = __ldg(&fx0[idxPopX<4, VSet::QF()>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))]);
                }

                if (ty == 0)
                { // s
                    pop[3] = __ldg(&fy1[idxPopY<0, VSet::QF()>(tx, tz, bx, bym1, bz)]);
                    pop[7] = __ldg(&fy1[idxPopY<1, VSet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)]);
                    pop[11] = __ldg(&fy1[idxPopY<2, VSet::QF()>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))]);
                    pop[14] = __ldg(&fy1[idxPopY<3, VSet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, bz)]);
                    pop[17] = __ldg(&fy1[idxPopY<4, VSet::QF()>(tx, tzp1, bx, bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                }
                else if (ty == (block::ny() - 1))
                { // n
                    pop[4] = __ldg(&fy0[idxPopY<0, VSet::QF()>(tx, tz, bx, byp1, bz)]);
                    pop[8] = __ldg(&fy0[idxPopY<1, VSet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, bz)]);
                    pop[12] = __ldg(&fy0[idxPopY<2, VSet::QF()>(tx, tzp1, bx, byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    pop[13] = __ldg(&fy0[idxPopY<3, VSet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)]);
                    pop[18] = __ldg(&fy0[idxPopY<4, VSet::QF()>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))]);
                }

                if (tz == 0)
                { // b
                    pop[5] = __ldg(&fz1[idxPopZ<0, VSet::QF()>(tx, ty, bx, by, bzm1)]);
                    pop[9] = __ldg(&fz1[idxPopZ<1, VSet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)]);
                    pop[11] = __ldg(&fz1[idxPopZ<2, VSet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)]);
                    pop[16] = __ldg(&fz1[idxPopZ<3, VSet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzm1)]);
                    pop[18] = __ldg(&fz1[idxPopZ<4, VSet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)]);
                }
                else if (tz == (block::nz() - 1))
                { // f
                    pop[6] = __ldg(&fz0[idxPopZ<0, VSet::QF()>(tx, ty, bx, by, bzp1)]);
                    pop[10] = __ldg(&fz0[idxPopZ<1, VSet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzp1)]);
                    pop[12] = __ldg(&fz0[idxPopZ<2, VSet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)]);
                    pop[15] = __ldg(&fz0[idxPopZ<3, VSet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)]);
                    pop[17] = __ldg(&fz0[idxPopZ<4, VSet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)]);
                }
            }

            /**
             * @brief Saves the populations in pop to the halo
             * @param pop The population density array from which the halo points are to be saved
             **/
            __device__ static inline void popSave(
                const scalar_t (&ptrRestrict pop)[VSet::Q()],
                scalar_t *const ptrRestrict gx0,
                scalar_t *const ptrRestrict gx1,
                scalar_t *const ptrRestrict gy0,
                scalar_t *const ptrRestrict gy1,
                scalar_t *const ptrRestrict gz0,
                scalar_t *const ptrRestrict gz1) noexcept
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
                    gx0[idxPopX<0, VSet::QF()>(ty, tz, blockIdx)] = pop[2];
                    gx0[idxPopX<1, VSet::QF()>(ty, tz, blockIdx)] = pop[8];
                    gx0[idxPopX<2, VSet::QF()>(ty, tz, blockIdx)] = pop[10];
                    gx0[idxPopX<3, VSet::QF()>(ty, tz, blockIdx)] = pop[14];
                    gx0[idxPopX<4, VSet::QF()>(ty, tz, blockIdx)] = pop[16];
                }
                if (East(x))
                { // e
                    gx1[idxPopX<0, VSet::QF()>(ty, tz, blockIdx)] = pop[1];
                    gx1[idxPopX<1, VSet::QF()>(ty, tz, blockIdx)] = pop[7];
                    gx1[idxPopX<2, VSet::QF()>(ty, tz, blockIdx)] = pop[9];
                    gx1[idxPopX<3, VSet::QF()>(ty, tz, blockIdx)] = pop[13];
                    gx1[idxPopX<4, VSet::QF()>(ty, tz, blockIdx)] = pop[15];
                }

                if (South(y))
                { // s
                    gy0[idxPopY<0, VSet::QF()>(tx, tz, blockIdx)] = pop[4];
                    gy0[idxPopY<1, VSet::QF()>(tx, tz, blockIdx)] = pop[8];
                    gy0[idxPopY<2, VSet::QF()>(tx, tz, blockIdx)] = pop[12];
                    gy0[idxPopY<3, VSet::QF()>(tx, tz, blockIdx)] = pop[13];
                    gy0[idxPopY<4, VSet::QF()>(tx, tz, blockIdx)] = pop[18];
                }
                if (North(y))
                { // n
                    gy1[idxPopY<0, VSet::QF()>(tx, tz, blockIdx)] = pop[3];
                    gy1[idxPopY<1, VSet::QF()>(tx, tz, blockIdx)] = pop[7];
                    gy1[idxPopY<2, VSet::QF()>(tx, tz, blockIdx)] = pop[11];
                    gy1[idxPopY<3, VSet::QF()>(tx, tz, blockIdx)] = pop[14];
                    gy1[idxPopY<4, VSet::QF()>(tx, tz, blockIdx)] = pop[17];
                }

                if (Back(z))
                { // b
                    gz0[idxPopZ<0, VSet::QF()>(tx, ty, blockIdx)] = pop[6];
                    gz0[idxPopZ<1, VSet::QF()>(tx, ty, blockIdx)] = pop[10];
                    gz0[idxPopZ<2, VSet::QF()>(tx, ty, blockIdx)] = pop[12];
                    gz0[idxPopZ<3, VSet::QF()>(tx, ty, blockIdx)] = pop[15];
                    gz0[idxPopZ<4, VSet::QF()>(tx, ty, blockIdx)] = pop[17];
                }
                if (Front(z))
                {
                    gz1[idxPopZ<0, VSet::QF()>(tx, ty, blockIdx)] = pop[5];
                    gz1[idxPopZ<1, VSet::QF()>(tx, ty, blockIdx)] = pop[9];
                    gz1[idxPopZ<2, VSet::QF()>(tx, ty, blockIdx)] = pop[11];
                    gz1[idxPopZ<3, VSet::QF()>(tx, ty, blockIdx)] = pop[16];
                    gz1[idxPopZ<4, VSet::QF()>(tx, ty, blockIdx)] = pop[18];
                }
            }

        private:
            /**
             * @brief The individual halo objects
             **/
            haloFace<VSet> fGhost_;
            haloFace<VSet> gGhost_;

            /**
             * @brief Check whether the current x, y or z index is at a block boundary
             * @param xyz The coordinate in the x, y or z directions
             * @return True if x, y or z is at a block boundary, false otherwise
             **/
            __device__ [[nodiscard]] static inline bool West(const label_t x) noexcept
            {
                return (threadIdx.x == 0 && x != 0);
            }
            __device__ [[nodiscard]] static inline bool East(const label_t x) noexcept
            {
                return (threadIdx.x == (block::nx() - 1) && x != (device::nx - 1));
            }
            __device__ [[nodiscard]] static inline bool South(const label_t y) noexcept
            {
                return (threadIdx.y == 0 && y != 0);
            }
            __device__ [[nodiscard]] static inline bool North(const label_t y) noexcept
            {
                return (threadIdx.y == (block::ny() - 1) && y != (device::ny - 1));
            }
            __device__ [[nodiscard]] static inline bool Back(const label_t z) noexcept
            {
                return (threadIdx.z == 0 && z != 0);
            }
            __device__ [[nodiscard]] static inline bool Front(const label_t z) noexcept
            {
                return (threadIdx.z == (block::nz() - 1) && z != (device::nz - 1));
            }
        };
    }
}

#endif