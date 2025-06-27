/**
Filename: halo.cuh
Contents: A class handling the device halo (Optimized with Shared Memory)
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/
// PASTE THIS CODE INTO: halo.cuh (Complete File)

#ifndef __MBLBM_HALO_CUH
#define __MBLBM_HALO_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "globalFunctions.cuh"
#include "haloFace.cuh"

namespace LBM
{
    namespace device
    {
        class halo
        {
        public:
            // --- Constructor and Swap function are UNCHANGED ---
            [[nodiscard]] halo(
                const std::vector<scalar_t> &fMom,
                const host::latticeMesh &mesh) noexcept
                : fGhost_(haloFace(fMom, mesh)),
                  gGhost_(haloFace(fMom, mesh)) {};

            __host__ inline void swap() noexcept
            {
                interfaceSwap(fGhost().x0Ref(), gGhost().x0Ref());
                interfaceSwap(fGhost().x1Ref(), gGhost().x1Ref());
                interfaceSwap(fGhost().y0Ref(), gGhost().y0Ref());
                interfaceSwap(fGhost().y1Ref(), gGhost().y1Ref());
                interfaceSwap(fGhost().z0Ref(), gGhost().z0Ref());
                interfaceSwap(fGhost().z1Ref(), gGhost().z1Ref());
            }

            // --- Accessors are UNCHANGED ---
            __device__ __host__ [[nodiscard]] inline constexpr const haloFace &fGhost() const noexcept { return fGhost_; }
            __device__ __host__ [[nodiscard]] inline constexpr const haloFace &gGhost() const noexcept { return gGhost_; }
            __device__ __host__ [[nodiscard]] inline constexpr haloFace &fGhost() noexcept { return fGhost_; }
            __device__ __host__ [[nodiscard]] inline constexpr haloFace &gGhost() noexcept { return gGhost_; }

            // --- OPTIMIZATION: NEW TWO-STAGE HALO EXCHANGE FUNCTIONS ---

            template <class VSet>
            __device__ void halo_global_to_shared_load(scalar_t *s_buffer) const noexcept
            {
                constexpr label_t QF = VSet::QF();
                constexpr label_t NX = block::nx();
                constexpr label_t NY = block::ny();
                constexpr label_t NZ = block::nz();

                scalar_t *s_load_west = s_buffer;
                scalar_t *s_load_east = s_load_west + QF * NY * NZ;
                scalar_t *s_load_south = s_load_east + QF * NY * NZ;
                scalar_t *s_load_north = s_load_south + QF * NX * NZ;
                scalar_t *s_load_back = s_load_north + QF * NX * NZ;
                scalar_t *s_load_front = s_load_back + QF * NX * NY;

                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;
                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;
                const label_t bxm1 = (bx - 1 + d_NUM_BLOCK_X) % d_NUM_BLOCK_X;
                const label_t bxp1 = (bx + 1 + d_NUM_BLOCK_X) % d_NUM_BLOCK_X;
                const label_t bym1 = (by - 1 + d_NUM_BLOCK_Y) % d_NUM_BLOCK_Y;
                const label_t byp1 = (by + 1 + d_NUM_BLOCK_Y) % d_NUM_BLOCK_Y;
                const label_t bzm1 = (bz - 1 + d_NUM_BLOCK_Z) % d_NUM_BLOCK_Z;
                const label_t bzp1 = (bz + 1 + d_NUM_BLOCK_Z) % d_NUM_BLOCK_Z;
                const label_t txp1 = (tx + 1 + NX) % NX;
                const label_t txm1 = (tx - 1 + NX) % NX;
                const label_t typ1 = (ty + 1 + NY) % NY;
                const label_t tym1 = (ty - 1 + NY) % NY;
                const label_t tzp1 = (tz + 1 + NZ) % NZ;
                const label_t tzm1 = (tz - 1 + NZ) % NZ;

                if (tx == 0)
                {
                    s_load_west[0 * NY * NZ + tz * NY + ty] = fGhost().x1()[idxPopX<0, QF>(ty, tz, bxm1, by, bz)];
                    s_load_west[1 * NY * NZ + tz * NY + ty] = fGhost().x1()[idxPopX<1, QF>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)];
                    s_load_west[2 * NY * NZ + tz * NY + ty] = fGhost().x1()[idxPopX<2, QF>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))];
                    s_load_west[3 * NY * NZ + tz * NY + ty] = fGhost().x1()[idxPopX<3, QF>(typ1, tz, bxm1, ((ty == (NY - 1)) ? byp1 : by), bz)];
                    s_load_west[4 * NY * NZ + tz * NY + ty] = fGhost().x1()[idxPopX<4, QF>(ty, tzp1, bxm1, by, ((tz == (NZ - 1)) ? bzp1 : bz))];
                }
                else if (tx == (NX - 1))
                {
                    s_load_east[0 * NY * NZ + tz * NY + ty] = fGhost().x0()[idxPopX<0, QF>(ty, tz, bxp1, by, bz)];
                    s_load_east[1 * NY * NZ + tz * NY + ty] = fGhost().x0()[idxPopX<1, QF>(typ1, tz, bxp1, ((ty == (NY - 1)) ? byp1 : by), bz)];
                    s_load_east[2 * NY * NZ + tz * NY + ty] = fGhost().x0()[idxPopX<2, QF>(ty, tzp1, bxp1, by, ((tz == (NZ - 1)) ? bzp1 : bz))];
                    s_load_east[3 * NY * NZ + tz * NY + ty] = fGhost().x0()[idxPopX<3, QF>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)];
                    s_load_east[4 * NY * NZ + tz * NY + ty] = fGhost().x0()[idxPopX<4, QF>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))];
                }
                if (ty == 0)
                {
                    s_load_south[0 * NX * NZ + tz * NX + tx] = fGhost().y1()[idxPopY<0, QF>(tx, tz, bx, bym1, bz)];
                    s_load_south[1 * NX * NZ + tz * NX + tx] = fGhost().y1()[idxPopY<1, QF>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)];
                    s_load_south[2 * NX * NZ + tz * NX + tx] = fGhost().y1()[idxPopY<2, QF>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))];
                    s_load_south[3 * NX * NZ + tz * NX + tx] = fGhost().y1()[idxPopY<3, QF>(txp1, tz, ((tx == (NX - 1)) ? bxp1 : bx), bym1, bz)];
                    s_load_south[4 * NX * NZ + tz * NX + tx] = fGhost().y1()[idxPopY<4, QF>(tx, tzp1, bx, bym1, ((tz == (NZ - 1)) ? bzp1 : bz))];
                }
                else if (ty == (NY - 1))
                {
                    s_load_north[0 * NX * NZ + tz * NX + tx] = fGhost().y0()[idxPopY<0, QF>(tx, tz, bx, byp1, bz)];
                    s_load_north[1 * NX * NZ + tz * NX + tx] = fGhost().y0()[idxPopY<1, QF>(txp1, tz, ((tx == (NX - 1)) ? bxp1 : bx), byp1, bz)];
                    s_load_north[2 * NX * NZ + tz * NX + tx] = fGhost().y0()[idxPopY<2, QF>(tx, tzp1, bx, byp1, ((tz == (NZ - 1)) ? bzp1 : bz))];
                    s_load_north[3 * NX * NZ + tz * NX + tx] = fGhost().y0()[idxPopY<3, QF>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)];
                    s_load_north[4 * NX * NZ + tz * NX + tx] = fGhost().y0()[idxPopY<4, QF>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))];
                }
                if (tz == 0)
                {
                    s_load_back[0 * NX * NY + ty * NX + tx] = fGhost().z1()[idxPopZ<0, QF>(tx, ty, bx, by, bzm1)];
                    s_load_back[1 * NX * NY + ty * NX + tx] = fGhost().z1()[idxPopZ<1, QF>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)];
                    s_load_back[2 * NX * NY + ty * NX + tx] = fGhost().z1()[idxPopZ<2, QF>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)];
                    s_load_back[3 * NX * NY + ty * NX + tx] = fGhost().z1()[idxPopZ<3, QF>(txp1, ty, ((tx == (NX - 1)) ? bxp1 : bx), by, bzm1)];
                    s_load_back[4 * NX * NY + ty * NX + tx] = fGhost().z1()[idxPopZ<4, QF>(tx, typ1, bx, ((ty == (NY - 1)) ? byp1 : by), bzm1)];
                }
                else if (tz == (NZ - 1))
                {
                    s_load_front[0 * NX * NY + ty * NX + tx] = fGhost().z0()[idxPopZ<0, QF>(tx, ty, bx, by, bzp1)];
                    s_load_front[1 * NX * NY + ty * NX + tx] = fGhost().z0()[idxPopZ<1, QF>(txp1, ty, ((tx == (NX - 1)) ? bxp1 : bx), by, bzp1)];
                    s_load_front[2 * NX * NY + ty * NX + tx] = fGhost().z0()[idxPopZ<2, QF>(tx, typ1, bx, ((ty == (NY - 1)) ? byp1 : by), bzp1)];
                    s_load_front[3 * NX * NY + ty * NX + tx] = fGhost().z0()[idxPopZ<3, QF>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)];
                    s_load_front[4 * NX * NY + ty * NX + tx] = fGhost().z0()[idxPopZ<4, QF>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)];
                }
            }

            template <class VSet>
            __device__ void popLoad_from_shared(scalar_t (&pop)[VSet::Q()], const scalar_t *s_buffer) const noexcept
            {
                constexpr label_t QF = VSet::QF();
                constexpr label_t NX = block::nx();
                constexpr label_t NY = block::ny();
                constexpr label_t NZ = block::nz();

                const scalar_t *s_load_west = s_buffer;
                const scalar_t *s_load_east = s_load_west + QF * NY * NZ;
                const scalar_t *s_load_south = s_load_east + QF * NY * NZ;
                const scalar_t *s_load_north = s_load_south + QF * NX * NZ;
                const scalar_t *s_load_back = s_load_north + QF * NX * NZ;
                const scalar_t *s_load_front = s_load_back + QF * NX * NY;

                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                if (tx == 0)
                {
                    pop[1] = s_load_west[0 * NY * NZ + tz * NY + ty];
                    pop[7] = s_load_west[1 * NY * NZ + tz * NY + ty];
                    pop[9] = s_load_west[2 * NY * NZ + tz * NY + ty];
                    pop[13] = s_load_west[3 * NY * NZ + tz * NY + ty];
                    pop[15] = s_load_west[4 * NY * NZ + tz * NY + ty];
                }
                else if (tx == (NX - 1))
                {
                    pop[2] = s_load_east[0 * NY * NZ + tz * NY + ty];
                    pop[8] = s_load_east[1 * NY * NZ + tz * NY + ty];
                    pop[10] = s_load_east[2 * NY * NZ + tz * NY + ty];
                    pop[14] = s_load_east[3 * NY * NZ + tz * NY + ty];
                    pop[16] = s_load_east[4 * NY * NZ + tz * NY + ty];
                }
                if (ty == 0)
                {
                    pop[3] = s_load_south[0 * NX * NZ + tz * NX + tx];
                    pop[7] = s_load_south[1 * NX * NZ + tz * NX + tx];
                    pop[11] = s_load_south[2 * NX * NZ + tz * NX + tx];
                    pop[14] = s_load_south[3 * NX * NZ + tz * NX + tx];
                    pop[17] = s_load_south[4 * NX * NZ + tz * NX + tx];
                }
                else if (ty == (NY - 1))
                {
                    pop[4] = s_load_north[0 * NX * NZ + tz * NX + tx];
                    pop[8] = s_load_north[1 * NX * NZ + tz * NX + tx];
                    pop[12] = s_load_north[2 * NX * NZ + tz * NX + tx];
                    pop[13] = s_load_north[3 * NX * NZ + tz * NX + tx];
                    pop[18] = s_load_north[4 * NX * NZ + tz * NX + tx];
                }
                if (tz == 0)
                {
                    pop[5] = s_load_back[0 * NX * NY + ty * NX + tx];
                    pop[9] = s_load_back[1 * NX * NY + ty * NX + tx];
                    pop[11] = s_load_back[2 * NX * NY + ty * NX + tx];
                    pop[16] = s_load_back[3 * NX * NY + ty * NX + tx];
                    pop[18] = s_load_back[4 * NX * NY + ty * NX + tx];
                }
                else if (tz == (NZ - 1))
                {
                    pop[6] = s_load_front[0 * NX * NY + ty * NX + tx];
                    pop[10] = s_load_front[1 * NX * NY + ty * NX + tx];
                    pop[12] = s_load_front[2 * NX * NY + ty * NX + tx];
                    pop[15] = s_load_front[3 * NX * NY + ty * NX + tx];
                    pop[17] = s_load_front[4 * NX * NY + ty * NX + tx];
                }
            }

            template <class VSet>
            __device__ void popSave_to_shared(const scalar_t (&pop)[VSet::Q()], scalar_t *s_buffer) noexcept
            {
                constexpr label_t QF = VSet::QF();
                constexpr label_t NX = block::nx();
                constexpr label_t NY = block::ny();
                constexpr label_t NZ = block::nz();

                scalar_t *s_save_west = s_buffer;
                scalar_t *s_save_east = s_save_west + QF * NY * NZ;
                scalar_t *s_save_south = s_save_east + QF * NY * NZ;
                scalar_t *s_save_north = s_save_south + QF * NX * NZ;
                scalar_t *s_save_back = s_save_north + QF * NX * NZ;
                scalar_t *s_save_front = s_save_back + QF * NX * NY;

                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                if (tx == 0)
                {
                    s_save_west[0 * NY * NZ + tz * NY + ty] = pop[2];
                    s_save_west[1 * NY * NZ + tz * NY + ty] = pop[8];
                    s_save_west[2 * NY * NZ + tz * NY + ty] = pop[10];
                    s_save_west[3 * NY * NZ + tz * NY + ty] = pop[14];
                    s_save_west[4 * NY * NZ + tz * NY + ty] = pop[16];
                }
                if (tx == (NX - 1))
                {
                    s_save_east[0 * NY * NZ + tz * NY + ty] = pop[1];
                    s_save_east[1 * NY * NZ + tz * NY + ty] = pop[7];
                    s_save_east[2 * NY * NZ + tz * NY + ty] = pop[9];
                    s_save_east[3 * NY * NZ + tz * NY + ty] = pop[13];
                    s_save_east[4 * NY * NZ + tz * NY + ty] = pop[15];
                }
                if (ty == 0)
                {
                    s_save_south[0 * NX * NZ + tz * NX + tx] = pop[4];
                    s_save_south[1 * NX * NZ + tz * NX + tx] = pop[8];
                    s_save_south[2 * NX * NZ + tz * NX + tx] = pop[12];
                    s_save_south[3 * NX * NZ + tz * NX + tx] = pop[13];
                    s_save_south[4 * NX * NZ + tz * NX + tx] = pop[18];
                }
                if (ty == (NY - 1))
                {
                    s_save_north[0 * NX * NZ + tz * NX + tx] = pop[3];
                    s_save_north[1 * NX * NZ + tz * NX + tx] = pop[7];
                    s_save_north[2 * NX * NZ + tz * NX + tx] = pop[11];
                    s_save_north[3 * NX * NZ + tz * NX + tx] = pop[14];
                    s_save_north[4 * NX * NZ + tz * NX + tx] = pop[17];
                }
                if (tz == 0)
                {
                    s_save_back[0 * NX * NY + ty * NX + tx] = pop[6];
                    s_save_back[1 * NX * NY + ty * NX + tx] = pop[10];
                    s_save_back[2 * NX * NY + ty * NX + tx] = pop[12];
                    s_save_back[3 * NX * NY + ty * NX + tx] = pop[15];
                    s_save_back[4 * NX * NY + ty * NX + tx] = pop[17];
                }
                if (tz == (NZ - 1))
                {
                    s_save_front[0 * NX * NY + ty * NX + tx] = pop[5];
                    s_save_front[1 * NX * NY + ty * NX + tx] = pop[9];
                    s_save_front[2 * NX * NY + ty * NX + tx] = pop[11];
                    s_save_front[3 * NX * NY + ty * NX + tx] = pop[16];
                    s_save_front[4 * NX * NY + ty * NX + tx] = pop[18];
                }
            }

            template <class VSet>
            __device__ void halo_shared_to_global_save(const scalar_t *s_buffer) noexcept
            {
                // --- Buffer Partitioning (Unchanged) ---
                constexpr label_t QF = VSet::QF();
                constexpr label_t NX = block::nx();
                constexpr label_t NY = block::ny();
                constexpr label_t NZ = block::nz();

                const scalar_t *s_save_west = s_buffer;
                const scalar_t *s_save_east = s_save_west + QF * NY * NZ;
                const scalar_t *s_save_south = s_save_east + QF * NY * NZ;
                const scalar_t *s_save_north = s_save_south + QF * NX * NZ;
                const scalar_t *s_save_back = s_save_north + QF * NX * NZ;
                const scalar_t *s_save_front = s_save_back + QF * NX * NY;

                // --- Index Calculation (Unchanged) ---
                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;
                const label_t x = tx + blockDim.x * blockIdx.x;
                const label_t y = ty + blockDim.y * blockIdx.y;
                const label_t z = tz + blockDim.z * blockIdx.z;
                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;

                // --- Shared to Global Write (FIXED: Loop unrolled) ---
                if (West(x))
                {
                    gGhost().x0()[idxPopX<0, QF>(ty, tz, bx, by, bz)] = s_save_west[0 * NY * NZ + tz * NY + ty];
                    gGhost().x0()[idxPopX<1, QF>(ty, tz, bx, by, bz)] = s_save_west[1 * NY * NZ + tz * NY + ty];
                    gGhost().x0()[idxPopX<2, QF>(ty, tz, bx, by, bz)] = s_save_west[2 * NY * NZ + tz * NY + ty];
                    gGhost().x0()[idxPopX<3, QF>(ty, tz, bx, by, bz)] = s_save_west[3 * NY * NZ + tz * NY + ty];
                    gGhost().x0()[idxPopX<4, QF>(ty, tz, bx, by, bz)] = s_save_west[4 * NY * NZ + tz * NY + ty];
                }
                if (East(x))
                {
                    gGhost().x1()[idxPopX<0, QF>(ty, tz, bx, by, bz)] = s_save_east[0 * NY * NZ + tz * NY + ty];
                    gGhost().x1()[idxPopX<1, QF>(ty, tz, bx, by, bz)] = s_save_east[1 * NY * NZ + tz * NY + ty];
                    gGhost().x1()[idxPopX<2, QF>(ty, tz, bx, by, bz)] = s_save_east[2 * NY * NZ + tz * NY + ty];
                    gGhost().x1()[idxPopX<3, QF>(ty, tz, bx, by, bz)] = s_save_east[3 * NY * NZ + tz * NY + ty];
                    gGhost().x1()[idxPopX<4, QF>(ty, tz, bx, by, bz)] = s_save_east[4 * NY * NZ + tz * NY + ty];
                }
                if (South(y))
                {
                    gGhost().y0()[idxPopY<0, QF>(tx, tz, bx, by, bz)] = s_save_south[0 * NX * NZ + tz * NX + tx];
                    gGhost().y0()[idxPopY<1, QF>(tx, tz, bx, by, bz)] = s_save_south[1 * NX * NZ + tz * NX + tx];
                    gGhost().y0()[idxPopY<2, QF>(tx, tz, bx, by, bz)] = s_save_south[2 * NX * NZ + tz * NX + tx];
                    gGhost().y0()[idxPopY<3, QF>(tx, tz, bx, by, bz)] = s_save_south[3 * NX * NZ + tz * NX + tx];
                    gGhost().y0()[idxPopY<4, QF>(tx, tz, bx, by, bz)] = s_save_south[4 * NX * NZ + tz * NX + tx];
                }
                if (North(y))
                {
                    gGhost().y1()[idxPopY<0, QF>(tx, tz, bx, by, bz)] = s_save_north[0 * NX * NZ + tz * NX + tx];
                    gGhost().y1()[idxPopY<1, QF>(tx, tz, bx, by, bz)] = s_save_north[1 * NX * NZ + tz * NX + tx];
                    gGhost().y1()[idxPopY<2, QF>(tx, tz, bx, by, bz)] = s_save_north[2 * NX * NZ + tz * NX + tx];
                    gGhost().y1()[idxPopY<3, QF>(tx, tz, bx, by, bz)] = s_save_north[3 * NX * NZ + tz * NX + tx];
                    gGhost().y1()[idxPopY<4, QF>(tx, tz, bx, by, bz)] = s_save_north[4 * NX * NZ + tz * NX + tx];
                }
                if (Back(z))
                {
                    gGhost().z0()[idxPopZ<0, QF>(tx, ty, bx, by, bz)] = s_save_back[0 * NX * NY + ty * NX + tx];
                    gGhost().z0()[idxPopZ<1, QF>(tx, ty, bx, by, bz)] = s_save_back[1 * NX * NY + ty * NX + tx];
                    gGhost().z0()[idxPopZ<2, QF>(tx, ty, bx, by, bz)] = s_save_back[2 * NX * NY + ty * NX + tx];
                    gGhost().z0()[idxPopZ<3, QF>(tx, ty, bx, by, bz)] = s_save_back[3 * NX * NY + ty * NX + tx];
                    gGhost().z0()[idxPopZ<4, QF>(tx, ty, bx, by, bz)] = s_save_back[4 * NX * NY + ty * NX + tx];
                }
                if (Front(z))
                {
                    gGhost().z1()[idxPopZ<0, QF>(tx, ty, bx, by, bz)] = s_save_front[0 * NX * NY + ty * NX + tx];
                    gGhost().z1()[idxPopZ<1, QF>(tx, ty, bx, by, bz)] = s_save_front[1 * NX * NY + ty * NX + tx];
                    gGhost().z1()[idxPopZ<2, QF>(tx, ty, bx, by, bz)] = s_save_front[2 * NX * NY + ty * NX + tx];
                    gGhost().z1()[idxPopZ<3, QF>(tx, ty, bx, by, bz)] = s_save_front[3 * NX * NY + ty * NX + tx];
                    gGhost().z1()[idxPopZ<4, QF>(tx, ty, bx, by, bz)] = s_save_front[4 * NX * NY + ty * NX + tx];
                }
            }

        private:
            haloFace fGhost_;
            haloFace gGhost_;
            __device__ [[nodiscard]] inline bool West(const label_t x) const noexcept { return (threadIdx.x == 0 && x != 0); }
            __device__ [[nodiscard]] inline bool East(const label_t x) const noexcept { return (threadIdx.x == (block::nx() - 1) && x != (d_nx - 1)); }
            __device__ [[nodiscard]] inline bool South(const label_t y) const noexcept { return (threadIdx.y == 0 && y != 0); }
            __device__ [[nodiscard]] inline bool North(const label_t y) const noexcept { return (threadIdx.y == (block::ny() - 1) && y != (d_ny - 1)); }
            __device__ [[nodiscard]] inline bool Back(const label_t z) const noexcept { return (threadIdx.z == 0 && z != 0); }
            __device__ [[nodiscard]] inline bool Front(const label_t z) const noexcept { return (threadIdx.z == (block::nz() - 1) && z != (d_nz - 1)); }

            template <typename T>
            __host__ void interfaceSwap(T *ptrRestrict &pt1, T *ptrRestrict &pt2) const noexcept
            {
                T *ptrRestrict temp = pt1;
                pt1 = pt2;
                pt2 = temp;
            }
        };
    }
}

#endif // __MBLBM_HALO_CUH