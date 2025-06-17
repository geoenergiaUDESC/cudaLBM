/**
Filename: haloFace.cuh
Contents: A handling an individual device halo face
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_HALOFACE_CUH
#define __MBLBM_HALOFACE_CUH

#include "../velocitySet/velocitySet.cuh"

namespace mbLBM
{
    namespace host
    {

    }

    [[nodiscard]] inline consteval std::size_t x_index() noexcept { return 0; }
    [[nodiscard]] inline consteval std::size_t y_index() noexcept { return 1; }
    [[nodiscard]] inline consteval std::size_t z_index() noexcept { return 2; }

    template <const std::size_t index>
    [[nodiscard]] inline constexpr std::size_t nFaces(
        const std::size_t nx,
        const std::size_t ny,
        const std::size_t nz) noexcept
    {
        if constexpr (index == x_index())
        {
            return ((nx * ny * nz) / block::nx()) * QF;
        }
        if constexpr (index == y_index())
        {
            return ((nx * ny * nz) / block::ny()) * QF;
        }
        if constexpr (index == z_index())
        {
            return ((nx * ny * nz) / block::nz()) * QF;
        }

        return 0;
    }

    template <const std::size_t index, const std::size_t side>
    [[nodiscard]] const std::vector<scalar_t> cpuInitialization_pop(
        const std::size_t nx,
        const std::size_t ny,
        const std::size_t nz,
        const std::vector<scalar_t> &fMom)
    {
        std::vector<scalar_t> face(nFaces<index>(nx, ny, nz), 0);

        // Loop over all blocks and threads
        for (label_t bz = 0; bz < NUM_BLOCK_Z; ++bz)
        {
            for (label_t by = 0; by < NUM_BLOCK_Y; ++by)
            {
                for (label_t bx = 0; bx < NUM_BLOCK_X; ++bx)
                {
                    for (label_t tz = 0; tz < block::nz(); ++tz)
                    {
                        for (label_t ty = 0; ty < block::ny(); ++ty)
                        {
                            for (label_t tx = 0; tx < block::nx(); ++tx)
                            {
                                // Skip out-of-bounds elements (equivalent to GPU version)
                                if (tx >= NX || ty >= NY || tz >= NZ)
                                {
                                    continue;
                                }

                                // zeroth moment
                                const scalar_t rhoVar = RHO_0 + fMom[idxMom<M_RHO_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t ux_t30 = fMom[idxMom<M_UX_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t uy_t30 = fMom[idxMom<M_UY_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t uz_t30 = fMom[idxMom<M_UZ_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t m_xx_t45 = fMom[idxMom<M_MXX_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t m_xy_t90 = fMom[idxMom<M_MXY_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t m_xz_t90 = fMom[idxMom<M_MXZ_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t m_yy_t45 = fMom[idxMom<M_MYY_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t m_yz_t90 = fMom[idxMom<M_MYZ_INDEX>(tx, ty, tz, bx, by, bz)];
                                const scalar_t m_zz_t45 = fMom[idxMom<M_MZZ_INDEX>(tx, ty, tz, bx, by, bz)];

                                // scalar_t pop[Q];
                                const std::array<scalar_t, Q> pop = VelocitySet::D3Q19::reconstruct(
                                    rhoVar,
                                    ux_t30, uy_t30, uz_t30,
                                    m_xx_t45, m_xy_t90, m_xz_t90,
                                    m_yy_t45, m_yz_t90, m_zz_t45);

                                // Handle ghost cells (equivalent to threadIdx.x/y/z checks)
                                if constexpr (index == x_index())
                                {
                                    if constexpr (side == 0)
                                    {
                                        if (tx == 0)
                                        { // w
                                            face[idxPopX<0>(ty, tz, bx, by, bz)] = pop[2];
                                            face[idxPopX<1>(ty, tz, bx, by, bz)] = pop[8];
                                            face[idxPopX<2>(ty, tz, bx, by, bz)] = pop[10];
                                            face[idxPopX<3>(ty, tz, bx, by, bz)] = pop[14];
                                            face[idxPopX<4>(ty, tz, bx, by, bz)] = pop[16];
                                        }
                                    }
                                    if constexpr (side == 1)
                                    {
                                        if (tx == (block::nx() - 1))
                                        {
                                            face[idxPopX<0>(ty, tz, bx, by, bz)] = pop[1];
                                            face[idxPopX<1>(ty, tz, bx, by, bz)] = pop[7];
                                            face[idxPopX<2>(ty, tz, bx, by, bz)] = pop[9];
                                            face[idxPopX<3>(ty, tz, bx, by, bz)] = pop[13];
                                            face[idxPopX<4>(ty, tz, bx, by, bz)] = pop[15];
                                        }
                                    }
                                }

                                if (index == y_index())
                                {
                                    if constexpr (side == 0)
                                    {
                                        if (ty == 0)
                                        { // s
                                            face[idxPopY<0>(tx, tz, bx, by, bz)] = pop[4];
                                            face[idxPopY<1>(tx, tz, bx, by, bz)] = pop[8];
                                            face[idxPopY<2>(tx, tz, bx, by, bz)] = pop[12];
                                            face[idxPopY<3>(tx, tz, bx, by, bz)] = pop[13];
                                            face[idxPopY<4>(tx, tz, bx, by, bz)] = pop[18];
                                        }
                                    }
                                    if constexpr (side == 1)
                                    {
                                        if (ty == (block::ny() - 1))
                                        {
                                            face[idxPopY<0>(tx, tz, bx, by, bz)] = pop[3];
                                            face[idxPopY<1>(tx, tz, bx, by, bz)] = pop[7];
                                            face[idxPopY<2>(tx, tz, bx, by, bz)] = pop[11];
                                            face[idxPopY<3>(tx, tz, bx, by, bz)] = pop[14];
                                            face[idxPopY<4>(tx, tz, bx, by, bz)] = pop[17];
                                        }
                                    }
                                }

                                if (index == z_index())
                                {
                                    if constexpr (side == 0)
                                    {
                                        if (tz == 0)
                                        { // b
                                            face[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[6];
                                            face[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[10];
                                            face[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[12];
                                            face[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[15];
                                            face[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[17];
                                        }
                                    }
                                    if constexpr (side == 1)
                                    {
                                        if (tz == (block::nz() - 1))
                                        {
                                            face[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[5];
                                            face[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[9];
                                            face[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[11];
                                            face[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[16];
                                            face[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[18];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return face;
    }

    namespace device
    {
        class haloFace
        {
        public:
            // [[nodiscard]] haloFace(const label_t nx, const label_t ny, const label_t nz) noexcept
            //     : x0_(device::allocate<scalar_t>(((nx * ny * nz) / block::nx()) * QF)),
            //       x1_(device::allocate<scalar_t>(((nx * ny * nz) / block::nx()) * QF)),
            //       y0_(device::allocate<scalar_t>(((nx * ny * nz) / block::ny()) * QF)),
            //       y1_(device::allocate<scalar_t>(((nx * ny * nz) / block::ny()) * QF)),
            //       z0_(device::allocate<scalar_t>(((nx * ny * nz) / block::nz()) * QF)),
            //       z1_(device::allocate<scalar_t>(((nx * ny * nz) / block::nz()) * QF)) {};

            [[nodiscard]] haloFace(
                const std::vector<scalar_t> &fMom,
                const label_t nx, const label_t ny, const label_t nz) noexcept
                : x0_(device::allocateArray(cpuInitialization_pop<x_index(), 0>(nx, ny, nz, fMom))),
                  x1_(device::allocateArray(cpuInitialization_pop<x_index(), 1>(nx, ny, nz, fMom))),
                  y0_(device::allocateArray(cpuInitialization_pop<y_index(), 0>(nx, ny, nz, fMom))),
                  y1_(device::allocateArray(cpuInitialization_pop<y_index(), 1>(nx, ny, nz, fMom))),
                  z0_(device::allocateArray(cpuInitialization_pop<z_index(), 0>(nx, ny, nz, fMom))),
                  z1_(device::allocateArray(cpuInitialization_pop<z_index(), 1>(nx, ny, nz, fMom))) {};

            ~haloFace() noexcept
            {
                // cudaFree(x0_);
                // cudaFree(x1_);
                // cudaFree(y0_);
                // cudaFree(y1_);
                // cudaFree(z0_);
                // cudaFree(z1_);
            }

            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x0() const noexcept
            {
                return x0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x1() const noexcept
            {
                return x1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y0() const noexcept
            {
                return y0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y1() const noexcept
            {
                return y1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z0() const noexcept
            {
                return z0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z1() const noexcept
            {
                return z1_;
            }

            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x0() noexcept
            {
                return x0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x1() noexcept
            {
                return x1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y0() noexcept
            {
                return y0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y1() noexcept
            {
                return y1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z0() noexcept
            {
                return z0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z1() noexcept
            {
                return z1_;
            }

            [[nodiscard]] inline constexpr scalar_t *&x0Ref() noexcept
            {
                return x0_;
            }
            [[nodiscard]] inline constexpr scalar_t *&x1Ref() noexcept
            {
                return x1_;
            }
            [[nodiscard]] inline constexpr scalar_t *&y0Ref() noexcept
            {
                return y0_;
            }
            [[nodiscard]] inline constexpr scalar_t *&y1Ref() noexcept
            {
                return y1_;
            }
            [[nodiscard]] inline constexpr scalar_t *&z0Ref() noexcept
            {
                return z0_;
            }
            [[nodiscard]] inline constexpr scalar_t *&z1Ref() noexcept
            {
                return z1_;
            }

        private:
            scalar_t *x0_;
            scalar_t *x1_;
            scalar_t *y0_;
            scalar_t *y1_;
            scalar_t *z0_;
            scalar_t *z1_;
        };
    }
}

#endif