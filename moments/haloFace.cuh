/**
Filename: haloFace.cuh
Contents: A handling an individual device halo face
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_HALOFACE_CUH
#define __MBLBM_HALOFACE_CUH

namespace mbLBM
{
    namespace host
    {

    }

    namespace device
    {
        class haloFace
        {
        public:
            [[nodiscard]] haloFace(const label_t nx, const label_t ny, const label_t nz) noexcept
                : x0_(device::allocate<scalar_t>(((nx * ny * nz) / BLOCK_NX) * QF)),
                  x1_(device::allocate<scalar_t>(((nx * ny * nz) / BLOCK_NX) * QF)),
                  y0_(device::allocate<scalar_t>(((nx * ny * nz) / BLOCK_NY) * QF)),
                  y1_(device::allocate<scalar_t>(((nx * ny * nz) / BLOCK_NY) * QF)),
                  z0_(device::allocate<scalar_t>(((nx * ny * nz) / BLOCK_NZ) * QF)),
                  z1_(device::allocate<scalar_t>(((nx * ny * nz) / BLOCK_NZ) * QF)) {};

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