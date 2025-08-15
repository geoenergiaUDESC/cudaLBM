/**
Filename: haloFace.cuh
Contents: A handling an individual device halo face
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_HALOFACE_CUH
#define __MBLBM_HALOFACE_CUH

namespace LBM
{
    namespace host
    {

    }

    namespace device
    {
        namespace haloFaces
        {
            /**
             * @brief Consteval functions used to distinguish between halo face normal directions
             * @return An unsigned integer corresponding to the correct direction
             **/
            [[nodiscard]] static inline consteval std::size_t x() noexcept { return 0; }
            [[nodiscard]] static inline consteval std::size_t y() noexcept { return 1; }
            [[nodiscard]] static inline consteval std::size_t z() noexcept { return 2; }
        }

        class haloFace
        {
        public:
            /**
             * @brief Constructs the halo faces from the moments and the mesh
             * @param fMom An std::vector containing the 10 interlaced moments
             * @param mesh The mesh
             **/
            [[nodiscard]] haloFace(const std::vector<scalar_t> &fMom, const host::latticeMesh &mesh) noexcept
                : x0_(device::allocateArray(initialise_pop<device::haloFaces::x(), 0>(fMom, mesh))),
                  x1_(device::allocateArray(initialise_pop<device::haloFaces::x(), 1>(fMom, mesh))),
                  y0_(device::allocateArray(initialise_pop<device::haloFaces::y(), 0>(fMom, mesh))),
                  y1_(device::allocateArray(initialise_pop<device::haloFaces::y(), 1>(fMom, mesh))),
                  z0_(device::allocateArray(initialise_pop<device::haloFaces::z(), 0>(fMom, mesh))),
                  z1_(device::allocateArray(initialise_pop<device::haloFaces::z(), 1>(fMom, mesh))) {};

            /**
             * @brief Destructor for the haloFace class
             **/
            ~haloFace() noexcept
            {
                cudaFree(x0_);
                cudaFree(x1_);
                cudaFree(y0_);
                cudaFree(y1_);
                cudaFree(z0_);
                cudaFree(z1_);
            }

            /**
             * @brief Provides read-only access to the halo faces
             * @return A const-qualified pointer to the halo faces
             **/
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

            /**
             * @brief Provides mutable access to the halo faces
             * @return A pointer to the halo faces
             **/
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

            /**
             * @brief Provides a reference to the halo faces pointer
             * @return A reference to the halo faces pointer
             * @note These methods are used to perform the pointer swap and should not be used in other contexts
             **/
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &x0Ref() noexcept
            {
                return x0_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &x1Ref() noexcept
            {
                return x1_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &y0Ref() noexcept
            {
                return y0_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &y1Ref() noexcept
            {
                return y1_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &z0Ref() noexcept
            {
                return z0_;
            }
            [[nodiscard]] inline constexpr scalar_t *ptrRestrict &z1Ref() noexcept
            {
                return z1_;
            }

        private:
            /**
             * @brief Halo faces pointers
             **/
            scalar_t *ptrRestrict x0_;
            scalar_t *ptrRestrict x1_;
            scalar_t *ptrRestrict y0_;
            scalar_t *ptrRestrict y1_;
            scalar_t *ptrRestrict z0_;
            scalar_t *ptrRestrict z1_;

            /**
             * @brief Calculates the number of faces on the side of a block depending upon the direction (x, y or z)
             * @param mesh The mesh
             **/
            template <const std::size_t faceIndex>
            __host__ [[nodiscard]] static inline constexpr std::size_t nFaces(const host::latticeMesh &mesh) noexcept
            {
                if constexpr (faceIndex == device::haloFaces::x())
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nx()) * VelocitySet::D3Q19::QF();
                }
                if constexpr (faceIndex == device::haloFaces::y())
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::ny()) * VelocitySet::D3Q19::QF();
                }
                if constexpr (faceIndex == device::haloFaces::z())
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nz()) * VelocitySet::D3Q19::QF();
                }

                return 0;
            }

            /**
             * @brief Initialises the population density halo on the device from the moments and the mesh
             * @param fMom The 10 interlaced moments
             * @param mesh The mesh
             **/
            template <const std::size_t faceIndex, const std::size_t side>
            __host__ [[nodiscard]] const std::vector<scalar_t> initialise_pop(const std::vector<scalar_t> &fMom, const host::latticeMesh &mesh) const noexcept
            {
                std::vector<scalar_t> face(nFaces<faceIndex>(mesh), 0);

                const label_t nBlockx = mesh.nxBlocks();
                const label_t nBlocky = mesh.nyBlocks();
                const label_t nBlockz = mesh.nzBlocks();

                // Loop over all blocks and threads
                for (label_t bz = 0; bz < nBlockz; ++bz)
                {
                    for (label_t by = 0; by < nBlocky; ++by)
                    {
                        for (label_t bx = 0; bx < nBlockx; ++bx)
                        {
                            for (label_t tz = 0; tz < block::nz(); ++tz)
                            {
                                for (label_t ty = 0; ty < block::ny(); ++ty)
                                {
                                    for (label_t tx = 0; tx < block::nx(); ++tx)
                                    {

                                        // Skip out-of-bounds elements (equivalent to GPU version)
                                        if (tx >= mesh.nx() || ty >= mesh.ny() || tz >= mesh.nz())
                                        {
                                            continue;
                                        }

                                        const label_t base = host::idxMom<0>(tx, ty, tz, bx, by, bz, nBlockx, nBlocky);

                                        // Contiguous moment access
                                        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::host_reconstruct(
                                            {rho0() + fMom[base + index::rho()],
                                             fMom[base + index::u()],
                                             fMom[base + index::v()],
                                             fMom[base + index::w()],
                                             fMom[base + index::xx()],
                                             fMom[base + index::xy()],
                                             fMom[base + index::xz()],
                                             fMom[base + index::yy()],
                                             fMom[base + index::yz()],
                                             fMom[base + index::zz()]});

                                        // Handle ghost cells (equivalent to threadIdx.x/y/z checks)
                                        if constexpr (faceIndex == device::haloFaces::x())
                                        {
                                            if constexpr (side == 0)
                                            {
                                                if (tx == 0)
                                                { // w
                                                    face[host::idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[2];
                                                    face[host::idxPopX<1, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[8];
                                                    face[host::idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[10];
                                                    face[host::idxPopX<3, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[14];
                                                    face[host::idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[16];
                                                }
                                            }
                                            if constexpr (side == 1)
                                            {
                                                if (tx == (block::nx() - 1))
                                                {
                                                    face[host::idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[1];
                                                    face[host::idxPopX<1, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[7];
                                                    face[host::idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[9];
                                                    face[host::idxPopX<3, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[13];
                                                    face[host::idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz, nBlockx, nBlocky)] = pop[15];
                                                }
                                            }
                                        }

                                        if constexpr (faceIndex == device::haloFaces::y())
                                        {
                                            if constexpr (side == 0)
                                            {
                                                if (ty == 0)
                                                { // s
                                                    face[host::idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[4];
                                                    face[host::idxPopY<1, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[8];
                                                    face[host::idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[12];
                                                    face[host::idxPopY<3, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[13];
                                                    face[host::idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[18];
                                                }
                                            }
                                            if constexpr (side == 1)
                                            {
                                                if (ty == (block::ny() - 1))
                                                {
                                                    face[host::idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[3];
                                                    face[host::idxPopY<1, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[7];
                                                    face[host::idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[11];
                                                    face[host::idxPopY<3, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[14];
                                                    face[host::idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz, nBlockx, nBlocky)] = pop[17];
                                                }
                                            }
                                        }

                                        if constexpr (faceIndex == device::haloFaces::z())
                                        {
                                            if constexpr (side == 0)
                                            {
                                                if (tz == 0)
                                                { // b
                                                    face[host::idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[6];
                                                    face[host::idxPopZ<1, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[10];
                                                    face[host::idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[12];
                                                    face[host::idxPopZ<3, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[15];
                                                    face[host::idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[17];
                                                }
                                            }
                                            if constexpr (side == 1)
                                            {
                                                if (tz == (block::nz() - 1))
                                                {
                                                    face[host::idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[5];
                                                    face[host::idxPopZ<1, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[9];
                                                    face[host::idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[11];
                                                    face[host::idxPopZ<3, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[16];
                                                    face[host::idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz, nBlockx, nBlocky)] = pop[18];
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
        };
    }
}

#endif