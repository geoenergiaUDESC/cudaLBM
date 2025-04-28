/**
Filename: ghostPtrs.cuh
Contents: A handling the ghost interface pointers on the device
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_GHOSTPTRS_CUH
#define __MBLBM_GHOSTPTRS_CUH

namespace mbLBM
{
    namespace device
    {
        /**
         * @brief Struct holding the number of lattice points on each face of a CUDA block
         **/
        struct nGhostFace
        {
        public:
            /**
             * @brief Construct from a latticeMesh object
             * @return An nGhostFace struct with xy, xz and yz defined by the mesh
             **/
            [[nodiscard]] inline constexpr nGhostFace(const latticeMesh &mesh) noexcept
                : xy(block::nx<std::size_t>() * block::ny<std::size_t>() * block::nBlocks<std::size_t>(mesh)),
                  xz(block::nx<std::size_t>() * block::nz<std::size_t>() * block::nBlocks<std::size_t>(mesh)),
                  yz(block::ny<std::size_t>() * block::nz<std::size_t>() * block::nBlocks<std::size_t>(mesh)) {};

            const std::size_t xy;
            const std::size_t xz;
            const std::size_t yz;
        };

        template <class VelSet>
        class ghostPtrs
        {
        public:
            /**
             * @brief Constructs a list of 6 unique pointers from a latticeMesh object
             * @return A ghostPtrs object constructed from the mesh
             * @param mesh The mesh used to define the amount of memory to allocate to the pointers
             **/
            [[nodiscard]] ghostPtrs(const latticeMesh &mesh) noexcept
                : nGhostFaces_(mesh),
                  x0_(allocateGhostPtr_x(nGhostFaces_)),
                  x1_(allocateGhostPtr_x(nGhostFaces_)),
                  y0_(allocateGhostPtr_y(nGhostFaces_)),
                  y1_(allocateGhostPtr_y(nGhostFaces_)),
                  z0_(allocateGhostPtr_z(nGhostFaces_)),
                  z1_(allocateGhostPtr_z(nGhostFaces_)) {};

            /**
             * @brief Default destructor
             **/
            ~ghostPtrs() noexcept {};

            /**
             * @brief Provides access to the underlying pointers
             * @return An immutable reference to a unique pointer
             **/
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &x0() const noexcept
            {
                return x0_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &x1() const noexcept
            {
                return x1_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &y0() const noexcept
            {
                return y0_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &y1() const noexcept
            {
                return y1_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &z0() const noexcept
            {
                return z0_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &z1() const noexcept
            {
                return z1_;
            }

            /**
             * @brief Returns the number of lattice points on each block face
             * @return Number of lattice points on a block face as a std::size_t
             **/
            [[nodiscard]] inline constexpr std::size_t n_xy() const noexcept
            {
                return nGhostFaces_.xy;
            }
            [[nodiscard]] inline constexpr std::size_t n_xz() const noexcept
            {
                return nGhostFaces_.xz;
            }
            [[nodiscard]] inline constexpr std::size_t n_yz() const noexcept
            {
                return nGhostFaces_.yz;
            }

        private:
            /**
             * @brief Number of lattice points on each face of a CUDA block
             **/
            const nGhostFace nGhostFaces_;

            /**
             * @brief Pointers to the arrays on the device
             **/
            const scalarPtr_t<decltype(scalarDeleter)> x0_;
            const scalarPtr_t<decltype(scalarDeleter)> x1_;
            const scalarPtr_t<decltype(scalarDeleter)> y0_;
            const scalarPtr_t<decltype(scalarDeleter)> y1_;
            const scalarPtr_t<decltype(scalarDeleter)> z0_;
            const scalarPtr_t<decltype(scalarDeleter)> z1_;

            /**
             * @brief Allocate the pointer on the x-faces (i.e. the y-z plane)
             * * @return A unique pointer to a block of memory of type scalar_t containing n_yz lattice points
             * @param nGhostFaces Struct containing the number of lattice points on each block face
             **/
            [[nodiscard]] scalarPtr_t<decltype(scalarDeleter)> allocateGhostPtr_x(const nGhostFace &nGhostFaces) const noexcept
            {
                constexpr const VelSet velSet;
                const std::size_t count = sizeof(scalar_t) * velSet.QF() * nGhostFaces.yz;

                // Allocate the pointer
                scalarPtr_t<decltype(scalarDeleter)> ptr(deviceMalloc<scalar_t>(count), scalarDeleter);

                // Set the memory to all zero values
                const cudaError_t i = cudaMemset(ptr.get(), 0, count);

                if (i != cudaSuccess)
                {
                    exceptions::program_exit(i, "Unable to allocate ghost pointer");
                }

                return ptr;
            }

            /**
             * @brief Allocate the pointer on the y-faces (i.e. the x-z plane)
             * * @return A unique pointer to a block of memory of type scalar_t containing n_xz lattice points
             * @param nGhostFaces Struct containing the number of lattice points on each block face
             **/
            [[nodiscard]] scalarPtr_t<decltype(scalarDeleter)> allocateGhostPtr_y(const nGhostFace &nGhostFaces) const noexcept
            {
                constexpr const VelSet velSet;
                const std::size_t count = sizeof(scalar_t) * velSet.QF() * nGhostFaces.xz;

                // Allocate the pointer
                scalarPtr_t<decltype(scalarDeleter)> ptr(deviceMalloc<scalar_t>(count), scalarDeleter);

                // Set the memory to all zero values
                const cudaError_t i = cudaMemset(ptr.get(), 0, count);

                if (i != cudaSuccess)
                {
                    exceptions::program_exit(i, "Unable to allocate ghost pointer");
                }

                return ptr;
            }

            /**
             * @brief Allocate the pointer on the z-faces (i.e. the x-y plane)
             * @return A unique pointer to a block of memory of type scalar_t containing n_xy lattice points
             * @param nGhostFaces Struct containing the number of lattice points on each block face
             **/
            [[nodiscard]] scalarPtr_t<decltype(scalarDeleter)> allocateGhostPtr_z(const nGhostFace &nGhostFaces) const noexcept
            {
                // Get the memory size
                constexpr const VelSet velSet;
                const std::size_t count = sizeof(scalar_t) * velSet.QF() * nGhostFaces.xy;

                // Allocate the pointer
                scalarPtr_t<decltype(scalarDeleter)> ptr(deviceMalloc<scalar_t>(count), scalarDeleter);

                // Set the memory to all zero values
                const cudaError_t i = cudaMemset(ptr.get(), 0, count);

                if (i != cudaSuccess)
                {
                    exceptions::program_exit(i, "Unable to allocate ghost pointer");
                }

                return ptr;
            }
        };

        // template <class VelSet>
        // __host__ inline void interfaceCudaMemcpy(ghostPtrs<VelSet> &dest, const ghostPtrs<VelSet> &src, const cudaMemcpyKind kind) noexcept
        // {
        //     // Copy the x interfaces
        //     cudaMemcpy(dest.x0().get(), src.x0().get(), src.nyz(), kind);
        //     cudaMemcpy(dest.x1().get(), src.x1().get(), src.nyz(), kind);

        //     // Copy the y interfaces
        //     cudaMemcpy(dest.y0().get(), src.y0().get(), src.nxz(), kind);
        //     cudaMemcpy(dest.y1().get(), src.y1().get(), src.nxz(), kind);

        //     // Copy the z interfaces
        //     cudaMemcpy(dest.z0().get(), src.z0().get(), src.nxy(), kind);
        //     cudaMemcpy(dest.z1().get(), src.z1().get(), src.nxy(), kind);
        // }
    }
}

#endif