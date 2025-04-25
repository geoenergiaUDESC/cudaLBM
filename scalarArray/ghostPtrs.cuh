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
                : x_0_(allocateGhostPtr_x(mesh)),
                  x_1_(allocateGhostPtr_x(mesh)),
                  y_0_(allocateGhostPtr_y(mesh)),
                  y_1_(allocateGhostPtr_y(mesh)),
                  z_0_(allocateGhostPtr_z(mesh)),
                  z_1_(allocateGhostPtr_z(mesh)) {};

            /**
             * @brief Default destructor
             **/
            ~ghostPtrs() noexcept {};

            /**
             * @brief Provides access to the underlying pointers
             * @return A reference to a unique pointer
             **/
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &x0() const noexcept
            {
                return x_0_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &x1() const noexcept
            {
                return x_1_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &y0() const noexcept
            {
                return y_0_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &y1() const noexcept
            {
                return y_1_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &z0() const noexcept
            {
                return z_0_;
            }
            [[nodiscard]] inline constexpr const scalarPtr_t<decltype(scalarDeleter)> &z1() const noexcept
            {
                return z_1_;
            }

        private:
            /**
             * @brief Pointers to the arrays on the device
             **/
            const scalarPtr_t<decltype(scalarDeleter)> x_0_;
            const scalarPtr_t<decltype(scalarDeleter)> x_1_;
            const scalarPtr_t<decltype(scalarDeleter)> y_0_;
            const scalarPtr_t<decltype(scalarDeleter)> y_1_;
            const scalarPtr_t<decltype(scalarDeleter)> z_0_;
            const scalarPtr_t<decltype(scalarDeleter)> z_1_;

            /**
             * @brief Allocate the pointer on the x-faces (i.e. the y-z plane)
             **/
            [[nodiscard]] scalarPtr_t<decltype(scalarDeleter)> allocateGhostPtr_x(const latticeMesh &mesh) const noexcept
            {
                const size_t NUMBER_GHOST_FACE_YZ = block::ny<size_t>() * block::nz<size_t>() * block::nxBlocks(mesh.nx()) * block::nyBlocks(mesh.ny()) * block::nzBlocks(mesh.nz());
                constexpr const VelSet velSet;
                const std::size_t count = sizeof(scalar_t) * velSet.QF() * NUMBER_GHOST_FACE_YZ;

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
             **/
            [[nodiscard]] scalarPtr_t<decltype(scalarDeleter)> allocateGhostPtr_y(const latticeMesh &mesh) const noexcept
            {
                const size_t NUMBER_GHOST_FACE_XZ = block::nx<size_t>() * block::nz<size_t>() * block::nxBlocks(mesh.nx()) * block::nyBlocks(mesh.ny()) * block::nzBlocks(mesh.nz());
                constexpr const VelSet velSet;
                const std::size_t count = sizeof(scalar_t) * velSet.QF() * NUMBER_GHOST_FACE_XZ;

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
             **/
            [[nodiscard]] scalarPtr_t<decltype(scalarDeleter)> allocateGhostPtr_z(const latticeMesh &mesh) const noexcept
            {
                // Get the memory size
                const size_t NUMBER_GHOST_FACE_XY = block::nx<size_t>() * block::ny<size_t>() * block::nxBlocks(mesh.nx()) * block::nyBlocks(mesh.ny()) * block::nzBlocks(mesh.nz());
                constexpr const VelSet velSet;
                const std::size_t count = sizeof(scalar_t) * velSet.QF() * NUMBER_GHOST_FACE_XY;

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
    }
}

#endif