/**
Filename: ghostPtrs.cuh
Contents: A handling the ghost interface pointers on the device
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_GHOSTPTRS_CUH
#define __MBLBM_GHOSTPTRS_CUH

namespace mbLBM
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
        [[nodiscard]] inline constexpr nGhostFace(const host::latticeMesh &mesh) noexcept
            : xy(block::nx<std::size_t>() * block::ny<std::size_t>() * block::nBlocks<std::size_t>(mesh)),
              xz(block::nx<std::size_t>() * block::nz<std::size_t>() * block::nBlocks<std::size_t>(mesh)),
              yz(block::ny<std::size_t>() * block::nz<std::size_t>() * block::nBlocks<std::size_t>(mesh)) {};

        const std::size_t xy;
        const std::size_t xz;
        const std::size_t yz;
    };

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
            [[nodiscard]] ghostPtrs(const host::latticeMesh &mesh) noexcept
                : nGhostFaces_(mesh),
                  x0_(scalarArray(nGhostFaces_.yz, 0)),
                  x1_(scalarArray(nGhostFaces_.yz, 0)),
                  y0_(scalarArray(nGhostFaces_.xz, 0)),
                  y1_(scalarArray(nGhostFaces_.xz, 0)),
                  z0_(scalarArray(nGhostFaces_.xy, 0)),
                  z1_(scalarArray(nGhostFaces_.xy, 0)) {};

            /**
             * @brief Default destructor
             **/
            ~ghostPtrs() noexcept
            {
#ifdef VERBOSE
                std::cout << "Freeing ghostPtrs object" << std::endl;
#endif
            };

            /**
             * @brief Provides access to the underlying pointers
             * @return An immutable reference to a unique pointer
             **/
            __device__ [[nodiscard]] inline constexpr const scalar_t *x0() const noexcept
            {
                return x0_.ptr();
            }
            __device__ [[nodiscard]] inline constexpr const scalar_t *x1() const noexcept
            {
                return x1_.ptr();
            }
            __device__ [[nodiscard]] inline constexpr const scalar_t *y0() const noexcept
            {
                return y0_.ptr();
            }
            __device__ [[nodiscard]] inline constexpr const scalar_t *y1() const noexcept
            {
                return y1_.ptr();
            }
            __device__ [[nodiscard]] inline constexpr const scalar_t *z0() const noexcept
            {
                return z0_.ptr();
            }
            __device__ [[nodiscard]] inline constexpr const scalar_t *z1() const noexcept
            {
                return z1_.ptr();
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
             * @brief The underlying device scalar arrays
             **/
            const scalarArray x0_;
            const scalarArray x1_;
            const scalarArray y0_;
            const scalarArray y1_;
            const scalarArray z0_;
            const scalarArray z1_;
        };
    }
}

#endif