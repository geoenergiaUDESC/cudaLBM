/**
Filename: moments.cuh
Contents: A class containing the arrays of the moment variables
**/

#ifndef __MBLBM_MOMENTS_CUH
#define __MBLBM_MOMENTS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "../latticeMesh/latticeMesh.cuh"
#include "../velocitySet/velocitySet.cuh"
#include "../globalFunctions.cuh"
#include "../array/array.cuh"

namespace mbLBM
{
    namespace host
    {
        class moments
        {
        public:
            /**
             * @brief Constructs the moments from a number of lattice points
             * @return An object holding the moments on the host
             * @param mesh The undecomposed solution mesh
             * @param readType The type of constructor
             * @note If readType is NO_READ, this constructor zero-initialises everything
             * @note If readType is MUST_READ, this constructor attempts to read the variables from input files
             **/
            [[nodiscard]] moments(const latticeMesh &mesh, const ctorType::type readType) noexcept
                : mesh_(mesh),
                  rho_(scalarArray(mesh, "rho", readType)),
                  u_(scalarArray(mesh, "u", readType)),
                  v_(scalarArray(mesh, "v", readType)),
                  w_(scalarArray(mesh, "w", readType)),
                  m_xx_(scalarArray(mesh, "m_xx", readType)),
                  m_xy_(scalarArray(mesh, "m_xy", readType)),
                  m_xz_(scalarArray(mesh, "m_xz", readType)),
                  m_yy_(scalarArray(mesh, "m_yy", readType)),
                  m_yz_(scalarArray(mesh, "m_yz", readType)),
                  m_zz_(scalarArray(mesh, "m_zz", readType)) {};

            /**
             * @brief Destructor for the host moments class
             **/
            ~moments() {};

            /**
             * @brief Returns the number of lattices in the x, y and z directions
             **/
            [[nodiscard]] inline constexpr const latticeMesh &mesh() const noexcept
            {
                return mesh_;
            }
            [[nodiscard]] inline constexpr label_t nx() const noexcept
            {
                return mesh_.nx();
            }
            [[nodiscard]] inline constexpr label_t ny() const noexcept
            {
                return mesh_.ny();
            }
            [[nodiscard]] inline constexpr label_t nz() const noexcept
            {
                return mesh_.nz();
            }
            [[nodiscard]] inline constexpr label_t nPoints() const noexcept
            {
                return mesh_.nPoints();
            }

            /**
             * @brief Returns the number of CUDA in the x, y and z directions
             **/
            [[nodiscard]] inline constexpr label_t nxBlocks() const noexcept
            {
                return mesh_.nxBlocks();
            }
            [[nodiscard]] inline constexpr label_t nyBlocks() const noexcept
            {
                return mesh_.nyBlocks();
            }
            [[nodiscard]] inline constexpr label_t nzBlocks() const noexcept
            {
                return mesh_.nzBlocks();
            }

            /**
             * @brief Provides immutable access to the moments
             * @return An immutable access to a scalarArray object
             **/
            [[nodiscard]] inline constexpr scalarArray const &rho() const noexcept
            {
                return rho_;
            }
            [[nodiscard]] inline constexpr scalarArray const &u() const noexcept
            {
                return u_;
            }
            [[nodiscard]] inline constexpr scalarArray const &v() const noexcept
            {
                return v_;
            }
            [[nodiscard]] inline constexpr scalarArray const &w() const noexcept
            {
                return w_;
            }
            [[nodiscard]] inline constexpr scalarArray const &m_xx() const noexcept
            {
                return m_xx_;
            }
            [[nodiscard]] inline constexpr scalarArray const &m_xy() const noexcept
            {
                return m_xy_;
            }
            [[nodiscard]] inline constexpr scalarArray const &m_xz() const noexcept
            {
                return m_xz_;
            }
            [[nodiscard]] inline constexpr scalarArray const &m_yy() const noexcept
            {
                return m_yy_;
            }
            [[nodiscard]] inline constexpr scalarArray const &m_yz() const noexcept
            {
                return m_yz_;
            }
            [[nodiscard]] inline constexpr scalarArray const &m_zz() const noexcept
            {
                return m_zz_;
            }

            /**
             * @brief Writes the moment variables to a file
             * @param time The solution time step
             **/
            void writeFile(const label_t time) const noexcept
            {
                rho_.saveFile(time);
                u_.saveFile(time);
                v_.saveFile(time);
                w_.saveFile(time);
                m_xx_.saveFile(time);
                m_xy_.saveFile(time);
                m_xz_.saveFile(time);
                m_yy_.saveFile(time);
                m_yz_.saveFile(time);
                m_zz_.saveFile(time);
            }

        private:
            /**
             * @brief Immutable reference to the mesh
             **/
            const latticeMesh &mesh_;

            /**
             * @brief The moment variables
             **/
            const scalarArray rho_;
            const scalarArray u_;
            const scalarArray v_;
            const scalarArray w_;
            const scalarArray m_xx_;
            const scalarArray m_xy_;
            const scalarArray m_xz_;
            const scalarArray m_yy_;
            const scalarArray m_yz_;
            const scalarArray m_zz_;
        };
    }

    namespace device
    {
        class moments
        {
        public:
            /**
             * @brief Constructs the device moments from a copy of the host moments
             * @return An object holding the moments on the device
             * @param deviceID The unique ID of the device onto which the moments are to be copied
             * @param hostMoments The moments as they exist on the host
             **/
            [[nodiscard]] moments(const deviceIndex_t deviceID, const mbLBM::host::moments &hostMoments)
                : ID_(deviceID),
                  err_(cudaSetDevice(ID_)),
                  rho_(scalarArray(hostMoments.rho())),
                  u_(scalarArray(hostMoments.u())),
                  v_(scalarArray(hostMoments.v())),
                  w_(scalarArray(hostMoments.w())),
                  m_xx_(scalarArray(hostMoments.m_xx())),
                  m_xy_(scalarArray(hostMoments.m_xy())),
                  m_xz_(scalarArray(hostMoments.m_xz())),
                  m_yy_(scalarArray(hostMoments.m_yy())),
                  m_yz_(scalarArray(hostMoments.m_yz())),
                  m_zz_(scalarArray(hostMoments.m_zz()))
            {
#ifdef VERBOSE
                std::cout << "Allocated partitioned moments on device number " << ID_ << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << hostMoments.nx() << ";" << std::endl;
                std::cout << "    ny = " << hostMoments.ny() << ";" << std::endl;
                std::cout << "    nz = " << hostMoments.nz() << ";" << std::endl;
                std::cout << "}" << std::endl;
                std::cout << std::endl;
#endif
                checkCudaErrors(cudaDeviceSynchronize());
            };

            /**
             * @brief Destructor for the device moments class
             **/
            ~moments() {};

            /**
             * @brief Provides access to the underlying pointers
             * @return A reference to a unique pointer
             **/
            __host__ __device__ [[nodiscard]] inline const scalar_t *rho() const noexcept
            {
                return rho_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *u() const noexcept
            {
                return u_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *v() const noexcept
            {
                return v_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *w() const noexcept
            {
                return w_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *m_xx() const noexcept
            {
                return m_xx_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *m_xy() const noexcept
            {
                return m_xy_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *m_xz() const noexcept
            {
                return m_xz_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *m_yy() const noexcept
            {
                return m_yy_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *m_yz() const noexcept
            {
                return m_yz_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline const scalar_t *m_zz() const noexcept
            {
                return m_zz_.ptr();
            }

            __host__ __device__ [[nodiscard]] inline scalar_t *rho() noexcept
            {
                return rho_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *u() noexcept
            {
                return u_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *v() noexcept
            {
                return v_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *w() noexcept
            {
                return w_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *m_xx() noexcept
            {
                return m_xx_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *m_xy() noexcept
            {
                return m_xy_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *m_xz() noexcept
            {
                return m_xz_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *m_yy() noexcept
            {
                return m_yy_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *m_yz() noexcept
            {
                return m_yz_.ptr();
            }
            __host__ __device__ [[nodiscard]] inline scalar_t *m_zz() noexcept
            {
                return m_zz_.ptr();
            }

        private:
            /**
             * @brief The device ID
             **/
            const deviceIndex_t ID_;

            /**
             * @brief The error code returned from setting the device
             **/
            const cudaError_t err_;

            /**
             * @brief The moment variables
             **/
            scalarArray rho_;
            scalarArray u_;
            scalarArray v_;
            scalarArray w_;
            scalarArray m_xx_;
            scalarArray m_xy_;
            scalarArray m_xz_;
            scalarArray m_yy_;
            scalarArray m_yz_;
            scalarArray m_zz_;
        };
    }
}

#include "ghostInterface.cuh"

#endif