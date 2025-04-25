/**
Filename: moments.cuh
Contents: A class containing the arrays of the moment variables
**/

#ifndef __MBLBM_MOMENTS_CUH
#define __MBLBM_MOMENTS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "velocitySet/velocitySet.cuh"
#include "globalFunctions.cuh"
#include "scalarArray.cuh"
#include "labelArray.cuh"

namespace mbLBM
{
    namespace host
    {
        class moments
        {
        public:
            /**
             * @brief Constructs the moments from a number of lattice points
             * @return Object containing all of the moments
             * @param mesh The undecomposed solution mesh
             * @note This constructor zero-initialises everything
             **/
            [[nodiscard]] moments(const latticeMesh &mesh) noexcept
                : mesh_(mesh),
                  rho_(scalarArray(mesh, 0)),
                  u_(scalarArray(mesh, 1)),
                  v_(scalarArray(mesh, 2)),
                  w_(scalarArray(mesh, 3)),
                  m_xx_(scalarArray(mesh, 4)),
                  m_xy_(scalarArray(mesh, 5)),
                  m_xz_(scalarArray(mesh, 6)),
                  m_yy_(scalarArray(mesh, 7)),
                  m_yz_(scalarArray(mesh, 8)),
                  m_zz_(scalarArray(mesh, 9)) {};

            /**
             * @brief Constructs the moments from a pre-defined partition range
             * @return Object containing all of the moments
             * @param mesh The undecomposed solution mesh
             * @param moms The original moments object to be partitioned
             **/
            [[nodiscard]] moments(const latticeMesh &mesh, const moments &moms) noexcept
                : mesh_(mesh),
                  rho_(scalarArray(mesh, moms.rho())),
                  u_(scalarArray(mesh, moms.u())),
                  v_(scalarArray(mesh, moms.v())),
                  w_(scalarArray(mesh, moms.w())),
                  m_xx_(scalarArray(mesh, moms.m_xx())),
                  m_xy_(scalarArray(mesh, moms.m_xy())),
                  m_xz_(scalarArray(mesh, moms.m_xz())),
                  m_yy_(scalarArray(mesh, moms.m_yy())),
                  m_yz_(scalarArray(mesh, moms.m_yz())),
                  m_zz_(scalarArray(mesh, moms.m_zz())) {
                  };

            /**
             * @brief Destructor
             **/
            ~moments() {};

            /**
             * @brief Returns the number of lattices in the x, y and z directions
             **/
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
             * @brief Provides immutable access to the moments
             * @return An immutable access to a scalarArray object
             **/
            [[nodiscard]] inline scalarArray const &rho() const noexcept
            {
                return rho_;
            }
            [[nodiscard]] inline scalarArray const &u() const noexcept
            {
                return u_;
            }
            [[nodiscard]] inline scalarArray const &v() const noexcept
            {
                return v_;
            }
            [[nodiscard]] inline scalarArray const &w() const noexcept
            {
                return w_;
            }
            [[nodiscard]] inline scalarArray const &m_xx() const noexcept
            {
                return m_xx_;
            }
            [[nodiscard]] inline scalarArray const &m_xy() const noexcept
            {
                return m_xy_;
            }
            [[nodiscard]] inline scalarArray const &m_xz() const noexcept
            {
                return m_xz_;
            }
            [[nodiscard]] inline scalarArray const &m_yy() const noexcept
            {
                return m_yy_;
            }
            [[nodiscard]] inline scalarArray const &m_yz() const noexcept
            {
                return m_yz_;
            }
            [[nodiscard]] inline scalarArray const &m_zz() const noexcept
            {
                return m_zz_;
            }

        private:
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
            };

            ~moments() {};

            /**
             * @brief Provides access to the underlying pointers
             * @return A reference to a unique pointer
             **/
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &rho() const noexcept { return rho_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &u() const noexcept { return u_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &v() const noexcept { return v_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &w() const noexcept { return w_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &m_xx() const noexcept { return m_xx_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &m_xy() const noexcept { return m_xy_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &m_xz() const noexcept { return m_xz_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &m_yy() const noexcept { return m_yy_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &m_yz() const noexcept { return m_yz_.ptr(); }
            [[nodiscard]] const scalarPtr_t<decltype(scalarDeleter)> &m_zz() const noexcept { return m_zz_.ptr(); }

        private:
            const deviceIndex_t ID_;
            const cudaError_t err_;
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
}

#endif