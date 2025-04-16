/**
Filename: moments.cuh
Contents: A class containing the arrays of the moment variables
**/

#ifndef __MBLBM_MOMENTS_CUH
#define __MBLBM_MOMENTS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "scalarArray.cuh"

namespace mbLBM
{
    /**
     * @brief Returns the number of lattice components
     * @return The number of stencil components for a D3Q19 lattice
     **/
    template <typename T>
    [[nodiscard]] inline constexpr T Q() noexcept
    {
        return 19;
    }

    namespace host
    {
        class moments
        {
        public:
            /**
             * @brief Constructs the moments from a number of lattice points
             * @return Object containing all of the moments
             * @param nPoints Total number of lattice points
             * @note This constructor zero-initialises everything
             **/
            [[nodiscard]] moments(const label_t nPoints)
                : nPoints_(nPoints),
                  rho_(scalarArray(nPoints)),
                  u_(scalarArray(nPoints)),
                  v_(scalarArray(nPoints)),
                  w_(scalarArray(nPoints)),
                  m_xx_(scalarArray(nPoints)),
                  m_xy_(scalarArray(nPoints)),
                  m_xz_(scalarArray(nPoints)),
                  m_yy_(scalarArray(nPoints)),
                  m_yz_(scalarArray(nPoints)),
                  m_zz_(scalarArray(nPoints)) {};

            /**
             * @brief Destructor
             **/
            ~moments() {};

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
            /**
             * @brief Total number of lattice points
             **/
            const label_t nPoints_;

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

    namespace device
    {
        class moments
        {
        public:
            [[nodiscard]] moments() {};

            ~moments() {};

        private:
        };
    }
}

#endif