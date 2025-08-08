/**
 * Filename: fieldIntegrate.cuh
 * Contents: Field integration schemes
 **/

#ifndef __MBLBM_FIELDINTEGRATE_CUH
#define __MBLBM_FIELDINTEGRATE_CUH

namespace LBM
{
    /**
     * @brief Calculates the integral of a scalar field along the x-axis.
     * @tparam order_ The order of the integration scheme. Currently, only 2nd order is implemented.
     * @return The integrated field of f with respect to x.
     * @param f The field to be integrated.
     * @param mesh The lattice mesh.
     * @note This function uses the cumulative trapezoidal rule. The integration constant is set by
     * assuming the integral is zero at x=0 for each (y, z) line.
     **/
    template <const label_t order_, typename TReturn, typename T, class M>
    __host__ [[nodiscard]] const std::vector<TReturn> integrate_x(
        const std::vector<T> &f,
        const M &mesh)
    {
        static_assert(order_ == 2, "Invalid integration scheme order: only 2nd order (Trapezoidal Rule) is currently implemented.");

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dx = 1.0;

        std::vector<TReturn> integral_f(f.size(), 0);

        for (label_t z = 0; z < nz; ++z)
        {
            for (label_t y = 0; y < ny; ++y)
            {
                // Initial condition for integration along this x-line
                integral_f[host::idxScalarGlobal(0, y, z, nx, ny)] = 0;

                // Cumulative integration using the trapezoidal rule
                for (label_t x = 1; x < nx; ++x)
                {
                    const label_t current_idx = host::idxScalarGlobal(x, y, z, nx, ny);
                    const label_t prev_idx = host::idxScalarGlobal(x - 1, y, z, nx, ny);

                    integral_f[current_idx] = integral_f[prev_idx] + static_cast<TReturn>(0.5 * dx * (static_cast<double>(f[prev_idx]) + static_cast<double>(f[current_idx])));
                }
            }
        }
        return integral_f;
    }

    /**
     * @brief Calculates the integral of a scalar field along the y-axis.
     **/
    template <const label_t order_, typename TReturn, typename T, class M>
    __host__ [[nodiscard]] const std::vector<TReturn> integrate_y(
        const std::vector<T> &f,
        const M &mesh)
    {
        static_assert(order_ == 2, "Invalid integration scheme order: only 2nd order (Trapezoidal Rule) is currently implemented.");

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dy = 1.0;

        std::vector<TReturn> integral_f(f.size(), 0);

        for (label_t z = 0; z < nz; ++z)
        {
            for (label_t x = 0; x < nx; ++x)
            {
                // Initial condition for integration along this y-line
                integral_f[host::idxScalarGlobal(x, 0, z, nx, ny)] = 0;

                // Cumulative integration using the trapezoidal rule
                for (label_t y = 1; y < ny; ++y)
                {
                    const label_t current_idx = host::idxScalarGlobal(x, y, z, nx, ny);
                    const label_t prev_idx = host::idxScalarGlobal(x, y - 1, z, nx, ny);

                    integral_f[current_idx] = integral_f[prev_idx] + static_cast<TReturn>(0.5 * dy * (static_cast<double>(f[prev_idx]) + static_cast<double>(f[current_idx])));
                }
            }
        }
        return integral_f;
    }

    /**
     * @brief Calculates the integral of a scalar field along the z-axis.
     **/
    template <const label_t order_, typename TReturn, typename T, class M>
    __host__ [[nodiscard]] const std::vector<TReturn> integrate_z(
        const std::vector<T> &f,
        const M &mesh)
    {
        static_assert(order_ == 2, "Invalid integration scheme order: only 2nd order (Trapezoidal Rule) is currently implemented.");

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dz = 1.0;

        std::vector<TReturn> integral_f(f.size(), 0);

        for (label_t y = 0; y < ny; ++y)
        {
            for (label_t x = 0; x < nx; ++x)
            {
                // Initial condition for integration along this z-line
                integral_f[host::idxScalarGlobal(x, y, 0, nx, ny)] = 0;

                // Cumulative integration using the trapezoidal rule
                for (label_t z = 1; z < nz; ++z)
                {
                    const label_t current_idx = host::idxScalarGlobal(x, y, z, nx, ny);
                    const label_t prev_idx = host::idxScalarGlobal(x, y, z - 1, nx, ny);

                    integral_f[current_idx] = integral_f[prev_idx] + static_cast<TReturn>(0.5 * dz * (static_cast<double>(f[prev_idx]) + static_cast<double>(f[current_idx])));
                }
            }
        }
        return integral_f;
    }
}

#endif // __MBLBM_FIELDINTEGRATE_CUH