/**
Filename: finiteDifference.cuh
Contents: Finite difference schemes up to 8th order
**/

#ifndef __MBLBM_FINITEDIFFERENCE_CUH
#define __MBLBM_FINITEDIFFERENCE_CUH

namespace LBM
{

    template <const label_t order_, typename TReturn, typename T, class M>
    __host__ [[nodiscard]] inline constexpr TReturn finiteDifference(
        const std::vector<T> &padded_line,
        const M &mesh,
        const label_t center,
        const T meshSpacing)
    {
        static_assert((order_ == 2) | (order_ == 4) | (order_ == 6) | (order_ == 8), "Invalid finite difference scheme order: valid orders are 2, 4, 6 and 8");

        if constexpr (order_ == 2)
        {
            return static_cast<TReturn>((padded_line[center + 1] - padded_line[center - 1]) / (2.0 * static_cast<double>(meshSpacing)));
        }

        if constexpr (order_ == 4)
        {
            return static_cast<TReturn>(
                (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                 1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                static_cast<double>(meshSpacing));
        }

        if constexpr (order_ == 6)
        {
            return static_cast<TReturn>(
                (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                 3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                 1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                static_cast<double>(meshSpacing));
        }

        if constexpr (order_ == 8)
        {
            return static_cast<TReturn>(
                (4.0 / 5.0 * (padded_line[center + 1] - padded_line[center - 1]) +
                 1.0 / 5.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                 4.0 / 105.0 * (padded_line[center + 3] - padded_line[center - 3]) +
                 1.0 / 280.0 * (padded_line[center + 4] - padded_line[center - 4])) /
                static_cast<double>(meshSpacing));
        }
    }

    /**
     * @brief Calculates the x derivative of a scalar field
     * @return The x-derivative of f
     * @param f The field to be differentiated
     * @param mesh The lattice mesh
     **/
    template <const label_t order_, typename TReturn, typename T, class M>
    __host__ [[nodiscard]] const std::vector<TReturn> dfdx(
        const std::vector<T> &f,
        const M &mesh)
    {
        static_assert((order_ == 2) | (order_ == 4) | (order_ == 6) | (order_ == 8), "Invalid finite difference scheme order: valid orders are 2, 4, 6 and 8");

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();

        const double dx = 1;

        std::vector<TReturn> dfdx(f.size(), 0);
        constexpr const label_t pad = order_ - 1;
        const label_t nx_padded = nx + 2 * pad;
        std::vector<double> padded_line(nx_padded, 0);

        for (label_t z = 0; z < nz; ++z)
        {
            for (label_t y = 0; y < ny; ++y)
            {
                // Fill interior region of padded_line
                for (label_t x = 0; x < nx; ++x)
                {
                    padded_line[pad + x] = static_cast<double>(f[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())]);
                }

                // Set left ghost cells (reflect & negate)
                for (label_t i = 0; i < pad; ++i)
                {
                    padded_line[i] = static_cast<double>(-f[host::idxScalarGlobal(pad - i, y, z, mesh.nx(), mesh.ny())]);
                }

                // Set right ghost cells (reflect & negate)
                for (label_t i = 0; i < pad; ++i)
                {
                    padded_line[pad + nx + i] = static_cast<double>(-f[host::idxScalarGlobal(nx - 2 - i, y, z, mesh.nx(), mesh.ny())]);
                }

                // Compute derivatives for each point in x-direction
                for (label_t x = 0; x < nx; ++x)
                {
                    const label_t center = pad + x;

                    dfdx[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = finiteDifference<order_, TReturn>(padded_line, mesh, center, dx);
                }
            }
        }
        return dfdx;
    }

    /**
     * @brief Calculates the y derivative of a scalar field
     * @return The y-derivative of f
     * @param f The field to be differentiated
     * @param mesh The lattice mesh
     **/
    template <const label_t order_, typename TReturn, typename T, class M>
    __host__ [[nodiscard]] const std::vector<TReturn> dfdy(
        const std::vector<T> &f,
        const M &mesh)
    {
        static_assert((order_ == 2) | (order_ == 4) | (order_ == 6) | (order_ == 8), "Invalid finite difference scheme order: valid orders are 2, 4, 6 and 8");

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dy = 1; // Adjust based on actual grid spacing

        std::vector<TReturn> dfdy(f.size(), 0);
        constexpr const label_t pad = order_ - 1;
        const label_t ny_padded = ny + 2 * pad;
        std::vector<double> padded_line(ny_padded, 0);

        for (label_t z = 0; z < nz; ++z)
        {
            for (label_t x = 0; x < nx; ++x)
            {
                // Fill interior region of padded_line
                for (label_t y = 0; y < ny; ++y)
                {
                    padded_line[pad + y] = static_cast<double>(f[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())]);
                }

                // Set bottom ghost cells (reflect & negate)
                for (label_t i = 0; i < pad; ++i)
                {
                    padded_line[i] = -static_cast<double>(f[host::idxScalarGlobal(x, pad - i, z, mesh.nx(), mesh.ny())]);
                }

                // Set top ghost cells (reflect & negate)
                for (label_t i = 0; i < pad; ++i)
                {
                    padded_line[pad + ny + i] = -static_cast<double>(f[host::idxScalarGlobal(x, ny - 2 - i, z, mesh.nx(), mesh.ny())]);
                }

                // Compute derivatives for each point in y-direction
                for (label_t y = 0; y < ny; ++y)
                {
                    const label_t center = pad + y;

                    // dfdx[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = finiteDifference<order_, TReturn>(padded_line, mesh, center, y);
                    dfdy[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = finiteDifference<order_, TReturn>(padded_line, mesh, center, dy);
                }
            }
        }
        return dfdy;
    }

    /**
     * @brief Calculates the z derivative of a scalar field
     * @return The z-derivative of f
     * @param f The field to be differentiated
     * @param mesh The lattice mesh
     * @note Currently employs a stencil which is the negative reflection of the interior points
     * @note This should probably be modified to use the positive reflection for a Neumann condition instead
     **/
    template <const label_t order_, typename TReturn, typename T, class M>
    __host__ [[nodiscard]] const std::vector<TReturn> dfdz(
        const std::vector<T> &f,
        const M &mesh)
    {
        static_assert((order_ == 2) | (order_ == 4) | (order_ == 6) | (order_ == 8), "Invalid finite difference scheme order: valid orders are 2, 4, 6 and 8");

        const label_t nx = mesh.nx();
        const label_t ny = mesh.ny();
        const label_t nz = mesh.nz();
        constexpr const double dz = 1; // Adjust based on actual grid spacing

        std::vector<TReturn> dfdz(f.size(), 0);
        constexpr const label_t pad = order_ - 1;
        const label_t nz_padded = nz + 2 * pad;
        std::vector<double> padded_line(nz_padded, 0);

        for (label_t y = 0; y < ny; ++y)
        {
            for (label_t x = 0; x < nx; ++x)
            {
                // Fill interior region of padded_line
                for (label_t z = 0; z < nz; ++z)
                {
                    padded_line[pad + z] = static_cast<double>(f[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())]);
                }

                // Set front ghost cells (reflect & negate)
                for (label_t i = 0; i < pad; ++i)
                {
                    padded_line[i] = -static_cast<double>(f[host::idxScalarGlobal(x, y, pad - i, mesh.nx(), mesh.ny())]);
                }

                // Set back ghost cells (reflect & negate)
                for (label_t i = 0; i < pad; ++i)
                {
                    padded_line[pad + nz + i] = -static_cast<double>(f[host::idxScalarGlobal(x, y, nz - 2 - i, mesh.nx(), mesh.ny())]);
                }

                // Compute derivatives for each point in z-direction
                for (label_t z = 0; z < nz; ++z)
                {
                    const label_t center = pad + z;

                    // dfdx[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = finiteDifference<order_, TReturn>(padded_line, mesh, center, dz);
                    dfdz[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = finiteDifference<order_, TReturn>(padded_line, mesh, center, dz);
                }
            }
        }
        return dfdz;
    }
}

#endif