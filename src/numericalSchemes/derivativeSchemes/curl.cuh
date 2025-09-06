/**
Filename: curl.cuh
Contents: Curl of a vector field
**/

#ifndef __MBLBM_CURL_CUH
#define __MBLBM_CURL_CUH

namespace LBM
{
    /**
     * @brief Calculates the curl of a vector field
     * @return The curl of (u, v, w)
     * @param u The x component of the vector
     * @param v The y component of the vector
     * @param w The z component of the vector
     * @param mesh The lattice mesh
     **/
    template <const label_t order_, typename T, class M>
    __host__ [[nodiscard]] const std::vector<std::vector<T>> curl(
        const std::vector<T> &u,
        const std::vector<T> &v,
        const std::vector<T> &w,
        const M &mesh)
    {
        // Calculate the derivatives
        const std::vector<double> dwdy = dfdy<order_, double>(w, mesh);
        const std::vector<double> dvdz = dfdz<order_, double>(v, mesh);

        const std::vector<double> dudz = dfdz<order_, double>(u, mesh);
        const std::vector<double> dwdx = dfdx<order_, double>(w, mesh);

        const std::vector<double> dvdx = dfdx<order_, double>(v, mesh);
        const std::vector<double> dudy = dfdy<order_, double>(u, mesh);

        std::vector<T> curl_x(u.size(), 0);
        std::vector<T> curl_y(u.size(), 0);
        std::vector<T> curl_z(u.size(), 0);

        for (label_t i = 0; i < curl_x.size(); i++)
        {
            curl_x[i] = static_cast<T>(dwdy[i] - dvdz[i]);
            curl_y[i] = static_cast<T>(dudz[i] - dwdx[i]);
            curl_z[i] = static_cast<T>(dvdx[i] - dudy[i]);
        }

        return {curl_x, curl_y, curl_z};
    }
}

#endif