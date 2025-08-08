/**
Filename: div.cuh
Contents: Divergence of a vector field
**/

#ifndef __MBLBM_DIV_CUH
#define __MBLBM_DIV_CUH

namespace LBM
{
    /**
     * @brief Calculates the divergence of a vector field
     * @return The divergence of (u, v, w)
     * @param u The x component of the vector
     * @param v The y component of the vector
     * @param w The z component of the vector
     * @param mesh The lattice mesh
     **/
    template <const label_t order_, typename T, class M>
    __host__ [[nodiscard]] const std::vector<T> div(
        const std::vector<T> &u,
        const std::vector<T> &v,
        const std::vector<T> &w,
        const M &mesh)
    {
        // Calculate the components of div
        const std::vector<double> dudx = dfdx<order_, double>(u, mesh);
        const std::vector<double> dvdy = dfdy<order_, double>(v, mesh);
        const std::vector<double> dwdz = dfdz<order_, double>(w, mesh);

        // Sum the components
        std::vector<T> divu(dudx.size(), 0);
        for (label_t i = 0; i < dudx.size(); i++)
        {
            divu[i] = static_cast<T>(dudx[i] + dvdy[i] + dwdz[i]);
        }

        // Return div
        return divu;
    }

}

#endif