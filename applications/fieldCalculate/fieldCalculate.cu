#include "fieldCalculate.cuh"

using namespace LBM;

// static constexpr const label_t SchemeOrder = 8;

[[nodiscard]] inline consteval label_t SchemeOrder() { return 8; }

template <typename T>
[[nodiscard]] const std::vector<T> mag(const std::vector<T> &u, const std::vector<T> &v, const std::vector<T> &w)
{
    // Add a size check here

    std::vector<scalar_t> magu(u.size(), 0);

    for (label_t i = 0; i < u.size(); i++)
    {
        magu[i] = std::sqrt((u[i] * u[i]) + (v[i] * v[i]) + (w[i] * w[i]));
    }

    return magu;
}

using VSet = VelocitySet::D3Q19;

int main(const int argc, const char *const argv[])
{

    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    const host::arrayCollection<scalar_t, ctorType::MUST_READ, VSet> hostMoments(
        programCtrl,
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"});

    // Get the fields
    const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoSOptimized(hostMoments.arr(), mesh);

    // Calculate the magnitude of velocity
    const std::vector<scalar_t> magu = mag(fields[index::u()], fields[index::v()], fields[index::w()]);

    // Calculate the divergence of velocity
    const std::vector<scalar_t> divu = div<SchemeOrder()>(fields[index::u()], fields[index::v()], fields[index::w()], mesh);

    // Calculate the vorticity
    const std::vector<std::vector<scalar_t>> omega = curl<SchemeOrder()>(fields[index::u()], fields[index::v()], fields[index::w()], mesh);

    // Calculate the magnitude of vorticity
    const std::vector<scalar_t> magomega = mag(omega[0], omega[1], omega[2]);

    constexpr label_t IntegrationOrder = 2;

    // Integrate the vorticity in all axes
    const std::vector<scalar_t> int_omega_x = integrate_x<IntegrationOrder, scalar_t>(omega[0], mesh);
    const std::vector<scalar_t> int_omega_y = integrate_y<IntegrationOrder, scalar_t>(omega[1], mesh);
    const std::vector<scalar_t> int_omega_z = integrate_z<IntegrationOrder, scalar_t>(omega[2], mesh);

    const std::vector<std::vector<scalar_t>> integratedOmega = {int_omega_x, int_omega_y, int_omega_z};

    // Write the files
    // postProcess::writeVTU({magu}, "mag[u].vtu", mesh, {"mag[u]"});
    // postProcess::writeVTU({divu}, "div[u].vtu", mesh, {"div[u]"});
    // postProcess::writeVTU(omega, "curl[u].vtu", mesh, {"curl_x[u]", "curl_y[u]", "curl_z[u]"});
    // postProcess::writeVTU({magomega}, "mag[curl[u]].vtu", mesh, {"mag[curl[u]]"});

    postProcess::writeVTU(integratedOmega, "integrated_omega.vtu", mesh, {"int_omega_x", "int_omega_y", "int_omega_z"});

    return 0;
}