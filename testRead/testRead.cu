#include "testRead.cuh"

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    const std::vector<scalar_t> rho = initialise_from_boundary_conditions<VSet>(mesh, "rho");
    const std::vector<scalar_t> u = initialise_from_boundary_conditions<VSet>(mesh, "u");
    const std::vector<scalar_t> v = initialise_from_boundary_conditions<VSet>(mesh, "v");
    const std::vector<scalar_t> w = initialise_from_boundary_conditions<VSet>(mesh, "w");
    const std::vector<scalar_t> m_xx = initialise_from_boundary_conditions<VSet>(mesh, "m_xx");
    const std::vector<scalar_t> m_xy = initialise_from_boundary_conditions<VSet>(mesh, "m_xy");
    const std::vector<scalar_t> m_xz = initialise_from_boundary_conditions<VSet>(mesh, "m_xz");
    const std::vector<scalar_t> m_yy = initialise_from_boundary_conditions<VSet>(mesh, "m_yy");
    const std::vector<scalar_t> m_yz = initialise_from_boundary_conditions<VSet>(mesh, "m_yz");
    const std::vector<scalar_t> m_zz = initialise_from_boundary_conditions<VSet>(mesh, "m_zz");

    std::vector<std::vector<scalar_t>> toPrint(10, std::vector<scalar_t>(mesh.nPoints(), 0));

    for (label_t bz = 0; bz < mesh.nzBlocks(); bz++)
    {
        for (label_t by = 0; by < mesh.nyBlocks(); by++)
        {
            for (label_t bx = 0; bx < mesh.nxBlocks(); bx++)
            {
                for (label_t tz = 0; tz < block::nz(); tz++)
                {
                    for (label_t ty = 0; ty < block::ny(); ty++)
                    {
                        for (label_t tx = 0; tx < block::nx(); tx++)
                        {
                            const label_t x = (bx * block::nx()) + tx;
                            const label_t y = (by * block::ny()) + ty;
                            const label_t z = (bz * block::nz()) + tz;

                            const label_t read_index = host::idx(tx, ty, tz, bx, by, bz, mesh);

                            const label_t write_index = host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny());

                            toPrint[0][write_index] = rho[read_index];
                            toPrint[1][write_index] = u[read_index];
                            toPrint[2][write_index] = v[read_index];
                            toPrint[3][write_index] = w[read_index];
                            toPrint[4][write_index] = m_xx[read_index];
                            toPrint[5][write_index] = m_xy[read_index];
                            toPrint[6][write_index] = m_xz[read_index];
                            toPrint[7][write_index] = m_yy[read_index];
                            toPrint[8][write_index] = m_yz[read_index];
                            toPrint[9][write_index] = m_zz[read_index];
                        }
                    }
                }
            }
        }
    }

    postProcess::writeVTU(toPrint, "test.vtu", mesh, {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"});

    // const boundaryFields u("u");

    // const boundaryFields v("v");

    // const boundaryFields w("w");

    // const boundaryFields m_xx("m_xx");

    return 0;
}