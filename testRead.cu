#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
#include "array/array.cuh"
#include "programControl.cuh"
#include "latticeMesh/latticeMesh.cuh"
#include "moments/moments.cuh"
#include "collision.cuh"
#include "postProcess.cuh"
#include "momentBasedD3Q19.cuh"
#include "fieldAverage.cuh"
#include "cavity.cuh"
#include "fileIO/fileIO.cuh"

using namespace LBM;

constexpr const std::string_view fileName = std::string_view("latticeMesh_1000.LBMBin");

int main(int argc, char *argv[])
{
    const host::latticeMesh mesh;

    const programControl programCtrl(argc, argv);

    const std::vector<scalar_t> fMom = readFieldFile<scalar_t>(std::string(fileName));

    std::cout << "Size of fMom: " << fMom.size() << std::endl;

    postProcess::writeTecplotHexahedralData(
        deinterleaveAoS(fMom, mesh),
        "out.dat",
        mesh,
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        "Title");

    return 0;
}