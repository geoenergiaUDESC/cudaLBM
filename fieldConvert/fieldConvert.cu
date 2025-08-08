#include "fieldConvert.cuh"

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    const std::vector<label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

    const std::string conversion = getConversionType(programCtrl);

    for (label_t timeStep = fileIO::getStartIndex(programCtrl); timeStep < fileNameIndices.size(); timeStep++)
    {
        const host::array<scalar_t, ctorType::MUST_READ> hostMoments(
            programCtrl,
            {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
            timeStep);

        const std::vector<std::vector<scalar_t>> soa = fileIO::deinterleaveAoSOptimized(hostMoments.arr(), mesh);

        auto it = writers.find(conversion);

        if (it != writers.end())
        {
            const std::string extension = (conversion == "vtu") ? ".vtu" : ".dat";

            const std::string filename = programCtrl.caseName() + "_" + std::to_string(fileNameIndices[timeStep]) + extension;

            const WriterFunction writer = it->second;

            writer(soa, filename, mesh, hostMoments.varNames(), "Title");
        }
        else
        {
            throw std::runtime_error("Unsupported conversion format: " + conversion);
        }
    }

    return 0;
}