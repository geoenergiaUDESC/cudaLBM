#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../momentBasedD3Q19.cuh"
#include "../fileIO/fileIO.cuh"
#include "../runTimeIO/runTimeIO.cuh"
#include "../postProcess.cuh"
#include "../fieldAverage.cuh"
#include "../inputControl.cuh"

using namespace LBM;

/**
 * @brief Veriefies if the command line has the argument -latestTime
 * @return A boolean depending on whether the argument -latestTime is found
 * @param argc First argument passed to main
 * @param argv Second argument passed to main
 **/
__host__ [[nodiscard]] bool isLatestTime(const int argc, const char *const argv[])
{
    const inputControl inputCtrl(argc, argv);

    const std::vector<std::string> commandLine = inputCtrl.parseCommandLine(argc, argv);

    for (label_t arg = 0; arg < commandLine.size(); arg++)
    {
        if (commandLine[arg] == "-latestTime")
        {
            return true;
        }
    }

    return false;
}

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh;

    const std::vector<label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

    if (isLatestTime(argc, argv))
    {
        const host::array<scalar_t, ctorType::MUST_READ> hostMoments(
            programCtrl,
            {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
            fileIO::timeIndices(programCtrl.caseName()).size() - 1);

        postProcess::writeTecplotHexahedralData(
            fileIO::deinterleaveAoSOptimized(hostMoments.arr(), mesh),
            programCtrl.caseName() + "_" + std::to_string(programCtrl.latestTime()) + ".dat",
            mesh,
            hostMoments.varNames(),
            "Title");
    }
    else
    {
        for (label_t timeStep = 0; timeStep < fileNameIndices.size(); timeStep++)
        {
            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "Processing time step: " << fileNameIndices[timeStep] << std::endl;

            // --- FASE 1: LEITURA ---
            const auto start_read = std::chrono::high_resolution_clock::now();
            const host::array<scalar_t, ctorType::MUST_READ> hostMoments(
                programCtrl,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                timeStep);
            const auto end_read = std::chrono::high_resolution_clock::now();
            const auto duration_read = std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read);

            // --- FASE 2: PROCESSAMENTO (Desentrelaçamento) ---
            const auto start_process = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<scalar_t>> soa = fileIO::deinterleaveAoSOptimized(hostMoments.arr(), mesh);
            const auto end_process = std::chrono::high_resolution_clock::now();
            const auto duration_process = std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process);

            // --- FASE 3: ESCRITA ---
            const auto start_write = std::chrono::high_resolution_clock::now();

            std::string datFileName = programCtrl.caseName() + "_" + std::to_string(fileNameIndices[timeStep]) + ".dat";

            postProcess::writeTecplotHexahedralData(
                soa,
                datFileName,
                mesh,
                hostMoments.varNames(),
                "Title");

            const auto end_write = std::chrono::high_resolution_clock::now();
            const auto duration_write = std::chrono::duration_cast<std::chrono::milliseconds>(end_write - start_write);

            // --- RELATÓRIO DE PERFORMANCE ---
            std::cout << "\nPerformance Report for time step " << fileNameIndices[timeStep] << ":" << std::endl;
            std::cout << "  - File Read Time:      " << duration_read.count() << " ms" << std::endl;
            std::cout << "  - Data Processing Time:  " << duration_process.count() << " ms" << std::endl;
            std::cout << "  - File Write Time:       " << duration_write.count() << " ms" << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
        }
    }

    return 0;
}