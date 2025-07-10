#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "momentBasedD3Q19.cuh"
#include "fileIO/fileIO.cuh"
#include "runTimeIO/runTimeIO.cuh"
#include "postProcess.cuh"
#include "fieldAverage.cuh"

using namespace LBM;

[[nodiscard]] const std::array<cudaStream_t, 1> createCudaStream() noexcept
{
    std::array<cudaStream_t, 1> streamsLBM;

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    return streamsLBM;
}

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh;

    VelocitySet::D3Q19::print();

    const host::array<scalar_t, ctorType::READ_IF_PRESENT> hostMoments(
        programCtrl,
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        mesh);

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup Streams
    const std::array<cudaStream_t, 1> streamsLBM = createCudaStream();

    // Perform device memory allocation
    device::array<scalar_t> deviceMoments(
        hostMoments.arr(),
        hostMoments.varNames(),
        mesh);
    device::halo blockHalo(hostMoments.arr(), mesh);

    // Copy symbols to device
    mesh.copyDeviceSymbols();
    programCtrl.copyDeviceSymbols(mesh.nx());

    // checkCudaErrors(cudaFuncSetCacheConfig(momentBasedD3Q19, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncSetCacheConfig(momentBasedD3Q19, cudaFuncCachePreferL1));

    std::cout << "Time loop start" << std::endl;
    std::cout << std::endl;

    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << "\n";
        }

        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
            deviceMoments.ptr(),
            blockHalo);

        blockHalo.swap();

        // if (programCtrl.save(timeStep))
        // {
        //     deviceMoments.write(programCtrl.caseName(), timeStep);
        // }
    }

    // Get ending time point and output the elapsed time
    const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::cout << "Elapsed time: " << runTimeIO::duration(std::chrono::duration_cast<std::chrono::seconds>(end - start).count()) << std::endl;
    std::cout << std::endl;
    std::cout << "MLUPS: " << runTimeIO::MLUPS<double>(mesh, programCtrl, start, end) << std::endl;
    std::cout << "End" << std::endl;

    return 0;
}