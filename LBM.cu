#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "momentBasedD3Q19.cuh"

#include "fileIO/fileIO.cuh"
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

int main(int argc, char *argv[])
{
    const host::latticeMesh mesh;

    const programControl programCtrl(argc, argv);

    VelocitySet::D3Q19::print();

    const host::array<scalar_t, ctorType::READ_IF_PRESENT> hostMoments(programCtrl, mesh);

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup Streams
    const std::array<cudaStream_t, 1> streamsLBM = createCudaStream();

    // Perform device memory allocation
    device::array<scalar_t> deviceMoments(
        hostMoments.arr(),
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        mesh);
    device::halo blockHalo(hostMoments.arr(), mesh);
    const device::array<nodeType_t> nodeTypes(host::nodeType(mesh), {"nodeTypes"}, mesh);

    // Set up time averaging
    device::array<scalar_t> momentsMean(
        host::moments(mesh, programCtrl.u_inf()),
        {"rhoMean", "uMean", "vMean", "wMean", "m_xxMean", "m_xyMean", "m_xzMean", "m_yyMean", "m_yzMean", "m_zzMean"},
        mesh);

    // Copy symbols to device
    mesh.copyDeviceSymbols();
    programCtrl.copyDeviceSymbols(mesh.nx());

    std::cout << "Time loop start" << std::endl;
    std::cout << std::endl;

    for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << "\n";
        }

        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
            deviceMoments.ptr(),
            nodeTypes.ptr(),
            blockHalo);

        checkCudaErrors(cudaDeviceSynchronize());

        fieldAverage::calculate<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
            deviceMoments.ptr(),
            momentsMean.ptr(),
            nodeTypes.ptr(),
            timeStep);

        blockHalo.swap();

        if (programCtrl.save(timeStep))
        {
            deviceMoments.write(programCtrl.caseName(), timeStep);

            postProcess::writeTecplotHexahedralData(
                fileIO::deinterleaveAoS(host::copyToHost(deviceMoments.ptr(), mesh.nPoints() * 10), mesh),
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".dat",
                mesh,
                deviceMoments.varNames(),
                "Title");

            // momentsMean.write(programCtrl.caseName(), timeStep);

            postProcess::writeTecplotHexahedralData(
                fileIO::deinterleaveAoS(host::copyToHost(momentsMean.ptr(), mesh.nPoints() * 10), mesh),
                programCtrl.caseName() + "Mean_" + std::to_string(timeStep) + ".dat",
                mesh,
                momentsMean.varNames(),
                "Title");
        }

        checkCudaErrors(cudaDeviceSynchronize());
    }

    std::cout << "End" << std::endl;

    return 0;
}