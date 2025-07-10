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

int main(int argc, char *argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh;

    VelocitySet::D3Q19::print();

    // const std::vector<std::vector<scalar_t>> hostMoments = host::moments_v2(mesh, programCtrl.u_inf());

    // device::array<scalar_t> rho(hostMoments[0], {"rho"}, mesh);
    // device::array<scalar_t> u(hostMoments[1], {"u"}, mesh);
    // device::array<scalar_t> v(hostMoments[2], {"v"}, mesh);
    // device::array<scalar_t> w(hostMoments[3], {"w"}, mesh);
    // device::array<scalar_t> m_xx(hostMoments[4], {"m_xx"}, mesh);
    // device::array<scalar_t> m_xy(hostMoments[5], {"m_xy"}, mesh);
    // device::array<scalar_t> m_xz(hostMoments[6], {"m_xz"}, mesh);
    // device::array<scalar_t> m_yy(hostMoments[7], {"m_yy"}, mesh);
    // device::array<scalar_t> m_yz(hostMoments[8], {"m_yz"}, mesh);
    // device::array<scalar_t> m_zz(hostMoments[9], {"m_zz"}, mesh);

    // const host::array<scalar_t, ctorType::NO_READ> uHost(programCtrl, mesh);

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
    // const device::array<nodeType_t> nodeTypes(host::nodeType(mesh), {"nodeTypes"}, mesh);

    // Set up time averaging
    // device::array<scalar_t> momentsMean(
    //     host::moments(mesh, programCtrl.u_inf()),
    //     {"rhoMean", "uMean", "vMean", "wMean", "m_xxMean", "m_xyMean", "m_xzMean", "m_yyMean", "m_yzMean", "m_zzMean"},
    //     mesh);

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

        // momentBasedD3Q19_v2<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
        //     rho.ptr(),
        //     u.ptr(),
        //     v.ptr(),
        //     w.ptr(),
        //     m_xx.ptr(),
        //     m_xy.ptr(),
        //     m_xz.ptr(),
        //     m_yy.ptr(),
        //     m_yz.ptr(),
        //     m_zz.ptr(),
        //     blockHalo);

        // checkCudaErrors(cudaDeviceSynchronize());
        // fieldAverage::calculate<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
        //     deviceMoments.ptr(),
        //     momentsMean.ptr(),
        //     nodeTypes.ptr(),
        //     timeStep);

        blockHalo.swap();

        // if (programCtrl.save(timeStep))
        // {
        //     // deviceMoments.write(programCtrl.caseName(), timeStep);

        //     if (timeStep > 0)
        //     {
        //         postProcess::writeTecplotHexahedralData(
        //             fileIO::deinterleaveAoS(host::copyToHost(deviceMoments.ptr(), deviceMoments.size()), mesh),
        //             programCtrl.caseName() + "_" + std::to_string(timeStep) + ".dat",
        //             mesh,
        //             deviceMoments.varNames(),
        //             "Title");
        //     }
        // }
    }

    // Get ending time point and output the elapsed time
    const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::cout << "Elapsed time: " << runTimeIO::duration(std::chrono::duration_cast<std::chrono::seconds>(end - start).count()) << std::endl;
    std::cout << std::endl;
    std::cout << "MLUPS: " << std::setprecision(15) << runTimeIO::MLUPS<double>(mesh, programCtrl, start, end) << std::endl;
    std::cout << "End" << std::endl;

    return 0;
}