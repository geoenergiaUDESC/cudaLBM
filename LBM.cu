#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "momentBasedD3Q19.cuh"

using namespace LBM;

int main(int argc, char *argv[])
{
    const host::latticeMesh mesh;

    const programControl programCtrl(argc, argv);

    const host::array<scalar_t> hostMoments(programCtrl, mesh, ctorType::READ_IF_PRESENT);

    // Set cuda device
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform device memory allocation
    device::array<scalar_t> deviceMoments(
        hostMoments.arr(),
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        mesh);
    device::halo blockHalo(hostMoments.arr(), mesh);
    const device::array<nodeType_t> nodeTypes(host::nodeType(mesh), {"nodeTypes"}, mesh);

    // // Set up time averaging
    // // device::array<scalar_t> momentsMean(
    // //     host::moments(mesh, programCtrl.u_inf()),
    // //     {"rhoMean", "uMean", "vMean", "wMean", "m_xxMean", "m_xyMean", "m_xzMean", "m_yyMean", "m_yzMean", "m_zzMean"},
    // //     mesh);

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy symbols to device
    mesh.copyDeviceSymbols();
    programCtrl.copyDeviceSymbols(mesh.nx());

    std::cout << "Restarting from t = " << programCtrl.latestTime() << std::endl;

    for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << std::endl;
        }

        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, 0>>>(
            deviceMoments.ptr(),
            nodeTypes.ptr(),
            blockHalo);

        // fieldAverage::calculate<<<mesh.gridBlock(), mesh.threadBlock(), 0, 0>>>(
        //     moments.ptr(),
        //     momentsMean.ptr(),
        //     nodeTypes.ptr(),
        //     timeStep);

        blockHalo.swap();

        if (programCtrl.save(timeStep))
        {
            deviceMoments.write(programCtrl.caseName(), timeStep);
        }

        checkCudaErrors(cudaDeviceSynchronize());
    }

    std::cout << "End" << std::endl;

    return 0;
}