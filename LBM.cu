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

int main(int argc, char *argv[])
{
    const host::latticeMesh mesh;

    const programControl programCtrl(argc, argv);

    // Set cuda device
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform device memory allocation
    device::array<scalar_t> moments(
        host::moments(mesh, programCtrl.u_inf()),
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        mesh);

    device::array<scalar_t> momentsMean(
        host::moments(mesh, programCtrl.u_inf()),
        {"rhoMean", "uMean", "vMean", "wMean", "m_xxMean", "m_xyMean", "m_xzMean", "m_yyMean", "m_yzMean", "m_zzMean"},
        mesh);

    const device::array<nodeType_t> nodeTypes(host::nodeType(mesh), {"nodeTypes"}, mesh);

    device::halo blockHalo(host::moments(mesh, programCtrl.u_inf()), mesh);

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

    for (label_t timeStep = 0; timeStep < programCtrl.nt(); timeStep++)
    {
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << std::endl;
        }

        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, 0>>>(
            moments.ptr(),
            nodeTypes.ptr(),
            blockHalo);

        fieldAverage::calculate<<<mesh.gridBlock(), mesh.threadBlock(), 0, 0>>>(
            moments.ptr(),
            momentsMean.ptr(),
            nodeTypes.ptr(),
            timeStep);

        blockHalo.swap();

        if (programCtrl.save(timeStep))
        {
            host::write(
                "latticeMesh_" + std::to_string(timeStep) + ".LBMBin",
                moments,
                timeStep);
        }
    }

    std::cout << "End" << std::endl;

    return 0;
}