#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "momentBasedD3Q19.cuh"

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    VSet::print();

    const host::array<scalar_t, ctorType::READ_IF_PRESENT> hostMoments(
        programCtrl,
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        mesh);

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup Streams
    const std::array<cudaStream_t, 1> streamsLBM = host::createCudaStream();

    // Perform device memory allocation
    device::array<scalar_t> rho(host::toDevice<0>(hostMoments.arr(), mesh), {"rho"}, mesh);
    device::array<scalar_t> u(host::toDevice<1>(hostMoments.arr(), mesh), {"u"}, mesh);
    device::array<scalar_t> v(host::toDevice<2>(hostMoments.arr(), mesh), {"v"}, mesh);
    device::array<scalar_t> w(host::toDevice<3>(hostMoments.arr(), mesh), {"w"}, mesh);
    device::array<scalar_t> mxx(host::toDevice<4>(hostMoments.arr(), mesh), {"m_xx"}, mesh);
    device::array<scalar_t> mxy(host::toDevice<5>(hostMoments.arr(), mesh), {"m_xy"}, mesh);
    device::array<scalar_t> mxz(host::toDevice<6>(hostMoments.arr(), mesh), {"m_xz"}, mesh);
    device::array<scalar_t> myy(host::toDevice<7>(hostMoments.arr(), mesh), {"m_yy"}, mesh);
    device::array<scalar_t> myz(host::toDevice<8>(hostMoments.arr(), mesh), {"m_yz"}, mesh);
    device::array<scalar_t> mzz(host::toDevice<9>(hostMoments.arr(), mesh), {"m_zz"}, mesh);

    const device::ptrCollection<10, scalar_t> devPtrs(
        rho.ptr(),
        u.ptr(),
        v.ptr(),
        w.ptr(),
        mxx.ptr(),
        mxy.ptr(),
        mxz.ptr(),
        myy.ptr(),
        myz.ptr(),
        mzz.ptr());

    device::array<scalar_t> deviceMoments(hostMoments, mesh);
    device::halo blockHalo(hostMoments.arr(), mesh);

    checkCudaErrors(cudaFuncSetCacheConfig(momentBasedD3Q19, cudaFuncCachePreferShared));

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
            devPtrs,
            blockHalo.fGhost(),
            blockHalo.gGhost());

        blockHalo.swap();

        if (programCtrl.save(timeStep))
        {
            fileIO::writeFile(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                hostMoments.varNames(),
                host::toHost(devPtrs, mesh),
                timeStep);
        }
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