#include "momentBasedD3Q19.cuh"

// #include "../testRead/testRead.cuh"

using namespace LBM;

// template <typename T, class VSet>
// class hostArray
// {
// public:
//     [[nodiscard]] constexpr hostArray(
//         const std::string &name,
//         const host::latticeMesh &mesh)
//         : arr_(initialConditions(mesh, name)),
//           name_(name) {};

//     ~hostArray() {};

//     /**
//      * @brief Provides read-only access to the underlying std::vector
//      * @return An immutable reference to an std::vector of type T
//      **/
//     [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
//     {
//         return arr_;
//     }

//     /**
//      * @brief Provides access to the variable names
//      * @return An immutable reference to an std::vector of std::strings
//      **/
//     __host__ [[nodiscard]] inline constexpr const std::string &name() const noexcept
//     {
//         return name_;
//     }

// private:
//     const std::vector<T> arr_;

//     const std::string name_;

//     [[nodiscard]] const std::vector<scalar_t> initialConditions(const host::latticeMesh &mesh, const std::string &fieldName)
//     {
//         const boundaryFields<VSet> field_b(fieldName);

//         std::vector<scalar_t> field(mesh.nPoints(), 0);

//         for (label_t bz = 0; bz < mesh.nzBlocks(); bz++)
//         {
//             for (label_t by = 0; by < mesh.nyBlocks(); by++)
//             {
//                 for (label_t bx = 0; bx < mesh.nxBlocks(); bx++)
//                 {
//                     for (label_t tz = 0; tz < block::nz(); tz++)
//                     {
//                         for (label_t ty = 0; ty < block::ny(); ty++)
//                         {
//                             for (label_t tx = 0; tx < block::nx(); tx++)
//                             {
//                                 const label_t x = (bx * block::nx()) + tx;
//                                 const label_t y = (by * block::ny()) + ty;
//                                 const label_t z = (bz * block::nz()) + tz;

//                                 const label_t index = host::idx(tx, ty, tz, bx, by, bz, mesh);

//                                 const bool is_west = (x == 0);
//                                 const bool is_east = (x == mesh.nx() - 1);
//                                 const bool is_south = (y == 0);
//                                 const bool is_north = (y == mesh.ny() - 1);
//                                 const bool is_front = (z == 0);
//                                 const bool is_back = (z == mesh.nz() - 1);

//                                 const label_t boundary_count =
//                                     static_cast<label_t>(is_west) +
//                                     static_cast<label_t>(is_east) +
//                                     static_cast<label_t>(is_south) +
//                                     static_cast<label_t>(is_north) +
//                                     static_cast<label_t>(is_front) +
//                                     static_cast<label_t>(is_back);
//                                 const scalar_t value_sum =
//                                     (is_west * field_b.West()) +
//                                     (is_east * field_b.East()) +
//                                     (is_south * field_b.South()) +
//                                     (is_north * field_b.North()) +
//                                     (is_front * field_b.Front()) +
//                                     (is_back * field_b.Back());

//                                 field[index] = boundary_count > 0 ? value_sum / static_cast<scalar_t>(boundary_count) : field_b.internalField();
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//         return field;
//     }
// };

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    VSet::print();

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup Streams
    const std::array<cudaStream_t, 1> streamsLBM = host::createCudaStream();

    // Allocate the arrays on the host first
    const host::array<scalar_t, VSet> h_rho("rho", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_u("u", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_v("v", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_w("w", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_m_xx("m_xx", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_m_xy("m_xy", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_m_xz("m_xz", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_m_yy("m_yy", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_m_yz("m_yz", mesh, programCtrl);
    const host::array<scalar_t, VSet> h_m_zz("m_zz", mesh, programCtrl);

    device::array<scalar_t> rho(h_rho, mesh);
    device::array<scalar_t> u(h_u, mesh);
    device::array<scalar_t> v(h_v, mesh);
    device::array<scalar_t> w(h_w, mesh);
    device::array<scalar_t> mxx(h_m_xx, mesh);
    device::array<scalar_t> mxy(h_m_xy, mesh);
    device::array<scalar_t> mxz(h_m_xz, mesh);
    device::array<scalar_t> myy(h_m_yy, mesh);
    device::array<scalar_t> myz(h_m_yz, mesh);
    device::array<scalar_t> mzz(h_m_zz, mesh);

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

    device::halo<VSet> blockHalo(
        {h_rho.arr(),
         h_u.arr(),
         h_v.arr(),
         h_w.arr(),
         h_m_xx.arr(),
         h_m_xy.arr(),
         h_m_xz.arr(),
         h_m_yy.arr(),
         h_m_yz.arr(),
         h_m_zz.arr()},
        mesh);

    checkCudaErrors(cudaFuncSetCacheConfig(momentBasedD3Q19, cudaFuncCachePreferShared));

    std::cout << "Time loop start" << std::endl;
    std::cout << std::endl;

    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        // Do the run-time IO
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << "\n";
        }

        // Checkpoint
        if (programCtrl.save(timeStep))
        {
            fileIO::writeFile(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                host::toHost(devPtrs, mesh),
                timeStep);
        }

        // Main kernel
        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
            devPtrs,
            blockHalo.fGhost(),
            blockHalo.gGhost());

        // Halo pointer swap
        blockHalo.swap();
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