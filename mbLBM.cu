#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
// #include "programControl.cuh"
// #include "inputControl.cuh"
// #include "latticeMesh/latticeMesh.cuh"
// #include "moments/moments.cuh"
// #include "collision.cuh"
// #include "postProcess.cuh"

using namespace mbLBM;

// // Class used to initialise constants on the device
// class deviceConstants
// {
// public:
//     [[nodiscard]] deviceConstants(
//         const host::latticeMesh &mesh,
//         const programControl &programCtrl) noexcept
//         : nxSuccess_(copySymbolToDevice(d_nx, mesh.nx())),
//           nySuccess_(copySymbolToDevice(d_ny, mesh.ny())),
//           nzSuccess_(copySymbolToDevice(d_nz, mesh.nz())),
//           ReSuccess_(copySymbolToDevice(d_Re, programCtrl.Re())),
//           u_infSuccess_(copySymbolToDevice(d_u_inf, programCtrl.u_inf())),
//           omegaSuccess_(copySymbolToDevice(d_omega, static_cast<scalar_t>(1.0) / (static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * (programCtrl.u_inf() * static_cast<scalar_t>(mesh.nx()) / programCtrl.Re())))),
//           nxBlocksSuccess_(copySymbolToDevice(NUM_BLOCK_X, mesh.nx() / block::nx())),
//           nyBlocksSuccess_(copySymbolToDevice(NUM_BLOCK_Y, mesh.ny() / block::ny())),
//           nzBlocksSuccess_(copySymbolToDevice(NUM_BLOCK_Z, mesh.nz() / block::nz())) {};

//     ~deviceConstants() {};

// private:
//     const cudaError_t nxSuccess_;
//     const cudaError_t nySuccess_;
//     const cudaError_t nzSuccess_;
//     const cudaError_t ReSuccess_;
//     const cudaError_t u_infSuccess_;
//     const cudaError_t omegaSuccess_;
//     const cudaError_t nxBlocksSuccess_;
//     const cudaError_t nyBlocksSuccess_;
//     const cudaError_t nzBlocksSuccess_;

//     template <typename T>
//     [[nodiscard]] cudaError_t copySymbolToDevice(const T &dev_symbol, const T src) const noexcept
//     {
//         const T src_ = src;

//         cudaError_t i = cudaMemcpyToSymbol(dev_symbol, &src_, sizeof(T), 0, cudaMemcpyHostToDevice);

//         if (i != cudaSuccess)
//         {
//             std::cout << "Failed to copy symbol to device in constructor of deviceConstants" << std::endl;
//         }

//         return i;
//     }
// };

// __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void devConstantsTest()
// {
// #ifdef LABEL_SIZE_32
//     printf("nx = %u\n", d_nx);
//     printf("ny = %u\n", d_ny);
//     printf("nz = %u\n", d_nz);

//     printf("Number of x blocks = %u\n", NUM_BLOCK_X);
//     printf("Number of y blocks = %u\n", NUM_BLOCK_Y);
//     printf("Number of z blocks = %u\n", NUM_BLOCK_Z);
// #elif LABEL_SIZE_64
//     printf("nx = %lu\n", d_nx);
//     printf("ny = %lu\n", d_ny);
//     printf("nz = %lu\n", d_nz);

//     printf("Number of x blocks = %lu\n", NUM_BLOCK_X);
//     printf("Number of y blocks = %lu\n", NUM_BLOCK_Y);
//     printf("Number of z blocks = %lu\n", NUM_BLOCK_Z);
// #endif

//     printf("Re = %0.12g\n", d_Re);
//     printf("u_inf = %0.12g\n", d_u_inf);
//     printf("omega = %0.12g\n", d_omega);
// }

// __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuInitialization_pop(
//     const scalar_t *const ptrRestrict d_mom,
//     const nodeType::type *const ptrRestrict nodeTypes,
//     scalar_t *const ptrRestrict f_x0,
//     scalar_t *const ptrRestrict f_x1,
//     scalar_t *const ptrRestrict f_y0,
//     scalar_t *const ptrRestrict f_y1,
//     scalar_t *const ptrRestrict f_z0,
//     scalar_t *const ptrRestrict f_z1)
// {
//     const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
//     const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
//     const label_t z = threadIdx.z + blockDim.z * blockIdx.z;
//     if (x >= d_nx || y >= d_ny || z >= d_nz)
//     {
//         return;
//     }

//     // size_t index = idxScalarGlobal(x, y, z);
//     // zeroth moment

//     // scalar_t rhoVar = RHO_0 + fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t ux_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t uy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t uz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t m_xx_t45 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t m_xy_t90 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t m_xz_t90 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t m_yy_t45 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t m_yz_t90 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
//     // scalar_t m_zz_t45 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

//     constexpr const scalar_t RHO_0 = 1.0;
//     scalar_t moments[10] = {
//         d_mom[idxMom__<0, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] + RHO_0,
//         d_mom[idxMom__<1, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<2, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<3, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<4, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<5, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<6, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<7, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<8, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
//         d_mom[idxMom__<9, 10>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)]};

//     //     scalar_t pop[Q];
//     //     scalar_t multiplyTerm;
//     //     scalar_t pics2;
//     // #include COLREC_RECONSTRUCTIONS

//     // Perform the reconstruction
//     scalar_t pop[VelocitySet::D3Q19::Q()];

//     // Definition of reconstruction seems correct
//     VelocitySet::D3Q19::reconstruct(moments, pop);

//     // thread xyz
//     const label_t tx = threadIdx.x;
//     const label_t ty = threadIdx.y;
//     const label_t tz = threadIdx.z;

//     // block xyz
//     const label_t bx = blockIdx.x;
//     const label_t by = blockIdx.y;
//     const label_t bz = blockIdx.z;

//     if (threadIdx.x == 0)
//     { // w
//         f_x0[idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[2];
//         f_x0[idxPopX<1, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[8];
//         f_x0[idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[10];
//         f_x0[idxPopX<3, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[14];
//         f_x0[idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[16];
//     }
//     else if (threadIdx.x == (block::nx() - 1))
//     {
//         f_x1[idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[1];
//         f_x1[idxPopX<1, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[7];
//         f_x1[idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[9];
//         f_x1[idxPopX<3, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[13];
//         f_x1[idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tz, bx, by, bz)] = pop[15];
//     }

//     if (threadIdx.y == 0)
//     { // s
//         f_y0[idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[4];
//         f_y0[idxPopY<1, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[8];
//         f_y0[idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[12];
//         f_y0[idxPopY<3, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[13];
//         f_y0[idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[18];
//     }
//     else if (threadIdx.y == (block::ny() - 1))
//     {
//         f_y1[idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[3];
//         f_y1[idxPopY<1, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[7];
//         f_y1[idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[11];
//         f_y1[idxPopY<3, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[14];
//         f_y1[idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tz, bx, by, bz)] = pop[17];
//     }

//     if (threadIdx.z == 0)
//     { // b
//         f_z0[idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[6];
//         f_z0[idxPopZ<1, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[10];
//         f_z0[idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[12];
//         f_z0[idxPopZ<3, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[15];
//         f_z0[idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[17];
//     }
//     else if (threadIdx.z == (block::nz() - 1))
//     {
//         f_z1[idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[5];
//         f_z1[idxPopZ<1, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[9];
//         f_z1[idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[11];
//         f_z1[idxPopZ<3, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[16];
//         f_z1[idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bz)] = pop[18];
//     }
// }

// void print(const dim3 &Dim3, const std::string &name) noexcept
// {
//     std::cout << name << ": " << Dim3.x << " " << Dim3.y << " " << Dim3.z << std::endl;
// }

#define GPU_INDEX 1

#define console_flush 0

// #define NUMBER_MOMENTS 10

#include "original.cuh"

template <typename T>
[[nodiscard]] const std::vector<T> deviceToHost(const T *f, const label_t nFields = 1)
{
    std::vector<T> F(NUMBER_LBM_POP_NODES * nFields, 0);

    const cudaError_t i = cudaMemcpy(F.data(), f, NUMBER_LBM_POP_NODES * sizeof(T) * nFields, cudaMemcpyDeviceToHost);

    if (i != cudaSuccess)
    {
        // exceptions::program_exit(i, "Unable to copy array");
    }
    else
    {
#ifdef VERBOSE
        std::cout << "Copied " << sizeof(T) * NUMBER_LBM_POP_NODES << " bytes of memory in cudaMemcpy from address " << f << " to the host" << std::endl;
#endif
    }

    return F;
}

void writeTecplotHexahedralData(
    const std::vector<scalar_t> &solutionData,
    const std::string &filename,
    const label_t ni,
    const label_t nj,
    const label_t nk,
    const std::string &title,
    const std::vector<std::string> &solutionVarNames)
{
    // Check input sizes
    const label_t numNodes = ni * nj * nk;
    if (!solutionData.empty() && solutionData.size() != numNodes * solutionVarNames.size())
    {
        std::cerr << "Error: Solution data size doesn't match grid dimensions and variable count\n";
        return;
    }

    std::ofstream outFile(filename);
    if (!outFile)
    {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    // Set high precision output
    outFile << std::setprecision(12);

    // Write Tecplot header
    outFile << "TITLE = \"" << title << "\"\n";
    outFile << "VARIABLES = \"X\" \"Y\" \"Z\" ";
    for (auto &name : solutionVarNames)
    {
        outFile << "\"" << name << "\" ";
    }
    outFile << "\n";

    // UNSTRUCTURED GRID FORMAT (explicit connectivity)
    const label_t numElements = (ni - 1) * (nj - 1) * (nk - 1);
    outFile << "ZONE T=\"Hexahedral Zone\", NODES=" << numNodes << ", ELEMENTS=" << numElements << ", DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n";

    // Write all node coordinates first
    // X coordinates
    for (label_t k = 0; k < nk; k++)
    {
        for (label_t j = 0; j < nj; j++)
        {
            for (label_t i = 0; i < ni; i++)
            {
                outFile << static_cast<double>(i) / static_cast<double>(ni - 1) << "\n";
            }
        }
    }

    // Y coordinates
    for (label_t k = 0; k < nk; k++)
    {
        for (label_t j = 0; j < nj; j++)
        {
            for (label_t i = 0; i < ni; i++)
            {
                outFile << static_cast<double>(j) / static_cast<double>(nj - 1) << "\n";
            }
        }
    }

    // Z coordinates
    for (label_t k = 0; k < nk; k++)
    {
        for (label_t j = 0; j < nj; j++)
        {
            for (label_t i = 0; i < ni; i++)
            {
                outFile << static_cast<double>(k) / static_cast<double>(nk - 1) << "\n";
            }
        }
    }

    // Write solution variable
    for (label_t i = 0; i < numNodes; i++)
    {
        outFile << solutionData[i] << "\n";
    }

    // Write connectivity (1-based indexing)
    for (label_t k = 0; k < nk - 1; k++)
    {
        for (label_t j = 0; j < nj - 1; j++)
        {
            for (label_t i = 0; i < ni - 1; i++)
            {
                // Get the 8 nodes of the hexahedron
                const label_t n0 = k * ni * nj + j * ni + i + 1;
                const label_t n1 = k * ni * nj + j * ni + (i + 1) + 1;
                const label_t n2 = k * ni * nj + (j + 1) * ni + (i + 1) + 1;
                const label_t n3 = k * ni * nj + (j + 1) * ni + i + 1;
                const label_t n4 = (k + 1) * ni * nj + j * ni + i + 1;
                const label_t n5 = (k + 1) * ni * nj + j * ni + (i + 1) + 1;
                const label_t n6 = (k + 1) * ni * nj + (j + 1) * ni + (i + 1) + 1;
                const label_t n7 = (k + 1) * ni * nj + (j + 1) * ni + i + 1;

                outFile << n0 << " " << n1 << " " << n2 << " " << n3 << " " << n4 << " " << n5 << " " << n6 << " " << n7 << "\n";
            }
        }
    }

    outFile.close();
    std::cout << "Successfully wrote Tecplot file: " << filename << "\n";
}

__host__ void interfaceFree(ghostInterfaceData &ghostInterface)
{
    cudaFree(ghostInterface.fGhost.X_0);
    cudaFree(ghostInterface.fGhost.X_1);
    cudaFree(ghostInterface.fGhost.Y_0);
    cudaFree(ghostInterface.fGhost.Y_1);
    cudaFree(ghostInterface.fGhost.Z_0);
    cudaFree(ghostInterface.fGhost.Z_1);

    cudaFree(ghostInterface.gGhost.X_0);
    cudaFree(ghostInterface.gGhost.X_1);
    cudaFree(ghostInterface.gGhost.Y_0);
    cudaFree(ghostInterface.gGhost.Y_1);
    cudaFree(ghostInterface.gGhost.Z_0);
    cudaFree(ghostInterface.gGhost.Z_1);
}

[[nodiscard]] __host__ const std::vector<scalar_t> save(const scalar_t *const h_fMom)
{
    std::vector<scalar_t> f(NX * NY * NZ, 0);

    for (label_t z = 0; z < NZ; z++)
    {
        for (label_t y = 0; y < NY; y++)
        {
            for (label_t x = 0; x < NX; x++)
            {
                f[idxScalarGlobal(x, y, z)] = h_fMom[idxMom__<M_UX_INDEX>(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)];
            }
        }
    }

    return f;
}

// int main(int argc, char *argv[])
int main(void)
{

    // const programControl programCtrl(argc, argv);
    // const host::latticeMesh mesh(ctorType::MUST_READ);
    // const host::moments moms(mesh, ctorType::MUST_READ);
    // const host::ghostInterface interface(moms);
    // const device::nodeTypeArray nodeTypes(mesh.nodeTypes());

    // Create an interlaced array for the moments
    // const std::vector<scalar_t> d_mom = device::interlaceVectors(
    //     moms.rho().arrRef(),
    //     moms.u().arrRef(),
    //     moms.v().arrRef(),
    //     moms.w().arrRef(),
    //     moms.m_xx().arrRef(),
    //     moms.m_xy().arrRef(),
    //     moms.m_xz().arrRef(),
    //     moms.m_yy().arrRef(),
    //     moms.m_yz().arrRef(),
    //     moms.m_zz().arrRef());

    // const host::array moments(
    //     mesh,
    //     device::interlaceVectors(
    //         moms.rho().arrRef(),
    //         moms.u().arrRef(),
    //         moms.v().arrRef(),
    //         moms.w().arrRef(),
    //         moms.m_xx().arrRef(),
    //         moms.m_xy().arrRef(),
    //         moms.m_xz().arrRef(),
    //         moms.m_yy().arrRef(),
    //         moms.m_yz().arrRef(),
    //         moms.m_zz().arrRef()),
    //     "moments");

    // // for (std::size_t i = 0; i < d_mom.size(); i++)
    // // {
    // //     if (d_mom[i] > 0)
    // //     {
    // //         std::cout << "i = " << i << std::endl;
    // //         break;
    // //     }
    // // }

    // // Move the interlaced vector to the device and back, then output it to a .dat file
    // device::scalarArray devMoments(moments);

    // {
    //     const std::vector<scalar_t> f_start = deviceToHost(mesh, devMoments.ptr(), 10);

    //     std::vector<scalar_t> u_start(mesh.nPoints(), 0);
    //     for (std::size_t i = 0; i < mesh.nPoints(); i++)
    //     {
    //         u_start[i] = f_start[(i * 10) + 1];
    //     }

    //     writeTecplotHexahedralData(
    //         u_start,
    //         "u_start.dat",
    //         mesh.nx(), mesh.ny(), mesh.nz(),
    //         "Title", {"u"});
    // }

    // // scalar_t *moments = device::allocateArray(d_mom);

    // // // scalar_t *moments = device::allocate<scalar_t>(mesh.nPoints() * 10);
    // // // cudaFree(moments);

    // // // const label_t n = mesh.nx() * mesh.ny() * mesh.nxBlocks() * mesh.nyBlocks() * mesh.nzBlocks() * VelocitySet::D3Q19::QF();
    // // // std::vector<scalar_t> f(n, 0);
    // // // std::cout << "n = " << n << std::endl;

    // const nGhostFace nFaces(mesh);

    // // // device::scalarArray moments(n, 0);

    // device::scalarArray f_x0(nFaces.yz, 0);
    // device::scalarArray f_x1(nFaces.yz, 0);
    // device::scalarArray f_y0(nFaces.xz, 0);
    // device::scalarArray f_y1(nFaces.xz, 0);
    // device::scalarArray f_z0(nFaces.xy, 0);
    // device::scalarArray f_z1(nFaces.xy, 0);

    // device::scalarArray g_x0(nFaces.yz, 0);
    // device::scalarArray g_x1(nFaces.yz, 0);
    // device::scalarArray g_y0(nFaces.xz, 0);
    // device::scalarArray g_y1(nFaces.xz, 0);
    // device::scalarArray g_z0(nFaces.xy, 0);
    // device::scalarArray g_z1(nFaces.xy, 0);

    // // // // writeTecplotHexahedralData(
    // // // //     moms.u().arrRef(),
    // // // //     "host_u_0.dat",
    // // // //     mesh.nx(), mesh.ny(), mesh.nz(),
    // // // //     "Title", {"u"});

    // // // // VelocitySet::D3Q19::print();

    // // // // device::ghostInterface devInterface(interface);
    // // // // device::moments devMoments(programCtrl.deviceList()[0], moms);

    // // // // const deviceConstants devConsts(mesh, programCtrl);

    // // // // devConstantsTest<<<1, 1, 0, 0>>>();

    // // // // print(block::threadBlock(), "threadBlock");
    // // // // print(block::gridBlock(mesh), "gridBlock");

    // // // // // device::scalarArray d_mom(mesh.nPoints() * 10, 0);

    // label_t timeStep = 0;
    // for (timeStep = 0; timeStep < 10001; timeStep++)
    // {
    //     // Do the output
    //     if ((timeStep % 100) == 0)
    //     {
    //         std::cout << "Time: " << timeStep << std::endl;

    //         checkCudaErrors(cudaDeviceSynchronize());
    //     }

    //     kernel_collide<<<block::gridBlock(mesh), block::threadBlock(), 0, 0>>>(
    //         devMoments.ptr(),
    //         nodeTypes.ptr(),
    //         f_x0.ptr(), f_x1.ptr(),
    //         f_y0.ptr(), f_y1.ptr(),
    //         f_z0.ptr(), f_z1.ptr(),
    //         g_x0.ptr(), g_x1.ptr(),
    //         g_y0.ptr(), g_y1.ptr(),
    //         g_z0.ptr(), g_z1.ptr());

    //     cudaStreamSynchronize(0);

    //     // devInterface.swap();
    //     checkCudaErrors(cudaDeviceSynchronize());

    //     std::swap(f_x0.ptrRef(), g_x0.ptrRef());
    //     std::swap(f_x1.ptrRef(), g_x1.ptrRef());
    //     std::swap(f_y0.ptrRef(), g_y0.ptrRef());
    //     std::swap(f_y1.ptrRef(), g_y1.ptrRef());
    //     std::swap(f_z0.ptrRef(), g_z0.ptrRef());
    //     std::swap(f_z1.ptrRef(), g_z1.ptrRef());
    // }

    // {
    //     const std::vector<scalar_t> f_end = deviceToHost(mesh, devMoments.ptr(), 10);

    //     std::vector<scalar_t> u_end(mesh.nPoints(), 0);
    //     for (std::size_t i = 0; i < mesh.nPoints(); i++)
    //     {
    //         u_end[i] = f_end[(i * 10) + 1];
    //     }

    //     writeTecplotHexahedralData(
    //         u_end,
    //         "u_end.dat",
    //         mesh.nx(), mesh.ny(), mesh.nz(),
    //         "Title", {"u"});
    // }

    // // checkCudaErrors(cudaDeviceSynchronize());

    // Setup saving folder
    // folderSetup();

    // set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // variable declaration
    scalar_t *d_fMom;
    ghostInterfaceData ghostInterface;
    unsigned int *dNodeType;
    unsigned int *hNodeType;
    scalar_t *h_fMom;
    scalar_t *rho;
    scalar_t *ux;
    scalar_t *uy;
    scalar_t *uz;

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    constexpr const dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    constexpr const dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    std::cout << "domain: " << NX << " " << NY << " " << NZ << std::endl;
    std::cout << "threadBlock: " << BLOCK_NX << " " << BLOCK_NY << " " << BLOCK_NZ << std::endl;
    std::cout << "gridBlock: " << NUM_BLOCK_X << " " << NUM_BLOCK_Y << " " << NUM_BLOCK_Z << std::endl;
    std::cout << "OMEGA = " << OMEGA << std::endl;
    std::cout << "VISC = " << VISC << std::endl;

    scalar_t **randomNumbers = nullptr; // useful for turbulence
    randomNumbers = (scalar_t **)malloc(sizeof(scalar_t *));

    allocateHostMemory(&h_fMom, &rho, &ux, &uy, &uz);
    allocateDeviceMemory(&d_fMom, &dNodeType, &ghostInterface);

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    initializeDomain(
        ghostInterface,
        d_fMom, h_fMom,
        hNodeType, dNodeType, randomNumbers,
        gridBlock, threadBlock);

    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    constexpr const label_t INI_STEP = 0;
    constexpr const label_t N_STEPS = 1001;

    for (label_t step = INI_STEP; step < N_STEPS; step++)
    {

        if ((step % 100) == 0)
        {
            std::cout << "Time: " << step << std::endl;
        }

        gpuMomCollisionStream<<<gridBlock, threadBlock, 0, 0>>>(d_fMom, dNodeType, ghostInterface);

        // swap interface pointers
        swapGhostInterfaces(ghostInterface);
    }

    std::cout << "Exited main loop" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    {
        checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

        writeTecplotHexahedralData(
            save(h_fMom),
            "u_end.dat",
            NX, NY, NZ,
            "Title", {"u"});
    }

    // {
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     const std::vector<scalar_t> f_end = deviceToHost(d_fMom, 10);
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     std::vector<scalar_t> u_end(NUMBER_LBM_POP_NODES, 0);
    //     for (std::size_t i = 0; i < NUMBER_LBM_POP_NODES; i++)
    //     {
    //         u_end[i] = f_end[(i * 10) + 1];
    //     }

    //     writeTecplotHexahedralData(
    //         u_end,
    //         "u_end.dat",
    //         NX, NY, NZ,
    //         "Title", {"u"});
    // }

    // checkCudaErrors(cudaDeviceSynchronize());

    // /* --------------------------------------------------------------------- */
    // /* ------------------------------ END LOO ------------------------------ */
    // /* --------------------------------------------------------------------- */

    // checkCudaErrors(cudaDeviceSynchronize());

    // // Calculate MLUPS

    // scalar_t MLUPS = recordElapsedTime(start_step, stop_step, step);
    // printf("MLUPS: %f\n", MLUPS);

    // /* ------------------------------ POST ------------------------------ */
    // checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

    // if (console_flush)
    // {
    //     fflush(stdout);
    // }

    // // saveMacr(h_fMom, rho, ux, uy, uz, NON_NEWTONIAN_FLUID_PARAMS
    // //          NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(PREFIX) step);

    // if (CHECKPOINT_SAVE)
    // {
    //     printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
    //     fflush(stdout);

    //     checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    //     interfaceCudaMemcpy(ghostInterface, ghostInterface.h_fGhost, ghostInterface.gGhost, cudaMemcpyDeviceToHost, QF);
    //     saveSimCheckpoint(d_fMom, ghostInterface, &step);
    // }
    // checkCudaErrors(cudaDeviceSynchronize());

    // // save info file
    // saveSimInfo(step, MLUPS);

    // /* ------------------------------ FREE ------------------------------ */
    // cudaFree(&d_fMom);
    // cudaFree(&dNodeType);
    // cudaFree(&hNodeType);

    // cudaFree(&h_fMom);
    // cudaFree(&rho);
    // cudaFree(&ux);
    // cudaFree(&uy);
    // cudaFree(&uz);

    // interfaceFree(ghostInterface);

    return 0;
}