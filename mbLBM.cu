#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
// #include "latticeMesh/latticeMesh.cuh"
#include "array/array.cuh"
// #include "programControl.cuh"
// #include "inputControl.cuh"
// #include "latticeMesh/latticeMesh.cuh"
// #include "moments/moments.cuh"
// #include "collision.cuh"
// #include "postProcess.cuh"

using namespace mbLBM;

constexpr const label_t GPU_INDEX = 1;

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

    // set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // Perform device memory allocation
    deviceArray<scalar_t> d_fMom(NUMBER_LBM_NODES * NUMBER_MOMENTS);
    deviceArray<unsigned int> dNodeType(NUMBER_LBM_NODES);

    // scalar_t *const d_fMom = device::allocate<scalar_t>(NUMBER_LBM_NODES * NUMBER_MOMENTS);
    // unsigned int *dNodeType = device::allocate<unsigned int>(NUMBER_LBM_NODES);
    ghostInterfaceData_t ghostInterface;

    // unsigned int *dNodeType;
    unsigned int *hNodeType;
    scalar_t *h_fMom;
    scalar_t *rho;
    scalar_t *ux;
    scalar_t *uy;
    scalar_t *uz;
    allocateHostMemory(&h_fMom, &rho, &ux, &uy, &uz);

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    constexpr const dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    constexpr const dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    std::cout << "domain: " << NX << " " << NY << " " << NZ << std::endl;
    std::cout << "threadBlock: " << BLOCK_NX << " " << BLOCK_NY << " " << BLOCK_NZ << std::endl;
    std::cout << "gridBlock: " << NUM_BLOCK_X << " " << NUM_BLOCK_Y << " " << NUM_BLOCK_Z << std::endl;
    std::cout << "OMEGA = " << OMEGA << std::endl;
    std::cout << "VISC = " << VISC << std::endl;

    // allocateDeviceMemory(&ghostInterface);

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    initializeDomain(
        ghostInterface,
        d_fMom.ptr(), h_fMom,
        hNodeType, dNodeType.ptr(),
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

        gpuMomCollisionStream<<<gridBlock, threadBlock, 0, 0>>>(
            d_fMom.ptr(),
            dNodeType.ptr(),
            ghostInterface);

        ghostInterface.swap();
    }

    std::cout << "Exited main loop" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    {
        checkCudaErrors(cudaMemcpy(h_fMom, d_fMom.ptr(), sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

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