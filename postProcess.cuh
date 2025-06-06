#ifndef __MBLBM_POSTPROCESS_CUH
#define __MBLBM_POSTPROCESS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    template <typename T>
    [[nodiscard]] const std::vector<T> deviceToHost(const T *f, const label_t nFields = 1) noexcept
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

    template <const label_t variableIndex>
    [[nodiscard]] __host__ const std::vector<scalar_t> save(const scalar_t *const h_fMom) noexcept
    {
        std::vector<scalar_t> f(NX * NY * NZ, 0);

        for (label_t z = 0; z < NZ; z++)
        {
            for (label_t y = 0; y < NY; y++)
            {
                for (label_t x = 0; x < NX; x++)
                {
                    f[idxScalarGlobal(x, y, z)] = h_fMom[idxMom<variableIndex>(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)];
                }
            }
        }

        return f;
    }

    void writeTecplotHexahedralData(
        const std::vector<scalar_t> &solutionData,
        const std::string &filename,
        const label_t ni,
        const label_t nj,
        const label_t nk,
        const std::string &title,
        const std::vector<std::string> &solutionVarNames) noexcept
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
}

#endif