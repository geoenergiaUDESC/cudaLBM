#ifndef __MBLBM_POSTPROCESS_CUH
#define __MBLBM_POSTPROCESS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    /**
     * @brief Copies a device variable to the host
     * @param fMom The device variable array
     * @param mesh The mesh
     * @return An std::vector of type T, de-interlaced from fMom
     **/
    template <const label_t variableIndex, typename T>
    [[nodiscard]] __host__ const std::vector<T> save(
        const T *const fMom,
        const host::latticeMesh &mesh) noexcept
    {
        std::vector<T> f(mesh.nx() * mesh.ny() * mesh.nz(), 0);

        for (label_t z = 0; z < mesh.nz(); z++)
        {
            for (label_t y = 0; y < mesh.ny(); y++)
            {
                for (label_t x = 0; x < mesh.nx(); x++)
                {
                    f[host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = fMom[host::idxMom<variableIndex>(x % block::nx(), y % block::ny(), z % block::nz(), x / block::nx(), y / block::ny(), z / block::nz(), mesh.nxBlocks(), mesh.nyBlocks())];
                }
            }
        }

        return f;
    }

    /**
     * @brief Writes a solution variable to a file
     * @param solutionData An std::vector containing the solution variable to be written
     * @param fileName The name of the file to be written
     * @param mesh The mesh
     * @param title Title of the file
     * @param solutionVarNames Names of the solution variables
     **/
    void writeTecplotHexahedralData(
        const std::vector<scalar_t> &solutionData,
        const std::string &filename,
        const host::latticeMesh &mesh,
        const std::string &title,
        const std::vector<std::string> &solutionVarNames) noexcept
    {
        // Check input sizes
        const label_t numNodes = mesh.nx() * mesh.ny() * mesh.nz();
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
        const label_t numElements = (mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);
        outFile << "ZONE T=\"Hexahedral Zone\", NODES=" << numNodes << ", ELEMENTS=" << numElements << ", DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n";

        // Write all node coordinates first
        // X coordinates
        for (label_t k = 0; k < mesh.nz(); k++)
        {
            for (label_t j = 0; j < mesh.ny(); j++)
            {
                for (label_t i = 0; i < mesh.nx(); i++)
                {
                    outFile << static_cast<double>(i) / static_cast<double>(mesh.nx() - 1) << "\n";
                }
            }
        }

        // Y coordinates
        for (label_t k = 0; k < mesh.nz(); k++)
        {
            for (label_t j = 0; j < mesh.ny(); j++)
            {
                for (label_t i = 0; i < mesh.nx(); i++)
                {
                    outFile << static_cast<double>(j) / static_cast<double>(mesh.ny() - 1) << "\n";
                }
            }
        }

        // Z coordinates
        for (label_t k = 0; k < mesh.nz(); k++)
        {
            for (label_t j = 0; j < mesh.ny(); j++)
            {
                for (label_t i = 0; i < mesh.nx(); i++)
                {
                    outFile << static_cast<double>(k) / static_cast<double>(mesh.nz() - 1) << "\n";
                }
            }
        }

        // Write solution variable
        for (label_t i = 0; i < numNodes; i++)
        {
            outFile << solutionData[i] << "\n";
        }

        // Write connectivity (1-based indexing)
        for (label_t k = 0; k < mesh.nz() - 1; k++)
        {
            for (label_t j = 0; j < mesh.ny() - 1; j++)
            {
                for (label_t i = 0; i < mesh.nx() - 1; i++)
                {
                    // Get the 8 nodes of the hexahedron
                    const label_t n0 = k * mesh.nx() * mesh.ny() + j * mesh.nx() + i + 1;
                    const label_t n1 = k * mesh.nx() * mesh.ny() + j * mesh.nx() + (i + 1) + 1;
                    const label_t n2 = k * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + (i + 1) + 1;
                    const label_t n3 = k * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + i + 1;
                    const label_t n4 = (k + 1) * mesh.nx() * mesh.ny() + j * mesh.nx() + i + 1;
                    const label_t n5 = (k + 1) * mesh.nx() * mesh.ny() + j * mesh.nx() + (i + 1) + 1;
                    const label_t n6 = (k + 1) * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + (i + 1) + 1;
                    const label_t n7 = (k + 1) * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + i + 1;

                    outFile << n0 << " " << n1 << " " << n2 << " " << n3 << " " << n4 << " " << n5 << " " << n6 << " " << n7 << "\n";
                }
            }
        }

        outFile.close();
        std::cout << "Successfully wrote Tecplot file: " << filename << "\n";
    }
}

#endif