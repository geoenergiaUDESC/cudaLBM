#ifndef __MBLBM_POSTPROCESS_CUH
#define __MBLBM_POSTPROCESS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Writes a solution variable to a file
         * @param solutionVars An std::vector of std::vectors containing the solution variable to be written
         * @param fileName The name of the file to be written
         * @param mesh The mesh
         * @param title Title of the file
         * @param solutionVarNames Names of the solution variables
         **/
        void writeTecplotHexahedralData(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const std::string &fileName,
            const host::latticeMesh &mesh,
            const std::vector<std::string> &solutionVarNames,
            const std::string &title) noexcept
        {
            // Check input sizes
            const label_t numNodes = mesh.nx() * mesh.ny() * mesh.nz();
            const size_t numVars = solutionVars.size();

            // Validate variable count matches names
            if (numVars != solutionVarNames.size())
            {
                std::cerr << "Error: Number of solution variables (" << numVars
                          << ") doesn't match variable name count ("
                          << solutionVarNames.size() << ")\n";
                return;
            }

            // Validate each variable has correct number of nodes
            for (size_t i = 0; i < numVars; i++)
            {
                if (solutionVars[i].size() != numNodes)
                {
                    std::cerr << "Error: Solution variable " << i << " has "
                              << solutionVars[i].size() << " elements, expected "
                              << numNodes << "\n";
                    return;
                }
            }

            std::ofstream outFile(fileName);
            if (!outFile)
            {
                std::cerr << "Error opening file: " << fileName << "\n";
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

            // UNSTRUCTURED GRID FORMAT
            const label_t numElements = (mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);
            outFile << "ZONE T=\"Hexahedral Zone\", NODES=" << numNodes
                    << ", ELEMENTS=" << numElements
                    << ", DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n";

            // Write node coordinates (X, Y, Z blocks)
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

            // Write solution variables (each as a separate block)
            for (const auto &varData : solutionVars)
            {
                for (const auto &value : varData)
                {
                    outFile << value << "\n";
                }
            }

            // Write connectivity (1-based indexing)
            for (label_t k = 0; k < mesh.nz() - 1; k++)
            {
                for (label_t j = 0; j < mesh.ny() - 1; j++)
                {
                    for (label_t i = 0; i < mesh.nx() - 1; i++)
                    {
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
            std::cout << "Successfully wrote Tecplot file: " << fileName << "\n";
        }
    }
}

#endif