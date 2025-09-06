#ifndef __MBLBM_TECPLOT_CUH
#define __MBLBM_TECPLOT_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

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
        void writeTecplot(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const std::string &fileName,
            const host::latticeMesh &mesh,
            const std::vector<std::string> &solutionVarNames,
            const std::string &title) noexcept
        {
            // Info on entering the function
            std::cout << "Writing tecplot unstructured grid to file" << fileName << std::endl;

            // Check input sizes
            const label_t numNodes = mesh.nx() * mesh.ny() * mesh.nz();
            const size_t numVars = solutionVars.size();

            // Validate variable count matches names
            if (numVars != solutionVarNames.size())
            {
                std::cerr << "Error: Number of solution variables (" << numVars << ") doesn't match variable name count (" << solutionVarNames.size() << ")\n";
                return;
            }

            // Validate each variable has correct number of nodes
            for (size_t i = 0; i < numVars; i++)
            {
                if (solutionVars[i].size() != numNodes)
                {
                    std::cerr << "Error: Solution variable " << i << " has " << solutionVars[i].size() << " elements, expected " << numNodes << "\n";
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
            outFile << std::setprecision(50);

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
            outFile << "ZONE T=\"Hexahedral Zone\", NODES=" << numNodes << ", ELEMENTS=" << numElements << ", DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n";

            const std::vector<double> coords = meshCoordinates<double>(mesh);

            // Write node coordinates (X, Y, Z blocks)
            // Write X
            for (label_t n = 0; n < numNodes; ++n)
            {
                outFile << coords[3 * n + 0] << "\n";
            }

            // Write Y
            for (label_t n = 0; n < numNodes; ++n)
            {
                outFile << coords[3 * n + 1] << "\n";
            }

            // Write Z
            for (label_t n = 0; n < numNodes; ++n)
            {
                outFile << coords[3 * n + 2] << "\n";
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
            const std::vector<label_t> connectivity = meshConnectivity<true>(mesh);
            for (label_t e = 0; e < numElements; ++e)
            {
                for (label_t n = 0; n < 8; ++n)
                {
                    outFile << connectivity[e * 8 + n] << (n < 7 ? " " : "\n");
                }
            }

            outFile.close();

            std::cout << "Successfully wrote Tecplot file: " << fileName << "\n";
        }
    }
}

#endif