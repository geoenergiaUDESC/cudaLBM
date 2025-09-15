/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Tecplot ASCII file writer

Namespace
    LBM::postProcess

SourceFiles
    Tecplot.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_TECPLOT_CUH
#define __MBLBM_TECPLOT_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Writes solution data to a Tecplot ASCII file in unstructured grid format
         * @param[in] solutionVars Vector of solution variable arrays (Structure of Arrays format)
         * @param[in] fileName Output filename for Tecplot data
         * @param[in] mesh Lattice mesh providing domain dimensions and structure
         * @param[in] solutionVarNames Names of the solution variables for Tecplot header
         * @param[in] title Title for the Tecplot file
         * @return None
         * @note Uses 1-based indexing for element connectivity (Tecplot convention)
         * @note Output format: BLOCK data packing with FEBRICK (hexahedral) elements
         * @note Uses high precision (50 digits) for numerical output
         *
         * This function writes simulation results to a Tecplot-compatible ASCII file
         * with the following structure:
         * 1. File header with title and variable declarations
         * 2. Coordinate data in separate blocks (X, Y, Z)
         * 3. Solution variables in separate blocks
         * 4. Element connectivity with 1-based indexing
         *
         * The function performs comprehensive validation of input data including:
         * - Variable count matching name count
         * - Node count consistency across all arrays
         * - File accessibility checks
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

            const std::vector<label_t> connectivity = meshConnectivity<true, label_t>(mesh);
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