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
    along with this program. If not, see <https://www.gnu.org/licenses/>.

Description
    VTS binary file writer

Namespace
    LBM::postProcess

SourceFiles
    VTS.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VTS_CUH
#define __MBLBM_VTS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Auxiliary template function that performs the VTU file writing.
         */
        __host__ void VTSWriter(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const std::string &fileName,
            const host::latticeMesh &mesh,
            const std::vector<std::string> &solutionVarNames) noexcept
        {
            // For a structured grid, we need different calculations
            // const label_t numNodes = mesh.nx() * mesh.ny() * mesh.nz();
            // const label_t numCells = (mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);
            const std::size_t numVars = solutionVars.size();

            // Get points in the correct order for structured grid (i fastest, then j, then k)
            const std::vector<scalar_t> points = meshCoordinates<scalar_t>(mesh);

            std::ofstream outFile(fileName, std::ios::binary);
            if (!outFile)
            {
                std::cerr << "Error opening file " << fileName << "\n";
                return;
            }

            std::stringstream xml;
            uint64_t currentOffset = 0;

            // Calculate extents - note the -1 for the maximum indices
            const label_t dimX = mesh.nx() - 1;
            const label_t dimY = mesh.ny() - 1;
            const label_t dimZ = mesh.nz() - 1;

            xml << "<?xml version=\"1.0\"?>\n";
            xml << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
            xml << "  <StructuredGrid WholeExtent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";
            xml << "    <Piece Extent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";

            // Point data (same as before)
            xml << "      <PointData Scalars=\"" << (solutionVarNames.empty() ? "" : solutionVarNames[0]) << "\">\n";
            for (std::size_t i = 0; i < numVars; ++i)
            {
                xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"" << solutionVarNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                currentOffset += sizeof(uint64_t) + solutionVars[i].size() * sizeof(scalar_t);
            }
            xml << "      </PointData>\n";

            // Points section (same as before)
            xml << "      <Points>\n";
            xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"Coordinates\" NumberOfComponents=\"" << 3 << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            xml << "      </Points>\n";
            currentOffset += sizeof(uint64_t) + points.size() * sizeof(scalar_t);

            // NO Cells section for StructuredGrid - this is the key difference!

            xml << "    </Piece>\n";
            xml << "  </StructuredGrid>\n";
            xml << "  <AppendedData encoding=\"raw\">_";

            outFile << xml.str();

            // Write point data arrays
            for (const auto &varData : solutionVars)
            {
                writeBinaryBlock(varData, outFile);
            }

            // Write points
            writeBinaryBlock(points, outFile);

            outFile << "</AppendedData>\n";
            outFile << "</VTKFile>\n";

            outFile.close();
            std::cout << "Successfully wrote structured VTU file: " << fileName << "\n";
        }

        /**
         * @brief Writes solution variables to an unstructured grid VTU file (.vtu)
         * This function checks the mesh size and dispatches to the implementation with
         * the appropriate index type (32-bit or 64-bit).
         */
        __host__ void writeVTS(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const std::string &fileName,
            const host::latticeMesh &mesh,
            const std::vector<std::string> &solutionVarNames) noexcept
        {
            const std::string fileExtension = ".vts";

            std::cout << "Writing VTS structured grid to " << fileName << fileExtension << std::endl;

            const uint64_t numNodes = static_cast<uint64_t>(mesh.nx()) * static_cast<uint64_t>(mesh.ny()) * static_cast<uint64_t>(mesh.nz());
            const std::size_t numVars = solutionVars.size();

            if (numVars != solutionVarNames.size())
            {
                std::cerr << "Error: The number of solution (" << numVars << ") does not match the count of variable names (" << solutionVarNames.size() << ")\n";
                return;
            }
            for (std::size_t i = 0; i < numVars; i++)
            {
                if (solutionVars[i].size() != numNodes)
                {
                    std::cerr << "Error: The solution variable " << i << " has " << solutionVars[i].size() << " elements, expected " << numNodes << "\n";
                    return;
                }
            }

            VTSWriter(solutionVars, fileName + fileExtension, mesh, solutionVarNames);
        }
    }
}

#endif