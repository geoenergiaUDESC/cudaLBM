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
    VTU binary file writer

Namespace
    LBM::postProcess

SourceFiles
    VTU.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VTU_CUH
#define __MBLBM_VTU_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Auxiliary template function that performs the VTU file writing.
         * @tparam IndexType The data type for the mesh indices (uint32_t or uint64_t).
         */
        template <typename IndexType>
        __host__ void VTUWriter(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const std::string &fileName,
            const host::latticeMesh &mesh,
            const std::vector<std::string> &solutionVarNames) noexcept
        {
            const label_t numNodes = mesh.nx() * mesh.ny() * mesh.nz();
            const label_t numElements = (mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);
            const std::size_t numVars = solutionVars.size();

            const std::vector<scalar_t> points = meshCoordinates<scalar_t>(mesh);
            const std::vector<IndexType> connectivity = meshConnectivity<false, IndexType>(mesh);
            const std::vector<IndexType> offsets = meshOffsets<IndexType>(mesh);

            std::ofstream outFile(fileName, std::ios::binary);
            if (!outFile)
            {
                std::cerr << "Error opening file " << fileName << "\n";
                return;
            }

            std::stringstream xml;
            uint64_t currentOffset = 0;

            xml << "<?xml version=\"1.0\"?>\n";
            xml << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
            xml << "  <UnstructuredGrid>\n";
            xml << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numElements << "\">\n";

            xml << "      <PointData Scalars=\"" << (solutionVarNames.empty() ? "" : solutionVarNames[0]) << "\">\n";
            for (std::size_t i = 0; i < numVars; ++i)
            {
                xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"" << solutionVarNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                currentOffset += sizeof(uint64_t) + solutionVars[i].size() * sizeof(scalar_t);
            }
            xml << "      </PointData>\n";

            xml << "      <Points>\n";
            xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            xml << "      </Points>\n";
            currentOffset += sizeof(uint64_t) + points.size() * sizeof(scalar_t);

            xml << "      <Cells>\n";
            // Usa o IndexType para obter o nome do tipo VTK correto
            xml << "        <DataArray type=\"" << getVtkTypeName<IndexType>() << "\" Name=\"connectivity\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            currentOffset += sizeof(uint64_t) + connectivity.size() * sizeof(IndexType);

            xml << "        <DataArray type=\"" << getVtkTypeName<IndexType>() << "\" Name=\"offsets\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            currentOffset += sizeof(uint64_t) + offsets.size() * sizeof(IndexType);

            xml << "        <DataArray type=\"" << getVtkTypeName<uint8_t>() << "\" Name=\"types\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            xml << "      </Cells>\n";

            xml << "    </Piece>\n";
            xml << "  </UnstructuredGrid>\n";
            xml << "  <AppendedData encoding=\"raw\">_";

            outFile << xml.str();

            for (const auto &varData : solutionVars)
            {
                writeBinaryBlock(varData, outFile);
            }
            writeBinaryBlock(points, outFile);
            writeBinaryBlock(connectivity, outFile);
            writeBinaryBlock(offsets, outFile);

            const std::vector<uint8_t> types(numElements, 12); // 12 é o código VTK para hexaedro
            writeBinaryBlock(types, outFile);

            outFile << "</AppendedData>\n";
            outFile << "</VTKFile>\n";

            outFile.close();
            std::cout << "Successfully wrote VTU file: " << fileName << "\n";
        }

        /**
         * @brief Writes solution variables to an unstructured grid VTU file (.vtu)
         * This function checks the mesh size and dispatches to the implementation with
         * the appropriate index type (32-bit or 64-bit).
         */
        __host__ void writeVTU(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const std::string &fileName,
            const host::latticeMesh &mesh,
            const std::vector<std::string> &solutionVarNames) noexcept
        {
            const std::string fileExtension = ".vtu";

            const std::string directoryPrefix = "postProcess";

            if (!std::filesystem::is_directory(directoryPrefix))
            {
                if (!std::filesystem::create_directory(directoryPrefix))
                {
                    std::cout << "Could not create directory: " + directoryPrefix << std::endl;
                    // throw std::runtime_error("Could not create directory: " + directoryPrefix);
                }
            }

            std::cout << "Writing VTU unstructured grid to " << directoryPrefix << "/" << fileName << fileExtension << std::endl;

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

            constexpr const uint64_t limit32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

            if (numNodes >= limit32)
            {
                std::cout << "Info: Mesh is large. Using 64-bit indices for VTU file.\n";
                VTUWriter<uint64_t>(solutionVars, directoryPrefix + "/" + fileName + fileExtension, mesh, solutionVarNames);
            }
            else
            {
                std::cout << "Info: Mesh is small. Using 32-bit indices for VTU file.\n";
                VTUWriter<uint32_t>(solutionVars, directoryPrefix + "/" + fileName + fileExtension, mesh, solutionVarNames);
            }
        }
    }
}

#endif