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

        /**
         * @brief Obtain the name of the type that corresponds to the C++ data type
         * @tparam T The C++ data type (e.g. float, int64_t)
         * @return A string containing the name of the VTK type (e.g. "Float32", "Int64")
         **/
        template <typename T>
        [[nodiscard]] inline constexpr std::string getVtkTypeName() noexcept
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return "Float32";
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return "Float64";
            }
            else if constexpr (std::is_same_v<T, int32_t>)
            {
                return "Int32";
            }
            else if constexpr (std::is_same_v<T, uint32_t>)
            {
                return "UInt32";
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                return "Int64";
            }
            else if constexpr (std::is_same_v<T, uint64_t>)
            {
                return "UInt64";
            }
            else if constexpr (std::is_same_v<T, uint8_t>)
            {
                return "UInt8";
            }
            else if constexpr (std::is_same_v<T, int8_t>)
            {
                return "Int8";
            }
            else
            {
                static_assert(std::is_same_v<T, void>, "Unsupported type for getVtkTypeName");
                return "Unknown";
            }
        }

        /**
         * @brief Escreve variáveis de solução para um arquivo de grade não estruturada VTU (.vtu) - VERSÃO CORRIGIDA
         * @param solutionVars Um std::vector de std::vectors contendo a variável de solução a ser escrita
         * @param fileName O nome do arquivo a ser escrito (.vtu)
         * @param mesh A malha
         * @param solutionVarNames Nomes das variáveis de solução
         **/
        void writeVTU(
            const std::vector<std::vector<scalar_t>> &solutionVars,
            const std::string &fileName,
            const host::latticeMesh &mesh,
            const std::vector<std::string> &solutionVarNames,
            [[maybe_unused]] const std::string &title = "") noexcept
        {
            // Info on entering the function
            std::cout << "Writing VTU unstructured grid to file" << fileName << std::endl;

            const label_t numNodes = mesh.nx() * mesh.ny() * mesh.nz();
            const label_t numElements = (mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);
            const size_t numVars = solutionVars.size();

            if (numVars != solutionVarNames.size())
            {
                std::cerr << "Erro: O número de variáveis de solução (" << numVars << ") não corresponde à contagem de nomes de variáveis (" << solutionVarNames.size() << ")\n";
                return;
            }

            for (size_t i = 0; i < numVars; i++)
            {
                if (solutionVars[i].size() != numNodes)
                {
                    std::cerr << "Erro: A variável de solução " << i << " tem " << solutionVars[i].size() << " elementos, esperado " << numNodes << "\n";
                    return;
                }
            }

            std::vector<scalar_t> points(numNodes * 3);
            for (label_t k = 0; k < mesh.nz(); ++k)
            {
                for (label_t j = 0; j < mesh.ny(); ++j)
                {
                    for (label_t i = 0; i < mesh.nx(); ++i)
                    {
                        const label_t index = k * mesh.nx() * mesh.ny() + j * mesh.nx() + i;
                        points[index * 3 + 0] = static_cast<scalar_t>(i) / static_cast<scalar_t>(mesh.nx() - 1);
                        points[index * 3 + 1] = static_cast<scalar_t>(j) / static_cast<scalar_t>(mesh.ny() - 1);
                        points[index * 3 + 2] = static_cast<scalar_t>(k) / static_cast<scalar_t>(mesh.nz() - 1);
                    }
                }
            }

            std::vector<label_t> connectivity(numElements * 8);
            label_t cell_idx = 0;
            for (label_t k = 0; k < mesh.nz() - 1; ++k)
            {
                for (label_t j = 0; j < mesh.ny() - 1; ++j)
                {
                    for (label_t i = 0; i < mesh.nx() - 1; ++i)
                    {
                        connectivity[cell_idx * 8 + 0] = k * mesh.nx() * mesh.ny() + j * mesh.nx() + i;
                        connectivity[cell_idx * 8 + 1] = k * mesh.nx() * mesh.ny() + j * mesh.nx() + (i + 1);
                        connectivity[cell_idx * 8 + 2] = k * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + (i + 1);
                        connectivity[cell_idx * 8 + 3] = k * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + i;
                        connectivity[cell_idx * 8 + 4] = (k + 1) * mesh.nx() * mesh.ny() + j * mesh.nx() + i;
                        connectivity[cell_idx * 8 + 5] = (k + 1) * mesh.nx() * mesh.ny() + j * mesh.nx() + (i + 1);
                        connectivity[cell_idx * 8 + 6] = (k + 1) * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + (i + 1);
                        connectivity[cell_idx * 8 + 7] = (k + 1) * mesh.nx() * mesh.ny() + (j + 1) * mesh.nx() + i;
                        cell_idx++;
                    }
                }
            }

            std::vector<label_t> offsets(numElements);
            for (label_t i = 0; i < numElements; ++i)
            {
                offsets[i] = (i + 1) * 8;
            }

            std::ofstream outFile(fileName, std::ios::binary);
            if (!outFile)
            {
                std::cerr << "Erro ao abrir o arquivo: " << fileName << "\n";
                return;
            }

            std::stringstream xml;
            uint64_t currentOffset = 0;

            xml << "<?xml version=\"1.0\"?>\n";
            xml << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
            xml << "  <UnstructuredGrid>\n";
            xml << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numElements << "\">\n";

            xml << "      <PointData Scalars=\"" << (solutionVarNames.empty() ? "" : solutionVarNames[0]) << "\">\n";
            for (size_t i = 0; i < numVars; ++i)
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
            xml << "        <DataArray type=\"" << getVtkTypeName<label_t>() << "\" Name=\"connectivity\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            currentOffset += sizeof(uint64_t) + connectivity.size() * sizeof(label_t);

            xml << "        <DataArray type=\"" << getVtkTypeName<label_t>() << "\" Name=\"offsets\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            currentOffset += sizeof(uint64_t) + offsets.size() * sizeof(label_t);

            xml << "        <DataArray type=\"" << getVtkTypeName<uint8_t>() << "\" Name=\"types\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
            xml << "      </Cells>\n";

            xml << "    </Piece>\n";
            xml << "  </UnstructuredGrid>\n";
            xml << "  <AppendedData encoding=\"raw\">_";

            outFile << xml.str();

            auto writeBlock = [&](const auto &vec)
            {
                using T = typename std::decay_t<decltype(vec)>::value_type;
                const uint64_t blockSize = vec.size() * sizeof(T);

                outFile.write(reinterpret_cast<const char *>(&blockSize), sizeof(uint64_t));

                outFile.write(reinterpret_cast<const char *>(vec.data()), static_cast<std::streamsize>(blockSize));
            };

            for (const auto &varData : solutionVars)
            {
                writeBlock(varData);
            }
            writeBlock(points);
            writeBlock(connectivity);
            writeBlock(offsets);

            const std::vector<uint8_t> types(numElements, 12);

            writeBlock(types);

            outFile << "</AppendedData>\n";
            outFile << "</VTKFile>\n";

            outFile.close();
            std::cout << "Successfully wrote VTU file: " << fileName << "\n";
        }
    }
}

#endif