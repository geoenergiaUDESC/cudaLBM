#ifndef __MBLBM_VTU_CUH
#define __MBLBM_VTU_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Obtain the name of the type that corresponds to the C++ data type
         * @tparam T The C++ data type (e.g. float, int64_t)
         * @return A string containing the name of the VTK type (e.g. "Float32", "Int64")
         **/
        template <typename T>
        [[nodiscard]] inline consteval const char *getVtkTypeName() noexcept
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
            std::cout << "Writing VTU unstructured grid to " << fileName << std::endl;

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

            const std::vector<scalar_t> points = meshCoordinates<scalar_t>(mesh);

            const std::vector<label_t> connectivity = meshConnectivity<false>(mesh);

            const std::vector<label_t> offsets = meshOffsets(mesh);

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