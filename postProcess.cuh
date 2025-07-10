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
        // void writeTecplotHexahedralData(
        //     const std::vector<std::vector<scalar_t>> &solutionVars,
        //     const std::string &fileName,
        //     const host::latticeMesh &mesh,
        //     const std::vector<std::string> &solutionVarNames,
        //     const std::string &title) noexcept
        // {
        //     const size_t numNodes = static_cast<size_t>(mesh.nx()) * mesh.ny() * mesh.nz();
        //     const size_t numElements = static_cast<size_t>(mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);

        //     // FILE *pFile = fopen(fileName.c_str(), "w");
        //     // if (pFile == nullptr)
        //     // {
        //     //     std::cerr << "Error opening file: " << fileName << "\n";
        //     //     return;
        //     // }

        //     std::ofstream outFile(fileName);
        //     if (!outFile)
        //     {
        //         std::cerr << "Error opening file: " << fileName << "\n";
        //         return;
        //     }

        //     // Write Tecplot header
        //     outFile << "TITLE = \"" << title << "\"\n";
        //     outFile << "VARIABLES = \"X\" \"Y\" \"Z\" ";
        //     for (auto &name : solutionVarNames)
        //     {
        //         outFile << "\"" << name << "\" ";
        //     }
        //     outFile << "\n";

        //     // UNSTRUCTURED GRID FORMAT
        //     // const label_t numElements = (mesh.nx() - 1) * (mesh.ny() - 1) * (mesh.nz() - 1);
        //     outFile << "ZONE T=\"Hexahedral Zone\", NODES=" << numNodes << ", ELEMENTS=" << numElements << ", DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n";

        //     // fprintf(pFile, "TITLE = \"%s\"\n", title.c_str());
        //     // fprintf(pFile, "VARIABLES = \"X\" \"Y\" \"Z\" ");
        //     // for (const auto &name : solutionVarNames)
        //     // {
        //     //     fprintf(pFile, "\"%s\" ", name.c_str());
        //     // }
        //     // fprintf(pFile, "\n");

        //     // fprintf(pFile, "ZONE T=\"Hexahedral Zone\", NODES=%zu, ELEMENTS=%zu, DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n", numNodes, numElements);

        //     // --- Escreve os Blocos de Dados ---
        //     // X, Y, Z coordinates
        //     for (label_t k = 0; k < mesh.nz(); k++)
        //     {
        //         for (label_t j = 0; j < mesh.ny(); j++)
        //         {
        //             for (label_t i = 0; i < mesh.nx(); i++)
        //             {
        //                 outFile << static_cast<double>(i) / static_cast<double>(mesh.nx() - 1) << "\n";
        //                 // fprintf(pFile, "%.12g\n", static_cast<double>(i) / static_cast<double>(mesh.nx() - 1));
        //             }
        //         }
        //     }
        //     for (label_t k = 0; k < mesh.nz(); k++)
        //     {
        //         for (label_t j = 0; j < mesh.ny(); j++)
        //         {
        //             for (label_t i = 0; i < mesh.nx(); i++)
        //             {
        //                 outFile << static_cast<double>(j) / static_cast<double>(mesh.ny() - 1) << "\n";
        //                 // fprintf(pFile, "%.12g\n", static_cast<double>(j) / static_cast<double>(mesh.ny() - 1));
        //             }
        //         }
        //     }
        //     for (label_t k = 0; k < mesh.nz(); k++)
        //     {
        //         for (label_t j = 0; j < mesh.ny(); j++)
        //         {
        //             for (label_t i = 0; i < mesh.nx(); i++)
        //             {
        //                 outFile << static_cast<double>(k) / static_cast<double>(mesh.nz() - 1) << "\n";
        //                 // fprintf(pFile, "%.12g\n", static_cast<double>(k) / static_cast<double>(mesh.nz() - 1));
        //             }
        //         }
        //     }

        //     // Solution variables
        //     for (const auto &varData : solutionVars)
        //     {
        //         for (const auto &value : varData)
        //         {
        //             outFile << value << "\n";
        //             // fprintf(pFile, "%.12g\n", static_cast<double>(value));
        //         }
        //     }

        //     // --- Escreve a Conectividade ---
        //     for (label_t k = 0; k < mesh.nz() - 1; k++)
        //     {
        //         const label_t k_offset = k * mesh.nx() * mesh.ny();
        //         const label_t k1_offset = (k + 1) * mesh.nx() * mesh.ny();
        //         for (label_t j = 0; j < mesh.ny() - 1; j++)
        //         {
        //             const label_t j_offset = j * mesh.nx();
        //             const label_t j1_offset = (j + 1) * mesh.nx();
        //             for (label_t i = 0; i < mesh.nx() - 1; i++)
        //             {
        //                 const label_t n0 = k_offset + j_offset + i + 1;
        //                 const label_t n1 = n0 + 1;
        //                 const label_t n3 = k_offset + j1_offset + i + 1;
        //                 const label_t n2 = n3 + 1;
        //                 const label_t n4 = k1_offset + j_offset + i + 1;
        //                 const label_t n5 = n4 + 1;
        //                 const label_t n7 = k1_offset + j1_offset + i + 1;
        //                 const label_t n6 = n7 + 1;

        //                 outFile << n0 << " " << n1 << " " << n2 << " " << n3 << " " << n4 << " " << n5 << " " << n6 << " " << n7 << "\n";
        //                 // fprintf(pFile, "%zu %zu %zu %zu %zu %zu %zu %zu\n", n0, n1, n2, n3, n4, n5, n6, n7);
        //             }
        //         }
        //     }

        //     // fclose(pFile);

        //     std::cout << "Successfully wrote Tecplot file: " << fileName << "\n";
        // }

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