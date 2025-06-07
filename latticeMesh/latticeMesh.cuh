/**
Filename: latticeMesh.cuh
Contents: A class holding information about the solution grid
**/

#ifndef __MBLBM_LATTICEMESH_CUH
#define __MBLBM_LATTICEMESH_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
#include "../globalFunctions.cuh"

namespace mbLBM
{
    namespace host
    {
        class latticeMesh
        {
        public:
            /**
             * @brief Default-constructs a latticeMesh object
             * @return A latticeMesh object
             * @note This constructor requires that a mesh file be present in the working directory
             * @note This constructor reads from the caseInfo file and is used primarily to construct the global mesh
             **/
            [[nodiscard]] latticeMesh() noexcept
                : nx_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "nx")),
                  ny_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "ny")),
                  nz_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "nz")),
                  nPoints_(nx_ * ny_ * nz_),
                  xOffset_(0),
                  yOffset_(0),
                  zOffset_(0),
                  globalOffset_(0),
                  nodeTypes_(extractNodeTypes(ctorType::MUST_READ))
            {
#ifdef VERBOSE
                std::cout << "Allocated global latticeMesh object:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << nx_ << ";" << std::endl;
                std::cout << "    ny = " << ny_ << ";" << std::endl;
                std::cout << "    nz = " << nz_ << ";" << std::endl;
                std::cout << "}" << std::endl;
                std::cout << std::endl;
#endif
            };

            /**
             * @brief Constructs a latticeMesh object with specified read type
             * @return A latticeMesh object
             * @param readType Specify the read type: NO_READ or MUST_READ
             * @note Usage of NO_READ sets all node types to UNDEFINED
             * @note Usage of MUST_READ requires that a mesh file be present in the working directory
             * @note This constructor reads from the caseInfo file and is used primarily to construct the global mesh
             **/
            [[nodiscard]] latticeMesh(const ctorType::type readType) noexcept
                : nx_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "nx")),
                  ny_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "ny")),
                  nz_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "nz")),
                  nPoints_(nx_ * ny_ * nz_),
                  xOffset_(0),
                  yOffset_(0),
                  zOffset_(0),
                  globalOffset_(0),
                  nodeTypes_(extractNodeTypes(readType))
            {
#ifdef VERBOSE
                std::cout << "Allocated global latticeMesh object:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << nx_ << ";" << std::endl;
                std::cout << "    ny = " << ny_ << ";" << std::endl;
                std::cout << "    nz = " << nz_ << ";" << std::endl;
                std::cout << "}" << std::endl;
                std::cout << std::endl;
#endif
            };

            /**
             * @brief Constructs a latticeMesh object from a pre-defined partition range
             * @param originalMesh The global mesh
             * @param partitionRange Struct containing index ranges for x, y and z
             * @return A latticeMesh object
             **/
            //             [[nodiscard]] latticeMesh(const latticeMesh &originalMesh, const blockRange_t partitionRange) noexcept
            //                 : nx_(partitionRange.xRange.end - partitionRange.xRange.begin),
            //                   ny_(partitionRange.yRange.end - partitionRange.yRange.begin),
            //                   nz_(partitionRange.zRange.end - partitionRange.zRange.begin),
            //                   nPoints_(nx_ * ny_ * nz_),
            //                   xOffset_(partitionRange.xRange.begin),
            //                   yOffset_(partitionRange.yRange.begin),
            //                   zOffset_(partitionRange.zRange.begin),
            //                   globalOffset_(blockLabel<label_t>(xOffset_, yOffset_, zOffset_, {nx_, ny_, nz_})),
            //                   nodeTypes_(extractNodeTypes("nodeTypes", ctorType::NO_READ))
            //             {
            // #ifdef VERBOSE
            //                 std::cout << "Allocated partitioned latticeMesh object:" << std::endl;
            //                 std::cout << "{" << std::endl;
            //                 std::cout << "    nx = " << nx_ << ";" << std::endl;
            //                 std::cout << "    ny = " << ny_ << ";" << std::endl;
            //                 std::cout << "    nz = " << nz_ << ";" << std::endl;
            //                 std::cout << "}" << std::endl;
            //                 std::cout << std::endl;
            // #endif
            //             };

            /**
             * @brief Destructor for the host latticeMesh class
             **/
            ~latticeMesh() {};

            /**
             * @brief Returns the number of lattices in the x, y and z directions
             * @return Number of lattices as a label_t
             **/
            __host__ __device__ [[nodiscard]] inline constexpr label_t nx() const noexcept
            {
                return nx_;
            }
            __host__ __device__ [[nodiscard]] inline constexpr label_t ny() const noexcept
            {
                return ny_;
            }
            __host__ __device__ [[nodiscard]] inline constexpr label_t nz() const noexcept
            {
                return nz_;
            }
            __host__ __device__ [[nodiscard]] inline constexpr label_t nPoints() const noexcept
            {
                return nPoints_;
            }

            /**
             * @brief Returns the number of CUDA blocks in the x, y and z directions
             * @return Number of CUDA blocks as a label_t
             **/
            __host__ __device__ [[nodiscard]] inline constexpr label_t nxBlocks() const noexcept
            {
                return nx_ / BLOCK_NX;
            }
            __host__ __device__ [[nodiscard]] inline constexpr label_t nyBlocks() const noexcept
            {
                return ny_ / BLOCK_NY;
            }
            __host__ __device__ [[nodiscard]] inline constexpr label_t nzBlocks() const noexcept
            {
                return nz_ / BLOCK_NZ;
            }
            __host__ __device__ [[nodiscard]] inline constexpr label_t nBlocks() const noexcept
            {
                return (nx_ / BLOCK_NX) * (ny_ / BLOCK_NY) * (nz_ / BLOCK_NZ);
            }

            /**
             * @brief Returns the index offsets relative to the global mesh matrix
             * @return Index offsets as a label_t
             **/
            [[nodiscard]] inline constexpr label_t xOffset() const noexcept
            {
                return xOffset_;
            }
            [[nodiscard]] inline constexpr label_t yOffset() const noexcept
            {
                return yOffset_;
            }
            [[nodiscard]] inline constexpr label_t zOffset() const noexcept
            {
                return zOffset_;
            }
            [[nodiscard]] inline constexpr label_t globalOffset() const noexcept
            {
                return globalOffset_;
            }

            /**
             * @brief Returns the location of the boundary points at a cardinal boundary from the mesh
             * @return Cardinal directions as a label_t
             **/
            [[nodiscard]] inline constexpr label_t West() const noexcept
            {
                return 0;
            }
            [[nodiscard]] inline constexpr label_t East() const noexcept
            {
                return nx_ - 1;
            }
            [[nodiscard]] inline constexpr label_t South() const noexcept
            {
                return 0;
            }
            [[nodiscard]] inline constexpr label_t North() const noexcept
            {
                return ny_ - 1;
            }
            [[nodiscard]] inline constexpr label_t Back() const noexcept
            {
                return 0;
            }
            [[nodiscard]] inline constexpr label_t Front() const noexcept
            {
                return nz_ - 1;
            }

            /**
             * @brief Returns an immutable reference to the node types
             * @return An immutable reference to an std::vector of int16_t
             **/
            [[nodiscard]] inline constexpr nodeTypeArray_t const &nodeTypes() const noexcept
            {
                return nodeTypes_;
            }

            /**
             * @brief Prints the global array label to the terminal
             * @param name Name of the field to be printed
             **/
            void print(const std::string &name) const noexcept
            {
                std::cout << name << std::endl;
                std::cout << "nx = " << nx_ << std::endl;
                std::cout << "ny = " << ny_ << std::endl;
                std::cout << "nz = " << nz_ << std::endl;
                std::cout << std::endl;
                for (label_t k = 0; k < nz_; k++)
                {
                    for (label_t j = 0; j < ny_; j++)
                    {
                        for (label_t i = 0; i < nx_; i++)
                        {
                            std::cout << blockLabel(i + xOffset_, j + yOffset_, k + zOffset_, {nx_, ny_, nz_}) << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
            }
            void print() const noexcept
            {
                std::cout << "nx = " << nx_ << std::endl;
                std::cout << "ny = " << ny_ << std::endl;
                std::cout << "nz = " << nz_ << std::endl;
                std::cout << std::endl;
                for (label_t k = 0; k < nz_; k++)
                {
                    for (label_t j = 0; j < ny_; j++)
                    {
                        for (label_t i = 0; i < nx_; i++)
                        {
                            std::cout << blockLabel(i + xOffset_, j + yOffset_, k + zOffset_, {nx_, ny_, nz_}) << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
            }

            /**
             * @brief Writes the mesh partition to a file
             * @param fileName The file name to which the mesh will be written
             **/
            void write(const std::string &fileName = "mesh") const noexcept
            {
                if (nodeTypeCheck())
                {
                    std::ofstream myFile;
                    myFile.open(fileName);

                    myFile << "nodeTypes[" << nodeTypes_.size() << "]:" << std::endl;
                    myFile << "{" << std::endl;
                    for (label_t n = 0; n < nodeTypes_.size(); n++)
                    {
                        myFile << "    " << nodeTypes_[n] << "\n";
                    }
                    myFile << "}" << std::endl;
                    myFile.close();
                }
                else
                {
                    std::cout << "Failed node type check: not writing mesh" << std::endl;
                }
            }

        private:
            /**
             * @brief The number of lattices in the x, y and z directions
             **/
            const label_t nx_;
            const label_t ny_;
            const label_t nz_;
            const label_t nPoints_;

            /**
             * @brief Index offsets relative to the global mesh matrix
             **/
            const label_t xOffset_;
            const label_t yOffset_;
            const label_t zOffset_;
            const label_t globalOffset_;

            /**
             * @brief Node types of each lattice node
             **/
            const nodeTypeArray_t nodeTypes_;

            /**
             * @brief Extracts the type of each lattice node
             * @return The types of each lattice node
             * @param readType The constructor type: MUST_READ or NO_READ
             **/
            [[nodiscard]] const nodeTypeArray_t extractNodeTypes(const ctorType::type readType) const noexcept
            {
                if (readType == ctorType::MUST_READ)
                {
                    const std::vector<std::string> fileString = string::readCaseDirectory("nodeTypes");
                    const string::functionNameLines_t lines(fileString, "nodeTypes");
                    nodeTypeArray_t v(lines.nLines, nodeType::UNDEFINED);
                    for (label_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
                    {
                        v[i - lines.openBracketLine - 1] = static_cast<nodeType::type>(stoi(fileString[i]));
                    }
                    return v;
                }
                else
                {
                    return nodeTypeArray_t(nPoints_, nodeType::UNDEFINED);
                }
            }

            /**
             * @brief Checks whether each node type has been defined
             * @return True if all nodes are defined, false otherwise
             **/
            [[nodiscard]] inline bool nodeTypeCheck() const noexcept
            {
                for (label_t n = 0; n < nodeTypes_.size(); n++)
                {
                    if (nodeTypes_[n] == nodeType::UNDEFINED)
                    {
                        return false;
                    }
                }
                return true;
            }
        };
    }

    // namespace device
    // {
    //     class latticeMesh
    //     {
    //     public:
    //         [[nodiscard]] latticeMesh() noexcept {};
    //         ~latticeMesh() {};

    //     private:
    //     };
    // }
}

#endif