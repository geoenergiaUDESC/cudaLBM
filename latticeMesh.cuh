/**
Filename: latticeMesh.cuh
Contents: A class holding information about the solution grid
**/

#ifndef __MBLBM_LATTICEMESH_CUH
#define __MBLBM_LATTICEMESH_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"

namespace mbLBM
{
    class latticeMesh
    {
    public:
        /**
         * @brief Default-constructs a latticeMesh object
         * @return A latticeMesh object
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
              globalOffset_(0)
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
        [[nodiscard]] latticeMesh(const latticeMesh &originalMesh, const blockRange_t partitionRange) noexcept
            : nx_(partitionRange.xRange.end - partitionRange.xRange.begin),
              ny_(partitionRange.yRange.end - partitionRange.yRange.begin),
              nz_(partitionRange.zRange.end - partitionRange.zRange.begin),
              nPoints_(nx_ * ny_ * nz_),
              xOffset_(partitionRange.xRange.begin),
              yOffset_(partitionRange.yRange.begin),
              zOffset_(partitionRange.zRange.begin),
              globalOffset_(blockLabel<label_t>(xOffset_, yOffset_, zOffset_, {nx_, ny_, nz_}))
        {
#ifdef VERBOSE
            std::cout << "Allocated partitioned latticeMesh object:" << std::endl;
            std::cout << "{" << std::endl;
            std::cout << "    nx = " << nx_ << ";" << std::endl;
            std::cout << "    ny = " << ny_ << ";" << std::endl;
            std::cout << "    nz = " << nz_ << ";" << std::endl;
            std::cout << "}" << std::endl;
            std::cout << std::endl;
#endif
        };

        /**
         * @brief Returns the number of lattices in the x, y and z directions
         **/
        [[nodiscard]] inline constexpr label_t nx() const noexcept
        {
            return nx_;
        }
        [[nodiscard]] inline constexpr label_t ny() const noexcept
        {
            return ny_;
        }
        [[nodiscard]] inline constexpr label_t nz() const noexcept
        {
            return nz_;
        }
        [[nodiscard]] inline constexpr label_t nPoints() const noexcept
        {
            return nPoints_;
        }

        /**
         * @brief Returns the index offsets relative to the global mesh matrix
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
         * @brief Prints the global array label to the terminal
         * @param name Name of the field to be printed
         **/
        void print(
            const std::string &name) const noexcept
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
                        std::cout << blockLabel<label_t>(i + xOffset_, j + yOffset_, k + zOffset_, {nx_, ny_, nz_}) << " ";
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
                        std::cout << blockLabel<label_t>(i + xOffset_, j + yOffset_, k + zOffset_, {nx_, ny_, nz_}) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
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
    };
}

#endif