/**
Filename: labelArray.cuh
Contents: A class representing a list of labels
The host version of this class is principally used to
partition the mesh prior to distribution to the devices
**/

#ifndef __MBLBM_LABELARRAYS_CUH
#define __MBLBM_LABELARRAYS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    namespace host
    {
        class labelArray
        {
        public:
            /**
             * @brief Constructs a label array from a number of lattice points
             * @return A label array
             * @param nPoints Total number of lattice points
             * @note This constructor zero-initialises everything
             **/
            [[nodiscard]] labelArray(const latticeMesh &mesh) noexcept
                : mesh_(mesh),
                  arr_(partitionLabelList(mesh.nPoints())) {};

            // Construct from the original and a mesh partition
            [[nodiscard]] labelArray(const labelArray &original, const latticeMesh &meshPartition) noexcept
                : mesh_(meshPartition),
                  arr_(partitionLabelList(original, meshPartition)) {};

            /**
             * @brief Destructor
             **/
            ~labelArray() noexcept {};

            /**
             * @brief Returns immutable access to the underlying array
             * @return An immutable reference to the underlying array
             **/
            [[nodiscard]] inline constexpr labelArray_t const &arrRef() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Total number of lattice points contained within the array
             * @return Total number of lattice points
             * @note This may not be equivalent to the total number of global lattice
             * points since scalarArray can be constructed from a partition
             **/
            [[nodiscard]] inline constexpr label_t nPoints() const noexcept
            {
                return arr_.size();
            }

            /**
             * @brief Prints the solution variable to the terminal in sequential z planes
             * @param name Name of the field to be printed
             **/
            void print(const std::string &name) const noexcept
            {
                std::cout << name << std::endl;
                std::cout << "nx = " << mesh_.nx() << std::endl;
                std::cout << "ny = " << mesh_.ny() << std::endl;
                std::cout << "nz = " << mesh_.nz() << std::endl;
                std::cout << std::endl;
                for (label_t k = 0; k < mesh_.nz(); k++)
                {
                    for (label_t j = 0; j < mesh_.ny(); j++)
                    {
                        for (label_t i = 0; i < mesh_.nx(); i++)
                        {
                            std::cout << arr_[blockLabel<label_t>(i, j, k, mesh_)] << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
            }
            void print() const noexcept
            {
                std::cout << "nx = " << mesh_.nx() << std::endl;
                std::cout << "ny = " << mesh_.ny() << std::endl;
                std::cout << "nz = " << mesh_.nz() << std::endl;
                std::cout << std::endl;
                for (label_t k = 0; k < mesh_.nz(); k++)
                {
                    for (label_t j = 0; j < mesh_.ny(); j++)
                    {
                        for (label_t i = 0; i < mesh_.nx(); i++)
                        {
                            std::cout << arr_[blockLabel<label_t>(i, j, k, mesh_)] << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
            }

        private:
            /**
             * @brief Immutable reference to the mesh
             **/
            const latticeMesh &mesh_;

            /**
             * @brief The underlying label array
             **/
            const labelArray_t arr_;

            /**
             * @brief Construct a list of lattice labels from a total number of points
             **/
            [[nodiscard]] labelArray_t partitionLabelList(const label_t n) const noexcept
            {
                labelArray_t labels(n);
                for (label_t i = 0; i < n; i++)
                {
                    labels[i] = i;
                }
                return labels;
            }

            /**
             * @brief Partition a list of lattice labels from the original and a specified range
             **/
            [[nodiscard]] labelArray_t partitionLabelList(const labelArray &original, const latticeMesh &mesh) const noexcept
            {
                labelArray_t labels(mesh.nPoints());
                for (label_t k = 0; k < mesh.nz(); k++)
                {
                    for (label_t j = 0; j < mesh.ny(); j++)
                    {
                        for (label_t i = 0; i < mesh.nx(); i++)
                        {
                            labels[i] = original.arrRef()[blockLabel<label_t>(i + mesh.xOffset(), j + mesh.yOffset(), k + mesh.zOffset(), mesh_)];
                        }
                    }
                }
                return labels;
            }
        };
    }
}

#endif