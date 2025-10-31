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
    A templated class for allocating arrays on the CPU

Namespace
    LBM::host

SourceFiles
    hostArray.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAY_CUH
#define __MBLBM_HOSTARRAY_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @class array
         * @brief Templated RAII wrapper for host memory management with field initialization
         * @tparam T Data type of array elements
         * @tparam VelocitySet Velocity set configuration for LBM simulation
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class array
        {
        public:
            /**
             * @brief Constructs a host array with field initialization
             * @param[in] name Name identifier for the field
             * @param[in] mesh Lattice mesh defining array dimensions
             * @param[in] programCtrl Program control parameters
             * @post Array is initialized from latest time step or initial conditions
             **/
            __host__ [[nodiscard]] array(
                const std::string &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : arr_(initialise_array(mesh, name, programCtrl)),
                  name_(name),
                  mesh_(mesh){};

            /**
             * @brief Destructor for the host array class
             **/
            ~array() {};

            /**
             * @brief Get read-only access to underlying data
             * @return Const reference to data vector
             **/
            __host__ [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Get field name identifier
             * @return Const reference to name string
             **/
            __host__ [[nodiscard]] inline constexpr const std::string &name() const noexcept
            {
                return name_;
            }

            /**
             * @brief Get the mesh
             * @return Const reference to mesh
             **/
            __host__ [[nodiscard]] inline constexpr const host::latticeMesh &mesh() const noexcept
            {
                return mesh_;
            }

            __host__ [[nodiscard]] inline consteval time::type timeType() const noexcept
            {
                return TimeType;
            }

        private:
            /**
             * @brief The underlying std::vector
             **/
            const std::vector<T> arr_;

            /**
             * @brief Names of the solution variable
             **/
            const std::string name_;

            /**
             * @brief Reference to the lattice mesh
             **/
            const host::latticeMesh mesh_;

            /**
             * @brief Initialize array from file or initial conditions
             * @param[in] mesh Lattice mesh for dimensioning
             * @param[in] fieldName Name of field to initialize
             * @param[in] programCtrl Program control parameters
             * @return Initialized data vector
             * @throws std::runtime_error if file operations fail
             **/
            __host__ [[nodiscard]] const std::vector<T> initialise_array(const host::latticeMesh &mesh, const std::string &fieldName, const programControl &programCtrl)
            {
                if (fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    const std::string fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";

                    return fileIO::readFieldByName<T>(fileName, fieldName);
                }
                else
                {
                    return initialConditions(mesh, fieldName);
                }
            }

            // Initialises the array from the caseName
            __host__ [[nodiscard]] const std::vector<T> initialise_array(
                const std::string &caseName,
                const host::latticeMesh &mesh,
                const label_t time)
            {
                if (fileIO::hasIndexedFiles(caseName))
                {
                    // Should take the field name rather than the case name
                    const std::string fileName = caseName + "_" + std::to_string(time) + ".LBMBin";

                    return fileIO::readFieldByName<T>(fileName, caseName);
                }
                else
                {
                    // Should throw if not found
                    return initialConditions(mesh, caseName);
                }
            }

            /**
             * @brief Apply initial conditions with boundary handling
             * @param[in] mesh Lattice mesh for dimensioning and boundary detection
             * @param[in] fieldName Name of field for boundary condition lookup
             * @return Initialized data vector with boundary conditions applied
             **/
            __host__ [[nodiscard]] const std::vector<T> initialConditions(const host::latticeMesh &mesh, const std::string &fieldName)
            {
                const boundaryFields<VelocitySet> bField(fieldName);

                std::vector<T> field(mesh.nPoints(), 0);

                for (label_t bz = 0; bz < mesh.nzBlocks(); bz++)
                {
                    for (label_t by = 0; by < mesh.nyBlocks(); by++)
                    {
                        for (label_t bx = 0; bx < mesh.nxBlocks(); bx++)
                        {
                            for (label_t tz = 0; tz < block::nz(); tz++)
                            {
                                for (label_t ty = 0; ty < block::ny(); ty++)
                                {
                                    for (label_t tx = 0; tx < block::nx(); tx++)
                                    {
                                        const label_t x = (bx * block::nx()) + tx;
                                        const label_t y = (by * block::ny()) + ty;
                                        const label_t z = (bz * block::nz()) + tz;

                                        const label_t index = host::idx(tx, ty, tz, bx, by, bz, mesh);

                                        const bool is_west = (x == 0);
                                        const bool is_east = (x == mesh.nx() - 1);
                                        const bool is_south = (y == 0);
                                        const bool is_north = (y == mesh.ny() - 1);
                                        const bool is_front = (z == 0);
                                        const bool is_back = (z == mesh.nz() - 1);

                                        const label_t boundary_count =
                                            static_cast<label_t>(is_west) +
                                            static_cast<label_t>(is_east) +
                                            static_cast<label_t>(is_south) +
                                            static_cast<label_t>(is_north) +
                                            static_cast<label_t>(is_front) +
                                            static_cast<label_t>(is_back);
                                        const T value_sum =
                                            (is_west * bField.West()) +
                                            (is_east * bField.East()) +
                                            (is_south * bField.South()) +
                                            (is_north * bField.North()) +
                                            (is_front * bField.Front()) +
                                            (is_back * bField.Back());

                                        field[index] = boundary_count > 0 ? value_sum / static_cast<T>(boundary_count) : bField.internalField();
                                    }
                                }
                            }
                        }
                    }
                }

                return field;
            }
        };

        /**
         * @class arrayCollection
         * @brief Templated container for multiple field arrays with flexible initialization
         * @tparam T Data type of array elements
         * @tparam cType Constructor type specification
         * @tparam VelocitySet Velocity set configuration for LBM simulation
         **/
        template <typename T, const ctorType::type cType, class VelocitySet>
        class arrayCollection
        {
        public:
            /**
             * @brief Construct from program control and variable names
             * @param[in] programCtrl Program control parameters
             * @param[in] varNames Names of variables to include in collection
             * @param[in] mesh Lattice mesh for dimensioning
             **/
            __host__ [[nodiscard]] arrayCollection(const programControl &programCtrl, const std::vector<std::string> &varNames, const host::latticeMesh &mesh)
                : arr_(initialiseVector(programCtrl, mesh)),
                  varNames_(varNames){};

            /**
             * @brief Construct from specific time index
             * @param[in] programCtrl Program control parameters
             * @param[in] varNames Names of variables to include
             * @param[in] timeIndex Specific time index to read from
             **/
            __host__ [[nodiscard]] arrayCollection(
                const programControl &programCtrl,
                const std::vector<std::string> &varNames,
                const label_t timeIndex)
                : arr_(initialiseVector(programCtrl, timeIndex)),
                  varNames_(varNames){};

            /**
             * @brief Construct from latest available time
             * @param[in] programCtrl Program control parameters
             * @param[in] varNames Names of variables to include
             **/
            __host__ [[nodiscard]] arrayCollection(
                const programControl &programCtrl,
                const std::vector<std::string> &varNames)
                : arr_(initialiseVector(programCtrl)),
                  varNames_(varNames){};

            // Constructs from a file prefix
            __host__ [[nodiscard]] arrayCollection(
                const std::string &fileNamePrefix,
                const std::vector<std::string> &varNames,
                const label_t timeIndex)
                : arr_(initialiseVector(fileNamePrefix, timeIndex)),
                  varNames_(varNames){};

            /**
             * @brief Destructor for the host arrayCollection class
             **/
            ~arrayCollection() {};

            /**
             * @brief Get read-only access to underlying data
             * @return Const reference to data vector
             **/
            __host__ [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Get variable names in collection
             * @return Const reference to variable names vector
             **/
            __host__ [[nodiscard]] inline const std::vector<std::string> &varNames() const noexcept
            {
                return varNames_;
            }

        private:
            /**
             * @brief The underlying std::vector
             **/
            const std::vector<T> arr_;

            /**
             * @brief Names of the solution variables
             **/
            const std::vector<std::string> varNames_;

            /**
             * @brief Initialize vector from mesh dimensions
             * @param[in] programCtrl Program control parameters
             * @param[in] mesh Lattice mesh for dimensioning
             * @return Initialized data vector
             * @throws std::runtime_error if indexed files not found
             **/
            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl, const host::latticeMesh &mesh) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the latest time step
                if (!fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    throw std::runtime_error("Did not find indexed case files");
                }

                const std::string fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";
                // std::cout << "Reading from file " << fileName << std::endl;
                // std::cout << std::endl;
                return fileIO::readFieldFile<T>(fileName);
            }

            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const std::string &fileNamePrefix, const label_t timeIndex) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the latest time step
                if (!fileIO::hasIndexedFiles(fileNamePrefix))
                {
                    throw std::runtime_error("Did not find indexed case files");
                }
                const std::string fileName = fileNamePrefix + "_" + std::to_string(fileIO::timeIndices(fileNamePrefix)[timeIndex]) + ".LBMBin";
                // std::cout << "Reading from file " << fileName << std::endl;
                // std::cout << std::endl;
                return fileIO::readFieldFile<T>(fileName);
            }

            /**
             * @brief Initialize vector from specific time index
             * @param[in] programCtrl Program control parameters
             * @param[in] timeIndex Time index to read from
             * @return Initialized data vector
             * @throws std::runtime_error if indexed files not found
             **/
            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl, const label_t timeIndex) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the correct time index
                if (fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    const std::string fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::timeIndices(programCtrl.caseName())[timeIndex]) + ".LBMBin";
                    // std::cout << "Reading from file " << fileName << std::endl;
                    // std::cout << std::endl;
                    return fileIO::readFieldFile<T>(fileName);
                }
                else
                {
                    throw std::runtime_error("Did not find indexed case files");
                }
            }

            /**
             * @brief Initialize vector from latest time
             * @param[in] programCtrl Program control parameters
             * @return Initialized data vector
             **/
            __host__ [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl) const
            {
                return initialiseVector(programCtrl, fileIO::getStartIndex(programCtrl, true));
            }
        };
    }

}

#endif
