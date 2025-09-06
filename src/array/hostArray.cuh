/**
Filename: hostArray.cuh
Contents: A templated class for various different types of arrays allocated on the host
**/

#ifndef __MBLBM_HOSTARRAY_CUH
#define __MBLBM_HOSTARRAY_CUH

namespace LBM
{
    namespace host
    {
        template <typename T, class VSet>
        class array
        {
        public:
            /**
             * @brief Constructor for the host array class
             * @param name Name of the solution variable
             * @param mesh The mesh
             * @param programCtrl The program control dictionary
             **/
            [[nodiscard]] array(
                const std::string &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : arr_(initialise_array(mesh, name, programCtrl)),
                  name_(name) {};

            /**
             * @brief Destructor for the host array class
             **/
            ~array() {};

            /**
             * @brief Provides read-only access to the underlying std::vector
             * @return An immutable reference to an std::vector of type T
             **/
            [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Provides access to the variable name
             * @return An immutable reference to a std::string
             **/
            __host__ [[nodiscard]] inline constexpr const std::string &name() const noexcept
            {
                return name_;
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
             * @brief Initialises the std::vector
             * @return A std::vector of type T initialised from either the latest time or the initial conditions
             * @param mesh The lattice mesh
             * @param fieldName The name of the solution variable
             * @param programCtrl The program control dictionary
             **/
            [[nodiscard]] const std::vector<T> initialise_array(const host::latticeMesh &mesh, const std::string &fieldName, const programControl &programCtrl)
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

            /**
             * @brief Initialises the std::vector from an initial conditions file
             * @return A std::vector of type T initialised from either the latest time or the initial conditions
             * @param mesh The lattice mesh
             * @param fieldName The name of the solution variable
             **/
            [[nodiscard]] const std::vector<T> initialConditions(const host::latticeMesh &mesh, const std::string &fieldName)
            {
                const boundaryFields<VSet> bField(fieldName);

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

        template <typename T, const ctorType::type cType, class VSet>
        class arrayCollection
        {
        public:
            /**
             * @brief Constructor for the host arrayCollection class
             * @param programCtrl The program control dictionary
             * @param mesh The mesh
             **/
            [[nodiscard]] arrayCollection(const programControl &programCtrl, const std::vector<std::string> &varNames, const host::latticeMesh &mesh)
                : arr_(initialiseVector(programCtrl, mesh)),
                  varNames_(varNames) {};

            /**
             * @brief Constructs the host array from an std::vector of type T
             * @param programCtrl Immutable reference to the program control
             * @param varNames The names of the variables
             * @param timeIndex The index of the time step
             * @return An array object constructed from f
             **/
            [[nodiscard]] arrayCollection(
                const programControl &programCtrl,
                const std::vector<std::string> &varNames,
                const label_t timeIndex)
                : arr_(initialiseVector(programCtrl, timeIndex)),
                  varNames_(varNames) {};

            /**
             * @brief Constructs the host arrayCollection from an std::vector of type T at the latest time
             * @param programCtrl Immutable reference to the program control
             * @param varNames The names of the variables
             * @return An array object constructed from f
             **/
            [[nodiscard]] arrayCollection(
                const programControl &programCtrl,
                const std::vector<std::string> &varNames)
                : arr_(initialiseVector(programCtrl)),
                  varNames_(varNames) {};

            /**
             * @brief Destructor for the host arrayCollection class
             **/
            ~arrayCollection() {};

            /**
             * @brief Provides read-only access to the underlying std::vector
             * @return An immutable reference to an std::vector of type T
             **/
            [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Provides access to the variable names
             * @return An immutable reference to an std::vector of std::strings
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
             * @brief Initialises the std::vector
             * @param programCtrl The program control dictionary
             * @param mesh The mesh
             **/
            [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl, const host::latticeMesh &mesh) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the latest time step
                if (!fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    throw std::runtime_error("Did not find indexed case files");
                }

                const std::string fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";
                std::cout << "Reading from file " << fileName << std::endl;
                std::cout << std::endl;
                return fileIO::readFieldFile<T>(fileName);
            }

            /**
             * @brief Initialises the std::vector
             * @param programCtrl The program control dictionary
             * @param timeIndex The index of the file
             **/
            [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl, const label_t timeIndex) const
            {
                static_assert(cType == ctorType::MUST_READ, "Invalid constructor type");

                // Get the correct time index
                if (fileIO::hasIndexedFiles(programCtrl.caseName()))
                {
                    const std::string fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::timeIndices(programCtrl.caseName())[timeIndex]) + ".LBMBin";
                    std::cout << "Reading from file " << fileName << std::endl;
                    std::cout << std::endl;
                    return fileIO::readFieldFile<T>(fileName);
                }
                else
                {
                    throw std::runtime_error("Did not find indexed case files");
                }
            }

            /**
             * @brief Initialises the std::vector from the latest time step
             * @param programCtrl The program control dictionary
             **/
            [[nodiscard]] const std::vector<T> initialiseVector(const programControl &programCtrl) const
            {
                return initialiseVector(programCtrl, fileIO::getStartIndex(programCtrl, true));
            }
        };
    }

}

#endif
