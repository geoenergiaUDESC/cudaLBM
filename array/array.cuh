/**
Filename: array.cuh
Contents: A templated class for various different types of arrays
**/

#ifndef __MBLBM_ARRAY_CUH
#define __MBLBM_ARRAY_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "latticeMesh.cuh"

namespace mbLBM
{
    namespace host
    {
        /**
         * @brief Reads an array type object into an appropriately typed std::vector
         * @return An std::vector of type T imported from the file pointed to by fieldName
         * @param fieldName The name of the field to read from file
         **/
        template <typename T>
        [[nodiscard]] const std::vector<T> read(const std::string &fieldName) noexcept
        {
            const std::vector<std::string> fileString = string::readCaseDirectory(fieldName);
            const string::functionNameLines_t lines(fileString, fieldName);

            std::vector<T> v(lines.nLines, 0);

            if constexpr (std::is_integral_v<T>)
            {
                if constexpr (std::is_signed_v<T>)
                {
                    for (std::size_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
                    {
                        v[i - lines.openBracketLine - 1] = static_cast<T>(stoi(fileString[i]));
                    }
                }
                else
                {
                    for (std::size_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
                    {
                        v[i - lines.openBracketLine - 1] = static_cast<T>(stoul(fileString[i]));
                    }
                }
            }
            else
            {
                for (std::size_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
                {
                    v[i - lines.openBracketLine - 1] = static_cast<T>(stod(fileString[i]));
                }
            }

            return v;
        }

        /**
         * @brief Explicit template specialisation of read for nodeType::type
         * @return An std::vector of type nodeType::type imported from the file pointed to by fieldName
         * @param fieldName The name of the field to read from file
         **/
        template <>
        [[nodiscard]] const std::vector<nodeType::type> read(const std::string &fieldName) noexcept
        {
            const std::vector<std::string> fileString = string::readCaseDirectory(fieldName);
            const string::functionNameLines_t lines(fileString, fieldName);

            std::vector<nodeType::type> v(lines.nLines, nodeType::UNDEFINED);

            for (std::size_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
            {
                v[i - lines.openBracketLine - 1] = static_cast<nodeType::type>(stoi(fileString[i]));
            }

            return v;
        }

        /**
         * @brief Either reads or default-initialises an appropriately typed std::vector
         * @return An std::vector of type T either imported from the file pointed to by fieldName OR zero-initialised
         * @param mesh The lattice mesh
         * @param name The name of the variable
         * @param readType The constructor type: MUST_READ or NO_READ
         **/
        template <typename N>
        [[nodiscard]] const std::vector<N> readOrDefault(const latticeMesh &mesh, const std::string &name, const ctorType::type readType) noexcept
        {
            if (readType == ctorType::MUST_READ)
            {
                return read<N>(name);
            }
            else
            {
                return std::vector<N>(mesh.nPoints(), 0);
            }
        }

        /**
         * @brief Explicit template specialisation of readOrDefault for nodeType::type
         * @return An std::vector of type T either imported from the file pointed to by fieldName OR zero-initialised
         * @param mesh The lattice mesh
         * @param name The name of the variable
         * @param readType The constructor type: MUST_READ or NO_READ
         **/
        template <>
        [[nodiscard]] const std::vector<nodeType::type> readOrDefault(const latticeMesh &mesh, const std::string &name, const ctorType::type readType) noexcept
        {
            if (readType == ctorType::MUST_READ)
            {
                return read<nodeType::type>(name);
            }
            else
            {
                return std::vector<nodeType::type>(mesh.nPoints(), nodeType::UNDEFINED);
            }
        }

        template <typename T>
        class array
        {
        public:
            /**
             * @brief Constructs a variable array from a latticeMesh object, name and a read
             * @return An array object
             * @param mesh The lattice mesh
             * @param name The name of the field
             * @param readType The type of constructor
             * @note This constructor zero-initialises everything if readType is not MUST_READ
             * @note This constructor attempts to read from file if readType is MUST_READ
             **/
            [[nodiscard]] array(const latticeMesh &mesh, const std::string &name, const ctorType::type readType) noexcept
                : mesh_(mesh),
                  arr_(readOrDefault<T>(mesh, name, readType)),
                  name_(name) {};

            /**
             * @brief Constructs a variable array from a latticeMesh object, name and a uniform value
             * @return An array object
             * @param mesh The lattice mesh
             * @param name The name of the field
             * @param value The initial value of the field
             * @note This constructor initialises everything to a uniform value
             **/
            [[nodiscard]] array(const latticeMesh &mesh, const std::string &name, const T value) noexcept
                : mesh_(mesh),
                  arr_(std::vector<T>(mesh.nPoints(), value)),
                  name_(name) {};

            /**
             * @brief Constructs a scalar solution variable from another scalarArray and a partition list
             * @return A partition scalar solution variable
             * @param mesh The partition of the mesh
             * @param originalArray The original scalar solution array to be partitioned
             * @note This constructor copies the elements corresponding to the mesh partition points into the new object
             **/
            [[nodiscard]] array(const latticeMesh &mesh, const array &originalArray) noexcept
                : mesh_(mesh),
                  arr_(partitionArray(mesh, originalArray)),
                  name_(originalArray.name()) {};

            /**
             * @brief Default destructor
             **/
            ~array() noexcept {};

            /**
             * @brief Returns the name of the variable
             * @return An std::string of the variable name
             **/
            [[nodiscard]] inline constexpr std::string name() const noexcept
            {
                return name_;
            }

            /**
             * @brief Returns immutable access to the underlying array
             * @return An immutable reference to the underlying array
             **/
            [[nodiscard]] inline scalarArray_t const &arrRef() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Total number of lattice points contained within the array
             * @return Total number of lattice points
             * @note This may not be equivalent to the total number of global lattice
             * points since scalarArray can be constructed from a partition
             **/
            [[nodiscard]] inline auto nPoints() const noexcept
            {
                return arr_.size();
            }

            /**
             * @brief Writes the array to a file at a time directory
             * @param time The time step
             **/
            void saveFile(const std::size_t time) const noexcept
            {
                std::ofstream myFile;
                myFile.open(std::to_string(time) + "/" + name_);

                myFile << name_ << "[" << arr_.size() << "]:" << std::endl;
                myFile << "{" << std::endl;
                for (label_t n = 0; n < arr_.size(); n++)
                {
                    myFile << "    " << arr_[n] << "\n";
                }
                myFile << "}" << std::endl;
                myFile.close();
            }

        private:
            /**
             * @brief An immutable reference to the solution mesh
             **/
            const latticeMesh mesh_;

            /**
             * @brief The underlying solution array
             **/
            const std::vector<T> arr_;

            /**
             * @brief The name of the field
             **/
            const std::string name_;
        };
    }
}

#endif
