/**
Filename: array.cuh
Contents: A templated class for various different types of arrays
**/

#ifndef __MBLBM_ARRAY_CUH
#define __MBLBM_ARRAY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../latticeMesh/latticeMesh.cuh"
#include "../programControl.cuh"
#include "../fileIO/fileIO.cuh"
#include "../cases/lidDrivenCavity/cavity.cuh"

namespace LBM
{
    namespace host
    {
        template <typename T, const ctorType::type cType>
        class array
        {
        public:
            /**
             * @brief Constructor for the host array class
             * @param programCtrl The program control dictionary
             * @param mesh The mesh
             **/
            [[nodiscard]] array(const programControl &programCtrl, const std::vector<std::string> &varNames, const host::latticeMesh &mesh)
                : arr_(initialiseVector(programCtrl, mesh)),
                  varNames_(varNames) {};

            /**
             * @brief Constructs the device array from an std::vector of type T
             * @param programCtrl Immutable reference to the program control
             * @param varNames The names of the variables
             * @param timeIndex The index of the time step
             * @return An array object constructed from f
             **/
            [[nodiscard]] array(
                const programControl &programCtrl,
                const std::vector<std::string> &varNames,
                const label_t timeIndex)
                : arr_(initialiseVector(programCtrl, timeIndex)),
                  varNames_(varNames) {};

            /**
             * @brief Constructs the device array from an std::vector of type T at the latest time
             * @param programCtrl Immutable reference to the program control
             * @param varNames The names of the variables
             * @return An array object constructed from f
             **/
            [[nodiscard]] array(
                const programControl &programCtrl,
                const std::vector<std::string> &varNames)
                : arr_(initialiseVector(programCtrl)),
                  varNames_(varNames) {};

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
                static_assert(cType == ctorType::NO_READ || cType == ctorType::MUST_READ || cType == ctorType::READ_IF_PRESENT, "Invalid constructor type");

                // Forced to read the file
                if constexpr (cType == ctorType::MUST_READ)
                {
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

                if constexpr (cType == ctorType::READ_IF_PRESENT)
                {
                    // Check if the files exist
                    if (fileIO::hasIndexedFiles(programCtrl.caseName()))
                    {
                        const std::string fileName = programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin";
                        std::cout << "Reading from file " << fileName << std::endl;
                        std::cout << std::endl;
                        return fileIO::readFieldFile<T>(fileName);
                    }
                    else
                    {
                        // Construct default
                        std::cout << "Constructing default" << std::endl;
                        std::cout << std::endl;
                        return host::moments(mesh, programCtrl.u_inf());
                    }
                }

                // Construct default
                if constexpr (cType == ctorType::NO_READ)
                {
                    std::cout << "Constructing default" << std::endl;
                    std::cout << std::endl;
                    return host::moments(mesh, programCtrl.u_inf());
                }
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

    namespace device
    {
        template <typename T>
        class array
        {
        public:
            /**
             * @brief Constructs the device array from an std::vector of type T
             * @param f The std::vector to be allocated on the device
             * @return An array object constructed from f
             **/
            [[nodiscard]] array(const std::vector<T> &f, const std::vector<std::string> &varNames, const host::latticeMesh &mesh)
                : ptr_(device::allocateArray<T>(f)),
                  varNames_(varNames),
                  mesh_(mesh) {};

            /**
             * @brief Constructs the device array from a host array
             * @tparam cType The constructor type of the host array
             * @param hostArray The array allocated on the host
             * @param mesh The lattice mesh
             **/
            template <const ctorType::type cType>
            [[nodiscard]] array(const host::array<T, cType> &hostArray, const host::latticeMesh &mesh)
                : ptr_(device::allocateArray<T>(hostArray.arr())),
                  varNames_(hostArray.varNames()),
                  mesh_(mesh){};

            /**
             * @brief Destructor for the array class
             **/
            ~array() noexcept
            {
                checkCudaErrors(cudaFree(ptr_));
            }

            /**
             * @brief Overloads the [] operator
             * @return The i-th index of the underlying array
             **/
            __device__ __host__ [[nodiscard]] inline T operator[](const label_t i) const noexcept
            {
                return ptr_[i];
            }

            /**
             * @brief Provides read-only access to the data
             * @return A const-qualified pointer to the data
             **/
            __device__ __host__ [[nodiscard]] inline const T *ptr() const noexcept
            {
                return ptr_;
            }

            /**
             * @brief Provides mutable access to the data
             * @return A pointer to the data
             **/
            __device__ __host__ [[nodiscard]] inline T *ptr() noexcept
            {
                return ptr_;
            }

            /**
             * @brief Provides access to the variable names
             * @return An immutable reference to an std::vector of std::strings
             **/
            __host__ [[nodiscard]] inline const std::vector<std::string> &varNames() const noexcept
            {
                return varNames_;
            }

            /**
             * @brief Provides access to a variable name
             * @return An std::string
             **/
            __host__ [[nodiscard]] inline const std::string &varName(const label_t var) const noexcept
            {
                return varNames_[var];
            }

            /**
             * @brief Provides access to the mesh
             * @return An immutable reference to a host::latticeMesh object
             **/
            __host__ [[nodiscard]] inline const host::latticeMesh &mesh() const noexcept
            {
                return mesh_;
            }

            /**
             * @brief Wraps the implementation of the binary write
             * @param filePrefix Prefix of the name of the file to be written
             * @param fields Object containing the solution variables encoded in interleaved AoS format
             * @param timeStep The current time step
             **/
            __host__ void write(const std::string &filePrefix, const std::size_t timeStep)
            {
                const std::size_t nVars = varNames_.size();
                const std::size_t nTotal = static_cast<std::size_t>(mesh_.nx()) * static_cast<std::size_t>(mesh_.ny()) * static_cast<std::size_t>(mesh_.nz()) * nVars;

                // Copy device -> host
                const std::vector<T> hostFields = host::toHost(ptr_, nTotal);

                // Write to file
                fileIO::writeFile(filePrefix + "_" + std::to_string(timeStep) + ".LBMBin", mesh_, varNames_, hostFields, timeStep);
            }

            /**
             * @brief Returns the total size of the array, i.e. the number of points * number of variables
             * @return The total size of the array as a label_t
             **/
            __host__ [[nodiscard]] inline constexpr label_t size() const noexcept
            {
                return mesh_.nPoints() * varNames_.size();
            }

        private:
            /**
             * @brief Pointer to the data
             **/
            T *const ptrRestrict ptr_;

            /**
             * @brief Names of the solution variables
             **/
            const std::vector<std::string> varNames_;

            /**
             * @brief Reference to the mesh
             **/
            const host::latticeMesh &mesh_;
        };
    }
}

#include "labelArray.cuh"
#include "nodeTypeArray.cuh"
#include "scalarArray.cuh"

#endif
