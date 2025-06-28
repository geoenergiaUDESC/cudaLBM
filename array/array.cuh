/**
Filename: array.cuh
Contents: A templated class for various different types of arrays
**/

#ifndef __MBLBM_ARRAY_CUH
#define __MBLBM_ARRAY_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "latticeMesh/latticeMesh.cuh"
#include "programControl.cuh"
// #include "memory/memory.cuh"
#include "fileIO/fileIO.cuh"
#include "cavity.cuh"

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
            [[nodiscard]] array(const programControl &programCtrl, const host::latticeMesh &mesh)
                : arr_(initialiseVector(programCtrl, mesh)) {};

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

        private:
            /**
             * @brief The underlying std::vector
             **/
            const std::vector<T> arr_;

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
                    // std::cout << "Found file. Reading" << std::endl;
                    return fileIO::readFieldFile<T>(programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin");
                }

                if constexpr (cType == ctorType::READ_IF_PRESENT)
                {
                    // Check if the files exist
                    if (fileIO::hasIndexedFiles(programCtrl.caseName()))
                    {
                        // Construct from file
                        // std::cout << "Found file. Reading" << std::endl;
                        return fileIO::readFieldFile<T>(programCtrl.caseName() + "_" + std::to_string(fileIO::latestTime(programCtrl.caseName())) + ".LBMBin");
                    }
                    else
                    {
                        // Construct default
                        std::cout << "Not reading" << std::endl;
                        return host::moments(mesh, programCtrl.u_inf());
                    }
                }

                // Construct default
                if constexpr (cType == ctorType::NO_READ)
                {
                    std::cout << "Not reading" << std::endl;
                    return host::moments(mesh, programCtrl.u_inf());
                }

                // Fallback
                return host::moments(mesh, programCtrl.u_inf());
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
                const std::vector<T> hostFields = host::copyToHost(ptr_, nTotal);

                // Write to file
                fileIO::writeFile(filePrefix + "_" + std::to_string(timeStep) + ".LBMBin", mesh_, varNames_, hostFields, timeStep);
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
