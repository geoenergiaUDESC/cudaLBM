/**
Filename: array.cuh
Contents: A templated class for various different types of arrays
**/

#ifndef __MBLBM_ARRAY_CUH
#define __MBLBM_ARRAY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../latticeMesh/latticeMesh.cuh"

namespace mbLBM
{
    namespace host
    {
        /**
         * @brief Allocates nPoints worth of T to ptr
         * @param ptr The pointer to which the memory is to be allocated
         * @param size The number of elements to be allocated to T
         **/
        template <typename T>
        __host__ void allocateMemory(T **ptr, const std::size_t nPoints) noexcept
        {
            const cudaError_t i = cudaMallocHost(ptr, sizeof(T) * nPoints);
            if (i != cudaSuccess)
            {
                exceptions::program_exit(i, "Unable to allocate array");
            }
        }

        /**
         * @brief Allocates a block of memory on the device and returns its pointer
         * @return A raw pointer to a block of memory
         * @param size The amount of memory to be allocated
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const std::size_t nPoints) noexcept
        {
            T *ptr;

            allocateMemory(&ptr, nPoints);

#ifdef VERBOSE
            std::cout << "Allocated " << sizeof(T) * nPoints << " bytes of memory in cudaMallocManaged to address " << ptr << std::endl;
#endif

            return ptr;
        }

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
                    for (label_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
                    {
                        v[i - lines.openBracketLine - 1] = static_cast<T>(stoi(fileString[i]));
                    }
                }
                else
                {
                    for (label_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
                    {
                        v[i - lines.openBracketLine - 1] = static_cast<T>(stoul(fileString[i]));
                    }
                }
            }
            else
            {
                for (label_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
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

            for (label_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
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
             * @return A partitioned scalar solution variable
             * @param mesh The partition of the mesh
             * @param originalArray The original scalar solution array to be partitioned
             * @note This constructor copies the elements corresponding to the mesh partition points into the new object
             **/
            [[nodiscard]] array(const latticeMesh &mesh, const array &originalArray) noexcept
                : mesh_(mesh),
                  arr_(partitionArray(mesh, originalArray)),
                  name_(originalArray.name()) {};

            /**
             * @brief Constructs an array from a std::vector of type T
             * @return An array of type T constructed from arr
             * @param mesh The mesh
             * @param arr The std::vector of type T from which the array is to be constructed
             * @param name The name of the array
             * @note This constructor copies the elements corresponding to the mesh partition points into the new object
             **/
            [[nodiscard]] array(const latticeMesh &mesh, const std::vector<T> &arr, const std::string &name) noexcept
                : mesh_(mesh),
                  arr_(arr),
                  name_(name) {};

            /**
             * @brief Default destructor
             **/
            ~array() noexcept {};

            /**
             * @brief Overloads the [] operator
             * @return The i-th index of the underlying array
             **/
            inline T operator[](const label_t i) const noexcept
            {
                return arr_[i];
            }

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

            [[nodiscard]] inline label_t nPoints() const noexcept
            {
                return static_cast<label_t>(arr_.size());
            }

            /**
             * @brief Writes the array to a file at a time directory
             * @param time The time step
             **/
            void saveFile(const label_t time) const noexcept
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

    namespace device
    {

        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> interlaceVectors(
            const std::vector<T> &v_0,
            const std::vector<T> &v_1,
            const std::vector<T> &v_2,
            const std::vector<T> &v_3,
            const std::vector<T> &v_4,
            const std::vector<T> &v_5,
            const std::vector<T> &v_6,
            const std::vector<T> &v_7,
            const std::vector<T> &v_8,
            const std::vector<T> &v_9) noexcept
        {
            std::vector<scalar_t> v(v_0.size() * 10, 0);

            for (std::size_t i = 0; i < v_0.size(); i++)
            {
                v[(i * 10) + 0] = v_0[i];
                v[(i * 10) + 1] = v_1[i];
                v[(i * 10) + 2] = v_2[i];
                v[(i * 10) + 3] = v_3[i];
                v[(i * 10) + 4] = v_4[i];
                v[(i * 10) + 5] = v_5[i];
                v[(i * 10) + 6] = v_6[i];
                v[(i * 10) + 7] = v_7[i];
                v[(i * 10) + 8] = v_8[i];
                v[(i * 10) + 9] = v_9[i];
            }

            return v;
        }

        /**
         * @brief Allocates nPoints worth of T to ptr
         * @param ptr The pointer to which the memory is to be allocated
         * @param size The number of elements to be allocated to T
         **/
        template <typename T>
        __host__ void allocateMemory(T **ptr, const std::size_t nPoints) noexcept
        {
            const cudaError_t i = cudaMallocManaged(ptr, sizeof(T) * nPoints);
            if (i != cudaSuccess)
            {
                exceptions::program_exit(i, "Unable to allocate array");
            }
        }

        /**
         * @brief Allocates a block of memory on the device and returns its pointer
         * @return A raw pointer to a block of memory
         * @param size The amount of memory to be allocated
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocate(const std::size_t nPoints) noexcept
        {
            T *ptr;

            allocateMemory(&ptr, nPoints);

#ifdef VERBOSE
            // std::cout << "Allocated " << sizeof(T) * nPoints << " bytes of memory in cudaMallocManaged to address " << ptr << std::endl;
            std::cout << "Allocated " << nPoints << " points to address " << ptr << std::endl;
#endif

            return ptr;
        }

        /**
         * @brief Copies a vector of type T to a device pointer of type T
         * @param ptr The pointer to which the vector is to be copied
         * @param f The vector which is to be copied to ptr
         **/
        template <typename T>
        __host__ void copy(T *ptr, const std::vector<T> &f) noexcept
        {
            const cudaError_t i = cudaMemcpy(ptr, f.data(), f.size() * sizeof(T), cudaMemcpyHostToDevice);

            if (i != cudaSuccess)
            {
                exceptions::program_exit(i, "Unable to copy array");
            }
            else
            {
#ifdef VERBOSE
                std::cout << "Copied " << sizeof(T) * f.size() << " bytes of memory in cudaMemcpy to address " << ptr << std::endl;
#endif
            }
        }

        /**
         * @brief Allocates a scalar array on the device
         * @return A T * object pointing to a block of memory on the GPU
         * @param f The pre-existing array on the host to be copied to the GPU
         **/
        // template <typename T, class F>
        // __host__ [[nodiscard]] T *allocateArray(const F &f) noexcept
        // {
        //     T *ptr = allocate<T>(f.nPoints());

        //     copy(ptr, f.arrRef());

        //     return ptr;
        // }

        /**
         * @brief Allocates a scalar array on the device
         * @return A T * object pointing to a block of memory on the GPU
         * @param f The pre-existing array on the host to be copied to the GPU
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const std::vector<T> &f) noexcept
        {
            T *ptr = allocate<T>(f.size());

            copy(ptr, f);

            return ptr;
        }

        /**
         * @brief Allocates a scalar array on the device
         * @return A pointer of type T pointing to a block of memory on the GPU
         * @param nPoints The number of scalar points to be allocated to the block of memory
         * @param val The value set
         **/
        template <typename T>
        __host__ [[nodiscard]] T *allocateArray(const label_t nPoints, const T val) noexcept
        {
            T *ptr = allocate<T>(nPoints);

            copy(ptr, scalarArray_t(nPoints, val));

            return ptr;
        }

        template <typename T>
        class array
        {
        public:
            [[nodiscard]] array(const label_t nPoints)
                : ptr_(device::allocate<T>(nPoints)) {};

            [[nodiscard]] array(const std::vector<T> &f)
                : ptr_(device::allocateArray<T>(f)) {};

            ~array()
            {
                cudaFree(ptr_);
            }

            /**
             * @brief Overloads the [] operator
             * @return The i-th index of the underlying array
             **/
            __device__ __host__ inline T operator[](const label_t i) const noexcept
            {
                return ptr_[i];
            }

            __device__ __host__ [[nodiscard]] inline const T *ptr() const noexcept
            {
                return ptr_;
            }

            __device__ __host__ [[nodiscard]] inline T *ptr() noexcept
            {
                return ptr_;
            }

            [[nodiscard]] inline T *&ptrRef() noexcept
            {
                return ptr_;
            }

        private:
            T *ptr_;
        };
    }
}

#include "labelArray.cuh"
#include "nodeTypeArray.cuh"
#include "scalarArray.cuh"

#endif
