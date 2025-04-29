/**
Filename: scalarArray.cuh
Contents: A class representing a scalar variable
The host version of this class is principally used to
partition the mesh prior to distribution to the devices
**/

#ifndef __MBLBM_SCALARARRAYS_CUH
#define __MBLBM_SCALARARRAYS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "latticeMesh.cuh"
#include "labelArray.cuh"

namespace mbLBM
{
    namespace host
    {
        class scalarArray
        {
        public:
            /**
             * @brief Constructs a scalar solution variable from a latticeMesh object
             * @return A scalar solution variable
             * @param mesh The lattice mesh
             * @note This constructor zero-initialises everything
             **/
            [[nodiscard]] inline scalarArray(const latticeMesh &mesh) noexcept
                : mesh_(mesh),
                  arr_(scalarArray_t(mesh.nPoints(), 0)) {};

            /**
             * @brief Constructs a scalar solution variable from a latticeMesh object
             * @return A scalar solution variable
             * @param mesh The lattice mesh
             * @param val The initial value
             * @note This constructor initialises everything to a uniform value
             **/
            [[nodiscard]] inline scalarArray(const latticeMesh &mesh, const scalar_t val) noexcept
                : mesh_(mesh),
                  arr_(scalarArray_t(mesh.nPoints(), val)) {};

            /**
             * @brief Constructs a scalar solution variable from another scalarArray and a partition list
             * @return A partition scalar solution variable
             * @param mesh The partition of the mesh
             * @param originalArray The original scalar solution array to be partitioned
             * @note This constructor copies the elements corresponding to the mesh partition points into the new object
             **/
            [[nodiscard]] inline scalarArray(const latticeMesh &mesh, const scalarArray &originalArray) noexcept
                : mesh_(mesh),
                  arr_(partitionArray(mesh, originalArray)) {};

            /**
             * @brief Destructor
             **/
            inline ~scalarArray() noexcept {};

            /**
             * @brief Returns immutable access to the underlying array
             * @return An immutable reference to the underlying array
             **/
            [[nodiscard]] inline scalarArray_t const &arrRef() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Returns mutable access to the underlying array
             * @return An mutable reference to the underlying array
             **/
            // [[nodiscard]] inline scalarArray_t &arrSet() noexcept
            // {
            //     return arr_;
            // }

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
             * @brief Prints the solution variable to the terminal in sequential z planes
             * @param name (Optional) Name of the variable to print to the terminal
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
             * @brief An immutable reference to the solution mesh
             **/
            const latticeMesh mesh_;

            /**
             * @brief The underlying solution array
             **/
            const scalarArray_t arr_;

            /**
             * @brief Constructs a scalar solution variable from another scalarArray and a partition list
             * @return The underlying solution array
             * @param mesh The partition of the mesh
             * @param originalArray The original scalar solution array to be partitioned
             **/
            [[nodiscard]] inline scalarArray_t partitionArray(const latticeMesh &mesh, const scalarArray &originalArray) const noexcept
            {
                scalarArray_t f(mesh.nPoints());
                for (label_t i = 0; i < mesh.nPoints(); i++)
                {
                    f[i] = originalArray.arrRef()[i + mesh.globalOffset()];
                }
                return f;
            }

            /**
             * @brief Used to partition an original array by an arbitrary list of partition indices
             * @return The elements of originalArray partitioned by partitionIndices
             * @param originalArray The original array to be partitioned
             * @param partitionIndices A list of unsigned integers corresponding to indices of originalArray
             **/
            [[nodiscard]] inline scalarArray_t partitionOriginal(const scalarArray &originalArray, const labelArray &partitionIndices) const noexcept
            {
                // Create an appropriately sized array
                scalarArray_t partitionedArray(partitionIndices.nPoints());
                for (std::size_t i = 0; i < partitionIndices.nPoints(); i++)
                {
                    partitionedArray[i] = originalArray.arrRef()[partitionIndices.arrRef()[i]];
                }
                return partitionedArray;
            }
        };
    }

    namespace device
    {
        class scalarArray
        {
        public:
            /**
             * @brief Constructs a scalar solution variable on the device from a copy of its host version
             * @return A scalarArray object copied from the host to the device
             * @param f The host copy of the scalar solution variable
             **/
            [[nodiscard]] scalarArray(const host::scalarArray &f) noexcept
                : ptr_(allocateDeviceScalarArray(f)) {};

            /**
             * @brief Constructs a scalar solution variable from an arbitrary number of points and a value
             * @return A scalarArray object of arbitrary number of points and value
             * @param nPoints The number of lattice points to assign
             * @param value The value to assign to all points of the array
             **/
            [[nodiscard]] scalarArray(const std::size_t nPoints, const scalar_t value) noexcept
                : ptr_(allocateDeviceScalarArray(nPoints, value)) {};

            /**
             * @brief Destructor
             **/
            ~scalarArray() noexcept
            {
#ifdef VERBOSE
                std::cout << "Freed memory" << std::endl;
#endif
                cudaFree((void *)ptr_);
            };

            /**
             * @brief Returns immutable access to the underlying pointer
             * @return A const-qualified pointer
             **/
            __device__ [[nodiscard]] inline constexpr const scalar_t *ptr() const noexcept
            {
                return ptr_;
            }

        private:
            /**
             * @brief Pointer to the underlying variable
             **/
            const scalar_t *ptr_;

            /**
             * @brief Allocates a block of memory on the device and returns its pointer
             * @return A raw pointer to a block of memory
             * @param size The amount of memory to be allocated
             **/
            template <typename T>
            [[nodiscard]] inline T *deviceMalloc(const std::size_t size) const noexcept
            {
                T *ptr;
                const cudaError_t i = cudaMalloc(static_cast<T **>(&ptr), size);

                if (i != cudaSuccess)
                {
                    exceptions::program_exit(i, "Unable to allocate array");
                }
#ifdef VERBOSE
                std::cout << "Allocated " << size << " bytes of memory in cudaMalloc to address " << ptr << std::endl;
#endif

                return ptr;
            }

            /**
             * @brief Allocates a scalar array on the device
             * @return A scalar_t * object pointing to a block of memory on the GPU
             * @param f The pre-existing array on the host to be copied to the GPU
             **/
            [[nodiscard]] const scalar_t *allocateDeviceScalarArray(const host::scalarArray &f) const noexcept
            {
                scalar_t *ptr = deviceMalloc<scalar_t>(f.nPoints() * sizeof(scalar_t));

                const cudaError_t i = cudaMemcpy(ptr, &(f.arrRef()[0]), f.nPoints() * sizeof(scalar_t), cudaMemcpyHostToDevice);

                if (i != cudaSuccess)
                {
                    exceptions::program_exit(i, "Unable to copy array");
                }

                return ptr;
            }

            /**
             * @brief Allocates a scalar array on the device
             * @return A scalar_t * object pointing to a block of memory on the GPU
             * @param nPoints The number of scalar points to be allocated to the block of memory
             * @param val The value set
             **/
            [[nodiscard]] const scalar_t *allocateDeviceScalarArray(const std::size_t nPoints, const scalar_t val) const noexcept
            {
                scalar_t *ptr = deviceMalloc<scalar_t>(nPoints * sizeof(scalar_t));

                const scalarArray_t f = scalarArray_t(nPoints, val);

                const cudaError_t i = cudaMemcpy(ptr, &(f[0]), nPoints * sizeof(scalar_t), cudaMemcpyHostToDevice);

                if (i != cudaSuccess)
                {
                    exceptions::program_exit(i, "Unable to set array");
                }

                return ptr;
            }
        };
    }
}

#include "ghostInterface.cuh"

#endif
