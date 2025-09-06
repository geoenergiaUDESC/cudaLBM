/**
Filename: deviceArray.cuh
Contents: A templated class for various different types of arrays allocated on the device
**/

#ifndef __MBLBM_DEVICEARRAY_CUH
#define __MBLBM_DEVICEARRAY_CUH

namespace LBM
{
    namespace device
    {
        template <typename T>
        class array
        {
        public:
            /**
             * @brief Constructs the device array from a host array
             * @tparam cType The constructor type of the host array
             * @param hostArray The array allocated on the host
             * @param mesh The lattice mesh
             **/
            template <class VSet>
            [[nodiscard]] array(const host::array<T, VSet> &hostArray, const host::latticeMesh &mesh)
                : ptr_(device::allocateArray<T>(hostArray.arr())),
                  name_(hostArray.name()),
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
            __host__ [[nodiscard]] inline const std::string &name() const noexcept
            {
                return name_;
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
             * @brief Returns the total size of the array, i.e. the number of points * number of variables
             * @return The total size of the array as a label_t
             **/
            __host__ [[nodiscard]] inline constexpr label_t size() const noexcept
            {
                return mesh_.nPoints();
            }

        private:
            /**
             * @brief Pointer to the data
             **/
            T *const ptrRestrict ptr_;

            /**
             * @brief Names of the solution variables
             **/
            const std::string &name_;

            /**
             * @brief Reference to the mesh
             **/
            const host::latticeMesh &mesh_;
        };
    }
}

#endif
