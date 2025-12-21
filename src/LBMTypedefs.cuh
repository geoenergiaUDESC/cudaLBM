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
    A list of typedefs used throughout the cudaLBM source code

Namespace
    LBM

SourceFiles
    LBMTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_TYPEDEFS_CUH
#define __MBLBM_TYPEDEFS_CUH

#include "LBMIncludes.cuh"
#include "errorHandler.cuh"

namespace LBM
{
    /**
     * @brief Shorthand for __restrict__
     **/
#define ptrRestrict __restrict__

    /**
     * @brief CUDA implementation of a std::integral constant
     * @param T The type of integral value
     * @param v The value
     **/
    template <typename T, T v>
    struct integralConstant
    {
        static constexpr const T value = v;
        using value_type = T;
        using type = integralConstant;
        __device__ __host__ inline consteval operator value_type() const noexcept { return value; }
        __device__ __host__ inline consteval value_type operator()() const noexcept { return value; }
    };

    /**
     * @brief Floating point type used for scalar types
     * @note Types are either 32 bit or 64 bit floating point numbers
     * @note These types are supplied via command line defines during compilation
     **/
#ifdef SCALAR_PRECISION_32
    typedef float scalar_t;
#elif SCALAR_PRECISION_64
    typedef double scalar_t;
#endif

    /**
     * @brief Struct used to hold three-dimensional coordinates of a point
     **/
    struct pointVector
    {
        const scalar_t x;
        const scalar_t y;
        const scalar_t z;
    };

    /**
     * @brief Label type used for scalar types
     * @note Types are either 32 bit or 64 bit unsigned integers
     * @note These types are supplied via command line defines during compilation
     **/
#ifdef LABEL_SIZE_32
    typedef uint32_t label_t;
#elif LABEL_SIZE_64
    typedef uint64_t label_t;
#endif

    /**
     * @brief Block dimensions descriptor
     * @details Stores lattice dimensions in 3D space
     **/
    struct blockLabel_t
    {
        const label_t nx; // < Lattice points in x-direction
        const label_t ny; // < Lattice points in y-direction
        const label_t nz; // < Lattice points in z-direction
    };

    /**
     * @brief Point indices descriptor
     * @details Stores point indices in 3D space
     **/
    struct pointLabel_t
    {
        const label_t x; // < Lattice point in x-direction
        const label_t y; // < Lattice point in y-direction
        const label_t z; // < Lattice point in z-direction
    };

    /**
     * @brief 1D range descriptor [begin, end)
     **/
    struct blockPartitionRange_t
    {
        const label_t begin; // < Inclusive start index
        const label_t end;   // < Exclusive end index
    };

    /**
     * @brief 3D block range descriptor
     * @details Defines a rectangular region in lattice space
     **/
    struct blockRange_t
    {
        const blockPartitionRange_t xRange; // < X-dimension range
        const blockPartitionRange_t yRange; // < Y-dimension range
        const blockPartitionRange_t zRange; // < Z-dimension range
    };

    /**
     * @brief Type used to contain a variable or variables over a block of shared memory
     * @param T The type of variable
     * @param N The number of variables
     * @param blockSize The number of lattice points in the shared memory block
     **/
    template <typename T, const label_t N, const label_t blockSize>
    struct sharedArray
    {
        T arr[N][blockSize];
    };

    /**
     * @brief Type used for compile-time indices
     **/
    template <const label_t label>
    using label_constant = const integralConstant<label_t, label>;
    template <const label_t label>
    using q_i = const integralConstant<label_t, label>;
    template <const label_t label>
    using m_i = const integralConstant<label_t, label>;

    namespace thread
    {
        /**
         * @brief Fixed-size array container for single-threaded device code
         * @tparam T Type of elements stored in the array
         * @tparam N Number of elements in the array (compile-time constant)
         **/
        template <typename T, const label_t N>
        class array
        {
        public:
            /**
             * @brief Constructs array with specified initial values
             * @tparam Args Variadic template parameter pack for initial values
             * @param args Initial values for array elements
             * @pre Number of arguments must exactly match template parameter N
             * @note Compile-time enforced check ensures correct number of arguments
             **/
            template <typename... Args>
            __device__ constexpr array(const Args... args) : data_{args...}
            {
                static_assert(sizeof...(Args) == N, "Incorrect number of arguments");
            }

            /**
             * @brief Default constructor (value-initializes all elements)
             * @note Elements will be default-initialized or zero-initialized
             **/
            array() = default;

            /**
             * @brief Compile-time mutable element access
             * @tparam index_ Compile-time index value
             * @param index Index tag (label_constant wrapper)
             * @return Reference to element at specified index
             * @pre index_ must be in range [0, N-1]
             * @note No runtime bounds checking - compile-time safe
             **/
            template <const label_t index_>
            __device__ T &operator[](const label_constant<index_> index) __restrict__ noexcept
            {
                return data_[index()];
            }

            /**
             * @brief Compile-time read-only element access
             * @tparam index_ Compile-time index value
             * @param index Index tag (label_constant wrapper)
             * @return Const reference to element at specified index
             * @pre index_ must be in range [0, N-1]
             * @note No runtime bounds checking - compile-time safe
             **/
            template <const label_t index_>
            __device__ const T &operator[](const label_constant<index_> index) __restrict__ const noexcept
            {
                return data_[index()];
            }

            /**
             * @brief Unified element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param idx Index value or compile-time index tag
             * @return Reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            template <typename Index>
            __device__ constexpr T &operator[](const Index idx) __restrict__ noexcept
            {
                if constexpr (std::is_integral_v<Index>)
                {
                    // Runtime index
                    return data_[idx];
                }
                else
                {
                    // Compile-time index (assuming Index is std::integral_constant)
                    static_assert(Index::value < N, "Index out of bounds");
                    return data_[Index::value];
                }
            }

            /**
             * @brief Unified read-only element access (compile-time or runtime)
             * @tparam Index Type of index (integral type or std::integral_constant)
             * @param idx Index value or compile-time index tag
             * @return Const reference to element at specified index
             * @pre Index must be in range [0, N-1]
             * @note Compile-time bounds checking for integral_constant types
             * @note Runtime access for integral types (no bounds checking)
             **/
            template <typename Index>
            __device__ constexpr const T &operator[](const Index idx) __restrict__ const noexcept
            {
                if constexpr (std::is_integral_v<Index>)
                {
                    // Runtime index
                    static_assert(std::is_integral_v<Index>, "Index is not a compile-time constant");
                    return data_[idx];
                }
                else
                {
                    // Compile-time index (assuming Index is std::integral_constant)
                    static_assert(Index::value < N, "Index out of bounds");
                    return data_[Index::value];
                }
            }

            /**
             * @brief Returns the number of elements in the array
             * @return Compile-time constant number of elements (N)
             * @note Consteval function - evaluated at compile time
             **/
            __device__ [[nodiscard]] inline consteval label_t size() const noexcept
            {
                return N;
            }

            /**
             * @brief Returns the sum of the elements of the array
             * @return Sum of all of the elements (N)
             * @note Constexpr function - potentially evaluated at compile time
             **/
            __device__ [[nodiscard]] inline constexpr T sum() const noexcept
            {
                return [&]<const label_t... Is>(std::index_sequence<Is...>)
                {
                    return (data_[Is] + ...);
                }(std::make_index_sequence<N>{});
            }

        private:
            /**
             * @brief The underlying data
             **/
            T ptrRestrict data_[N];
        };
    }

    /**
     * @brief Type used for lattice indices
     **/
    template <const label_t q_>
    using lattice_constant = const std::integral_constant<label_t, q_>;

    /**
     * @brief Label type used for GPU indices
     * @note Has to be int because cudaSetDevice operates on int
     **/
    typedef int deviceIndex_t;

    /**
     * @brief Label type used for MPI ranks
     * @note Has to be int because MPI_Comm_rank and etc take &int
     **/
    typedef int mpiRank_t;

    /**
     * @brief Constructor read types
     * @note Has to be enumerated because there are only so many possible read configurations
     **/
    namespace ctorType
    {
        typedef enum Enum : int
        {
            NO_READ = 0,
            MUST_READ = 1,
            READ_IF_PRESENT = 2
        } type;
    }
    // template <const ctorType::type T>
    // using constructorType = const std::integral_constant<ctorType::type, T>;

    namespace time
    {
        typedef enum Enum : int
        {
            instantaneous = 0,
            timeAverage = 1
        } type;
    }
    // template <const time::::type T>
    // using timeType = const std::integral_constant<time::::type, T>;

    namespace device
    {
        /**
         * @brief Device constant variables
         * @note These variables MUST be initialised on the GPU at program start with cudaMemcpyToSymbol
         **/
        __device__ __constant__ label_t nx;
        __device__ __constant__ label_t ny;
        __device__ __constant__ label_t nz;
        __device__ __constant__ scalar_t Re;
        __device__ __constant__ scalar_t tau;
        __device__ __constant__ scalar_t u_inf;
        __device__ __constant__ scalar_t L_char;
        __device__ __constant__ scalar_t omega;
        __device__ __constant__ scalar_t t_omegaVar;
        __device__ __constant__ scalar_t omegaVar_d2;
        __device__ __constant__ label_t NUM_BLOCK_X;
        __device__ __constant__ label_t NUM_BLOCK_Y;
        __device__ __constant__ label_t NUM_BLOCK_Z;
        __device__ __constant__ scalar_t We;
        __device__ __constant__ scalar_t sigma;
        __device__ __constant__ scalar_t gamma;

        /**
         * @brief Class holding N device pointers of type T
         **/
        template <label_t N, typename T>
        class ptrCollection
        {
        public:
            static_assert(N > 0, "N must be positive"); // Ensure N is valid

            /**
             * @brief Variadic constructor: construct from an arbitrary number of pointers
             * @return A pointer collection object constructed from args
             * @param args An arbitrary number N of pointers of type T
             **/
            template <typename... Args>
            __device__ __host__ constexpr ptrCollection(const Args... args)
                : ptrs_{args...} // Initialize array with arguments
            {
                static_assert(sizeof...(Args) == N, "Incorrect number of arguments");

                static_assert((std::is_convertible_v<Args, T *> && ...), "All arguments must be convertible to T*");
            }

            /**
             * @brief Provides access to the GPU pointer
             * @param i The index of the pointer
             **/
            template <const label_t i>
            __device__ __host__ [[nodiscard]] inline constexpr T *ptr() const noexcept
            {
                static_assert(i < N, "Invalid pointer access");

                return ptrs_[i];
            }

            __host__ [[nodiscard]] constexpr std::array<T *, N> to_array() const noexcept
            {
                std::array<T *, N> arr;

                for (label_t i = 0; i < N; i++)
                {
                    arr[i] = ptrs_[i];
                }

                return arr;
            }

        private:
            /**
             * @brief The underlying pointers
             **/
            T *const ptrRestrict ptrs_[N];
        };
    }
}

#include "hardwareConfig.cuh"

#endif