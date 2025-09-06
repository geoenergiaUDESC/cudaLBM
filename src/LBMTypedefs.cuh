/**
Filename: LBMTypedefs.cuh
Contains: A list of typedefs used throughout the cudaLBM source code
**/

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
    template <const label_t q_>
    using label_constant = const integralConstant<label_t, q_>;

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
            template <label_t index_>
            __device__ T &operator()(const label_constant<index_> index) __restrict__ noexcept
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
            template <label_t index_>
            __device__ const T &operator()(const label_constant<index_> index) __restrict__ const noexcept
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
    template <const ctorType::type T>
    using constructorType = const std::integral_constant<ctorType::type, T>;

    /**
     * @brief Device constant variables
     * @note These variables MUST be initialised on the GPU at program start with cudaMemcpyToSymbol
     **/
    namespace device
    {
        __device__ __constant__ label_t nx;
        __device__ __constant__ label_t ny;
        __device__ __constant__ label_t nz;
        __device__ __constant__ scalar_t Re;
        __device__ __constant__ scalar_t tau;
        __device__ __constant__ scalar_t u_inf;
        __device__ __constant__ scalar_t omega;
        __device__ __constant__ scalar_t t_omegaVar;
        __device__ __constant__ scalar_t omegaVar_d2;
        __device__ __constant__ label_t NUM_BLOCK_X;
        __device__ __constant__ label_t NUM_BLOCK_Y;
        __device__ __constant__ label_t NUM_BLOCK_Z;

        /**
         * @brief Struct holding N device pointers of type T
         **/
        template <label_t N, typename T>
        struct ptrCollection
        {
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