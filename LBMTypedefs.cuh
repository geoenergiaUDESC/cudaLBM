/**
Filename: LBMTypedefs.cuh
Contains: A list of typedefs used throughout the cudaLBM source code
**/

#ifndef __MBLBM_TYPEDEFS_CUH
#define __MBLBM_TYPEDEFS_CUH

#include "LBMIncludes.cuh"

namespace LBM
{
    /**
     * @brief Checks CUDA API call result and terminates on error
     * @param err CUDA error code to verify
     * @param loc Automatically captured call location (C++20)
     *
     * If error occurs, prints diagnostic message to stderr and exits program.
     * @n Example: checkCudaErrors(cudaMalloc(&ptr, size));
     **/
    void checkCudaErrors(
        const cudaError_t err,
        const std::source_location &loc = std::source_location::current()) noexcept
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error at " << loc.file_name() << "(" << loc.line() << "):" << loc.function_name() << ": [" << static_cast<int>(err) << "] " << cudaGetErrorString(err) << "." << std::endl;
            std::exit(-1);
        }
    }

    /**
     * @brief Checks CUDA API call result and terminates on error
     * @param err CUDA error code to verify
     * @param loc Automatically captured call location (C++20)
     *
     * If error occurs, prints diagnostic message to stderr and exits program.
     * @n Example: checkCudaErrors(cudaMalloc(&ptr, size));
     **/
    inline void checkCudaErrorsInline(
        const cudaError_t err,
        const std::source_location &loc = std::source_location::current()) noexcept
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error at " << loc.file_name() << "(" << loc.line() << "):" << loc.function_name() << ": [" << static_cast<int>(err) << "] " << cudaGetErrorString(err) << "." << std::endl;
            std::exit(-1);
        }
    }

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
        __host__ __device__ constexpr operator value_type() const noexcept { return value; }
        __host__ __device__ constexpr value_type operator()() const noexcept { return value; }
    };

    /**
     * @brief Launch bounds information
     * @note These variables are device specific - enable modification later
     **/
    [[nodiscard]] inline consteval auto MAX_THREADS_PER_BLOCK() noexcept { return 512; }
    [[nodiscard]] inline consteval auto MIN_BLOCKS_PER_MP() noexcept { return 4; }
#define launchBounds __launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP())

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

#define ptrRestrict __restrict__

    /**
     * @brief Type used for arrays of scalars
     **/
    typedef std::vector<scalar_t> scalarArray_t;

    /**
     * @brief Type used to contain the moments within a given stack frame
     **/
    using momentArray_t = scalar_t[10];
    // using momentArray_t_v2 = scalar_t[16];

    /**
     * @brief Label type used for scalar types
     * @note Types are either 32 bit or 64 bit unsigned integers
     * @note These types are supplied via command line defines during compilation
     **/
#ifdef LABEL_SIZE_32
    typedef uint32_t label_t;
#elif LABEL_SIZE_64
    typedef std::size_t label_t;
#endif

    typedef uint8_t nodeType_t;

    /**
     * @brief Type used for arrays of labels
     **/
    typedef std::vector<label_t> labelArray_t;

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
     * @brief Type used for MPI errors
     * @note Has to be enumerated because there are only so many MPI error codes
     **/
    typedef enum Enum : int
    {
        SUCCESS = 0,                    // Successful return code.
        ERR_BUFFER = 1,                 // Invalid buffer pointer.
        ERR_COUNT = 2,                  // Invalid count argument.
        ERR_TYPE = 3,                   // Invalid datatype argument.
        ERR_TAG = 4,                    // Invalid tag argument.
        ERR_COMM = 5,                   // Invalid communicator.
        ERR_RANK = 6,                   // Invalid rank.
        ERR_REQUEST = 7,                // Invalid MPI_Request handle.
        ERR_ROOT = 8,                   // Invalid root.
        ERR_GROUP = 9,                  // Null group passed to function.
        ERR_OP = 10,                    // Invalid operation.
        ERR_TOPOLOGY = 11,              // Invalid topology.
        ERR_DIMS = 12,                  // Illegal dimension argument.
        ERR_ARG = 13,                   // Invalid argument.
        ERR_UNKNOWN = 14,               // Unknown error.
        ERR_TRUNCATE = 15,              // Message truncated on receive.
        ERR_OTHER = 16,                 // Other error; use Error_string.
        ERR_INTERN = 17,                // Internal error code.
        ERR_IN_STATUS = 18,             // Look in status for error value.
        ERR_PENDING = 19,               // Pending request.
        ERR_ACCESS = 20,                // Permission denied.
        ERR_AMODE = 21,                 // Unsupported amode passed to open.
        ERR_ASSERT = 22,                // Invalid assert.
        ERR_BAD_FILE = 23,              // Invalid file name (for example, path name too long).
        ERR_BASE = 24,                  // Invalid base.
        ERR_CONVERSION = 25,            // An error occurred in a user-supplied data-conversion function.
        ERR_DISP = 26,                  // Invalid displacement.
        ERR_DUP_DATAREP = 27,           // Conversion functions could not be registered because a data representation identifier that was already defined was passed to Register_datarep.
        ERR_FILE_EXISTS = 28,           // File exists.
        ERR_FILE_IN_USE = 29,           // File operation could not be completed, as the file is currently open by some process.
        ERR_FILE = 30,                  // Invalid file handle.
        ERR_INFO_KEY = 31,              // Illegal info key.
        ERR_INFO_NOKEY = 32,            // No such key.
        ERR_INFO_VALUE = 33,            // Illegal info value.
        ERR_INFO = 34,                  // Invalid info object.
        ERR_IO = 35,                    // I/O error.
        ERR_KEYVAL = 36,                // Illegal key value.
        ERR_LOCKTYPE = 37,              // Invalid locktype.
        ERR_NAME = 38,                  // Name not found.
        ERR_NO_MEM = 39,                // Memory exhausted.
        ERR_NOT_SAME = 40,              // Collective argument not identical on all processes, or collective routines called in a different order by different processes.
        ERR_NO_SPACE = 41,              // Not enough space.
        ERR_NO_SUCH_FILE = 42,          // File (or directory) does not exist.
        ERR_PORT = 43,                  // Invalid port.
        ERR_PROC_ABORTED = 74,          // Operation failed because a remote peer has aborted.
        ERR_QUOTA = 44,                 // Quota exceeded.
        ERR_READ_ONLY = 45,             // Read-only file system.
        ERR_RMA_CONFLICT = 46,          // Conflicting accesses to window.
        ERR_RMA_SYNC = 47,              // Erroneous RMA synchronization.
        ERR_SERVICE = 48,               // Invalid publish/unpublish.
        ERR_SIZE = 49,                  // Invalid size.
        ERR_SPAWN = 50,                 // Error spawning.
        ERR_UNSUPPORTED_DATAREP = 51,   // Unsupported datarep passed to MPI_File_set_view.
        ERR_UNSUPPORTED_OPERATION = 52, // Unsupported operation, such as seeking on a file that supports only sequential access.
        ERR_WIN = 53,                   // Invalid window.
        T_ERR_MEMORY = 54,              // Out of memory.
        T_ERR_NOT_INITIALIZED = 55,     // Interface not initialized.
        T_ERR_CANNOT_INIT = 56,         // Interface not in the state to be initialized.
        T_ERR_INVALID_INDEX = 57,       // The enumeration index is invalid.
        T_ERR_INVALID_ITEM = 58,        // The item index queried is out of range.
        T_ERR_INVALID_HANDLE = 59,      // The handle is invalid.
        T_ERR_OUT_OF_HANDLES = 60,      // No more handles available.
        T_ERR_OUT_OF_SESSIONS = 61,     // No more sessions available.
        T_ERR_INVALID_SESSION = 62,     // Session argument is not a valid session.
        T_ERR_CVAR_SET_NOT_NOW = 63,    // Variable cannot be set at this moment.
        T_ERR_CVAR_SET_NEVER = 64,      // Variable cannot be set until end of execution.
        T_ERR_PVAR_NO_STARTSTOP = 65,   // Variable cannot be started or stopped.
        T_ERR_PVAR_NO_WRITE = 66,       // Variable cannot be written or reset.
        T_ERR_PVAR_NO_ATOMIC = 67,      // Variable cannot be read and written atomically.
        ERR_RMA_RANGE = 68,             // Target memory is not part of the window (in the case of a window created with MPI_Win_create_dynamic, target memory is not attached.
        ERR_RMA_ATTACH = 69,            // Memory cannot be attached (e.g., because of resource exhaustion).
        ERR_RMA_FLAVOR = 70,            // Passed window has the wrong flavor for the called function.
        ERR_RMA_SHARED = 71,            // Memory cannot be shared (e.g., some process in the group of the specified communicator cannot expose shared memory).
        T_ERR_INVALID = 72,             // Invalid use of the interface or bad parameter values(s).
        T_ERR_INVALID_NAME = 73,        // The variable or category name is invalid.
        ERR_SESSION = 78,               // Invalid session
        ERR_LASTCODE = 93               // Last error code.
    } mpiError_t;

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

    __device__ __constant__ label_t d_nx;
    __device__ __constant__ label_t d_ny;
    __device__ __constant__ label_t d_nz;
    __device__ __constant__ scalar_t d_Re;
    __device__ __constant__ scalar_t d_tau;
    __device__ __constant__ scalar_t d_u_inf;
    __device__ __constant__ scalar_t d_omega;
    __device__ __constant__ label_t d_NUM_BLOCK_X;
    __device__ __constant__ label_t d_NUM_BLOCK_Y;
    __device__ __constant__ label_t d_NUM_BLOCK_Z;
}

#endif