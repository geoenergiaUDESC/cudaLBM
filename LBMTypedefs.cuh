/*
Filename: LBMTypedefs.cuh
Contains: A list of typedefs used throughout the cudaLBM source code
*/

#ifndef __MBLBM_TYPEDEFS_CUH
#define __MBLBM_TYPEDEFS_CUH

#include "LBMIncludes.cuh"

namespace mbLBM
{

/**
 * @brief Floating point type used for scalar types
 * @note Types are either 32 bit or 64 bit floating point numbers
 * @note These types are supplied via command line defines during compilation
 */
#ifdef SCALAR_PRECISION_32
    typedef float scalar_t;
#elif SCALAR_PRECISION_64
    typedef double scalar_t;
#endif

/**
 * @brief Label type used for scalar types
 * @note Types are either 32 bit or 64 bit unsigned integers
 * @note These types are supplied via command line defines during compilation
 */
#ifdef LABEL_SIZE_32
    typedef uint32_t label_t;
#elif LABEL_SIZE_64
    typedef std::size_t label_t;
#endif

    /**
     * @brief Label type used for GPU indices
     * @note Has to be int because cudaSetDevice operates on int
     */

    typedef int deviceIndex_t;

}

#endif