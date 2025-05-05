/**
Filename: velocitySet.cuh
Contents: Base class used for definition of the velocity set
**/

#ifndef __MBLBM_VELOCITYSET_CUH
#define __MBLBM_VELOCITYSET_CUH

namespace mbLBM
{
    namespace VelocitySet
    {
        class velocitySet
        {
        public:
            /**
             * @brief Constructor for the velocitySet class
             * @return A velocitySet object
             * @note This constructor is consteval
             **/
            [[nodiscard]] inline consteval velocitySet() {};

            /**
             * @brief Parameters used by both D3Q19 and D3Q27 velocity sets
             * @return Scalar constants used by the velocity set
             * @note These methods are consteval
             **/
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t as2() noexcept
            {
                return 3.0;
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
            {
                return 1.0 / as2();
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t F_M_0_SCALE() noexcept
            {
                return 1.0;
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t F_M_I_SCALE() noexcept
            {
                return as2();
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t F_M_II_SCALE() noexcept
            {
                return as2() * as2() / 2.0;
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t F_M_IJ_SCALE() noexcept
            {
                return as2() * as2();
            }

            /**
             * @brief Multiplies all of the moments by their coefficients
             * @param moments The moments to be scaled
             **/
            __device__ static inline void scale(scalar_t (&moments)[10]) noexcept
            {
                moments[0] = moments[0] * F_M_0_SCALE();

                moments[1] = moments[1] * F_M_I_SCALE();
                moments[2] = moments[2] * F_M_I_SCALE();
                moments[3] = moments[3] * F_M_I_SCALE();

                moments[4] = moments[4] * F_M_II_SCALE();
                moments[5] = moments[5] * F_M_IJ_SCALE();
                moments[6] = moments[6] * F_M_IJ_SCALE();
                moments[7] = moments[7] * F_M_II_SCALE();
                moments[8] = moments[8] * F_M_IJ_SCALE();
                moments[9] = moments[9] * F_M_II_SCALE();
            }
        };
    }
}

#include "D3Q19.cuh"

namespace mbLBM
{
    /**
     * @brief Velocity set used throughout the code
     * @note For now only D3Q19 is implemented
     * @note These types are supplied via command line defines during compilation
     **/
#ifdef STENCIL_TYPE_D3Q19
    typedef VelocitySet::D3Q19 vSet;
#endif
}

#endif