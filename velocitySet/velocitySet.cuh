/**
Filename: velocitySet.cuh
Contents: Base class used for definition of the velocity set
**/

#ifndef __MBLBM_VELOCITYSET_CUH
#define __MBLBM_VELOCITYSET_CUH

namespace LBM
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
                // return static_cast<scalar_t>(3.0);
                return 3;
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
            {
                return static_cast<double>(1.0) / static_cast<double>(3.0);
                // return static_cast<scalar_t>(1.0) / as2();
            }

            __device__ __host__ [[nodiscard]] static inline consteval scalar_t scale_i() noexcept
            {
                return 3.0;
                // return as2();
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t scale_ii() noexcept
            {
                return 4.5;
                // return as2() * as2() / static_cast<scalar_t>(2.0);
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t scale_ij() noexcept
            {
                return 9.0;
                // return as2() * as2();
            }

            /**
             * @brief Scale the moments by constant values
             **/
            __device__ static inline void scale(scalar_t *const ptrRestrict moms)
            {
                // Scale the moments correctly
                moms[1] = scale_i() * moms[1];
                moms[2] = scale_i() * moms[2];
                moms[3] = scale_i() * moms[3];
                moms[4] = scale_ii() * (moms[4]);
                moms[5] = scale_ij() * (moms[5]);
                moms[6] = scale_ij() * (moms[6]);
                moms[7] = scale_ii() * (moms[7]);
                moms[8] = scale_ij() * (moms[8]);
                moms[9] = scale_ii() * (moms[9]);
            }
        };
    }
}

#include "D3Q19.cuh"

namespace LBM
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