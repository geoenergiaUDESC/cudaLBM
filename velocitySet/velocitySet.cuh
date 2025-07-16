/**
Filename: velocitySet.cuh
Contents: Base class used for definition of the velocity set
**/

#ifndef __MBLBM_VELOCITYSET_CUH
#define __MBLBM_VELOCITYSET_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"

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
            [[nodiscard]] inline consteval velocitySet() noexcept {};

            /**
             * @brief Parameters used by both D3Q19 and D3Q27 velocity sets
             * @return Scalar constants used by the velocity set
             * @note These methods are consteval
             **/
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t as2() noexcept
            {
                // return static_cast<scalar_t>(3.0);
                return 3.0;
            }
            __device__ __host__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
            {
                return static_cast<scalar_t>(static_cast<double>(1.0) / static_cast<double>(3.0));
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
            __device__ static inline void scale(scalar_t (&ptrRestrict moments)[10]) noexcept
            {
                // Scale the moments correctly
                moments[1] = scale_i() * moments[1];
                moments[2] = scale_i() * moments[2];
                moments[3] = scale_i() * moments[3];
                moments[4] = scale_ii() * (moments[4]);
                moments[5] = scale_ij() * (moments[5]);
                moments[6] = scale_ij() * (moments[6]);
                moments[7] = scale_ii() * (moments[7]);
                moments[8] = scale_ij() * (moments[8]);
                moments[9] = scale_ii() * (moments[9]);
            }

            // __device__ static inline void scale_v2(momentArray &ptrRestrict moments) noexcept
            // {
            //     // Scale the moments correctly
            //     moments.u = scale_i() * moments.u;
            //     moments.v = scale_i() * moments.v;
            //     moments.w = scale_i() * moments.w;
            //     moments.m_xx = scale_ii() * moments.m_xx;
            //     moments.m_xy = scale_ij() * moments.m_xy;
            //     moments.m_xz = scale_ij() * moments.m_xz;
            //     moments.m_yy = scale_ii() * moments.m_yy;
            //     moments.m_yz = scale_ij() * moments.m_yz;
            //     moments.m_zz = scale_ii() * moments.m_zz;
            // }
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