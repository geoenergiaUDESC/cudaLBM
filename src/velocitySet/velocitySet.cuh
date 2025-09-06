/**
Filename: velocitySet.cuh
Contents: Base class used for definition of the velocity set
**/

#ifndef __MBLBM_VELOCITYSET_CUH
#define __MBLBM_VELOCITYSET_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"
// #include "../array/array.cuh"

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
            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T as2() noexcept
            {
                return static_cast<T>(3.0);
            }
            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T cs2() noexcept
            {
                return static_cast<T>(static_cast<double>(1.0) / static_cast<double>(3.0));
            }

            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T scale_i() noexcept
            {
                return static_cast<T>(3.0);
            }
            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T scale_ii() noexcept
            {
                return static_cast<T>(4.5);
            }
            template <typename T>
            __device__ __host__ [[nodiscard]] static inline consteval T scale_ij() noexcept
            {
                return static_cast<T>(9.0);
            }

            /**
             * @brief Scale the moments by constant values
             * @param moments The moment variables
             **/
            template <class A>
            __device__ static inline void scale(A &moments) noexcept
            {
                // Scale the moments correctly
                moments(label_constant<1>()) = scale_i<scalar_t>() * (moments(label_constant<1>()));
                moments(label_constant<2>()) = scale_i<scalar_t>() * (moments(label_constant<2>()));
                moments(label_constant<3>()) = scale_i<scalar_t>() * (moments(label_constant<3>()));
                moments(label_constant<4>()) = scale_ii<scalar_t>() * (moments(label_constant<4>()));
                moments(label_constant<5>()) = scale_ij<scalar_t>() * (moments(label_constant<5>()));
                moments(label_constant<6>()) = scale_ij<scalar_t>() * (moments(label_constant<6>()));
                moments(label_constant<7>()) = scale_ii<scalar_t>() * (moments(label_constant<7>()));
                moments(label_constant<8>()) = scale_ij<scalar_t>() * (moments(label_constant<8>()));
                moments(label_constant<9>()) = scale_ii<scalar_t>() * (moments(label_constant<9>()));
            }

        private:
        };
    }
}

#include "D3Q19.cuh"

#endif