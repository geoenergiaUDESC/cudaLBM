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
            static constexpr const scalar_t as2_ = 3.0;
            static constexpr const scalar_t cs2_ = 1.0 / as2_;
            static constexpr const scalar_t F_M_0_SCALE_ = 1.0;
            static constexpr const scalar_t F_M_I_SCALE_ = as2_;
            static constexpr const scalar_t F_M_II_SCALE_ = as2_ * as2_ / 2.0;
            static constexpr const scalar_t F_M_IJ_SCALE_ = as2_ * as2_;
        };
    }
}

#include "D3Q19.cuh"

#endif