/**
Filename: secondOrder.cuh
Contents: Definition of the second order collision GPU kernel
**/

#ifndef __MBLBM_COLLISION_SECOND_ORDER_CUH
#define __MBLBM_COLLISION_SECOND_ORDER_CUH

namespace LBM
{
    class secondOrder : private collision
    {
    public:
        /**
         * @brief Constructor for the secondOrder class
         * @return A secondOrder object
         * @note This constructor is consteval
         **/
        [[nodiscard]] inline consteval secondOrder() noexcept {};

        /**
         * @brief Performs the collision operation
         * @param moments The 10 solution moments
         **/
        __device__ static inline void collide(thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            // Velocity updates are removed since force terms are zero
            // Diagonal moment updates (remove force terms)
            moments(label_constant<4>()) = device::t_omegaVar * moments(label_constant<4>()) + device::omegaVar_d2 * (moments(label_constant<1>())) * (moments(label_constant<1>()));
            moments(label_constant<7>()) = device::t_omegaVar * moments(label_constant<7>()) + device::omegaVar_d2 * (moments(label_constant<2>())) * (moments(label_constant<2>()));
            moments(label_constant<9>()) = device::t_omegaVar * moments(label_constant<9>()) + device::omegaVar_d2 * (moments(label_constant<3>())) * (moments(label_constant<3>()));

            // Off-diagonal moment updates (remove force terms)
            moments(label_constant<5>()) = device::t_omegaVar * moments(label_constant<5>()) + device::omega * (moments(label_constant<1>())) * (moments(label_constant<2>()));
            moments(label_constant<6>()) = device::t_omegaVar * moments(label_constant<6>()) + device::omega * (moments(label_constant<1>())) * (moments(label_constant<3>()));
            moments(label_constant<8>()) = device::t_omegaVar * moments(label_constant<8>()) + device::omega * (moments(label_constant<2>())) * (moments(label_constant<3>()));
        }

    private:
    };
}

#endif