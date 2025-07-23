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
        __device__ static inline void collide(scalar_t (&ptrRestrict moments)[10]) noexcept
        {
            // Velocity updates are removed since force terms are zero
            // Diagonal moment updates (remove force terms)
            moments[4] = device::t_omegaVar * moments[4] + device::omegaVar_d2 * (moments[1]) * (moments[1]);
            moments[7] = device::t_omegaVar * moments[7] + device::omegaVar_d2 * (moments[2]) * (moments[2]);
            moments[9] = device::t_omegaVar * moments[9] + device::omegaVar_d2 * (moments[3]) * (moments[3]);

            // Off-diagonal moment updates (remove force terms)
            moments[5] = device::t_omegaVar * moments[5] + device::omega * (moments[1]) * (moments[2]);
            moments[6] = device::t_omegaVar * moments[6] + device::omega * (moments[1]) * (moments[3]);
            moments[8] = device::t_omegaVar * moments[8] + device::omega * (moments[2]) * (moments[3]);
        }

    private:
    };
}

#endif