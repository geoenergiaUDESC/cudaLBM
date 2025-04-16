/**
Filename: mpiStatus.cuh
Contents: A class handling the initialisation of CUDA-aware MPI
**/

#ifndef __MBLBM_MPISTATUS_CUH
#define __MBLBM_MPISTATUS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    class mpiStatus
    {
    public:
        /**
         * @brief Constructor for the mpiStatus class
         **/
        [[nodiscard]] mpiStatus(int argc, char *argv[]) noexcept
            : mpiStatus_(mpiInitialise(argc, argv))
        {
            if (!mpiStatus_ == MPI_SUCCESS)
            {
                exceptions::program_exit(mpiStatus_, "MPI failed to initialise, returned code " + std::to_string(mpiStatus_));
            }
        };

        /**
         * @brief Destructor for the mpiStatus class
         **/
        ~mpiStatus() noexcept
        {
            const mpiError_t i = static_cast<mpiError_t>(MPI_Finalize());

            if (!i == mpiError_t::SUCCESS)
            {
                exceptions::program_exit(i, "MPI failed to finalise, returned code " + std::to_string(i));
            }
        };

        /**
         * @brief Returns the MPI status
         * @return The MPI status
         **/
        [[nodiscard]] inline mpiError_t status() const noexcept
        {
            return mpiStatus_;
        }

    private:
        /**
         * @brief The MPI status
         **/
        const mpiError_t mpiStatus_;

        /**
         * @brief Initialises MPI from the input arguments
         * @return The MPI status
         **/
        [[nodiscard]] mpiError_t mpiInitialise(int argc, char **argv) const noexcept
        {
            return static_cast<mpiError_t>(MPI_Init(&argc, &argv));
        }
    };
}

#endif