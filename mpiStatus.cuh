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
        [[nodiscard]] mpiStatus(int argc, char *argv[])
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
        ~mpiStatus()
        {
            const int i = MPI_Finalize();

            if (!i == MPI_SUCCESS)
            {
                exceptions::program_exit(i, "MPI failed to finalise, returned code " + std::to_string(i));
            }
        };

        /**
         * @brief Returns the MPI status
         * @return The MPI status
         **/
        [[nodiscard]] inline mpiStatus_t status() const noexcept
        {
            return mpiStatus_;
        }

    private:
        /**
         * @brief The MPI status
         **/
        const int mpiStatus_;

        /**
         * @brief Initialises MPI from the input arguments
         * @return The MPI status
         **/
        [[nodiscard]] int mpiInitialise(int argc, char **argv) const noexcept
        {
            return MPI_Init(&argc, &argv);
        }
    };
}

#endif