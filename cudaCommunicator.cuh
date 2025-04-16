/**
Filename: cudaCommunicator.cuh
Contents: A class handling parallel execution
**/

#ifndef __MBLBM_CUDACOMMUNICATOR_CUH
#define __MBLBM_CUDACOMMUNICATOR_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    /**
     * @brief Constructor for the cudaCommunicator class
     **/
    class cudaCommunicator
    {
    public:
        [[nodiscard]] cudaCommunicator() noexcept
            : myRank_(myRankInitialise()),
              totalRank_(totalRankInitialise())
        {
            std::cout << "MPI rank: " << myRank_ << std::endl;
            std::cout << "Total rank: " << totalRank_ << std::endl;
        };

        ~cudaCommunicator() noexcept {};

        /**
         * @brief Returns the process MPI rank
         **/
        [[nodiscard]] inline mpiRank_t rank() const noexcept
        {
            return myRank_;
        }

        /**
         * @brief Returns the total number of MPI ranks
         **/
        [[nodiscard]] inline mpiRank_t totalRank() const noexcept
        {
            return totalRank_;
        }

    private:
        /**
         * @brief The process MPI rank
         **/
        const mpiRank_t myRank_;

        /**
         * @brief Returns the process MPI rank by calling MPI_Comm_rank
         * @return The process MPI rank
         **/
        [[nodiscard]] mpiRank_t myRankInitialise() const noexcept
        {
            mpiRank_t rank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }

        /**
         * @brief The total number of MPI ranks
         **/
        const mpiRank_t totalRank_;

        /**
         * @brief Returns the total number of MPI ranks by calling MPI_Comm_size
         * @return The total number of MPI ranks
         **/
        [[nodiscard]] mpiRank_t totalRankInitialise() const noexcept
        {
            mpiRank_t rank = -1;
            MPI_Comm_size(MPI_COMM_WORLD, &rank);
            return rank;
        }
    };
}

#endif