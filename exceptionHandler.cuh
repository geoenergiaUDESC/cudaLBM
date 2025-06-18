/**
Filename: exceptionHandler.cuh
Contents: A generic function to exit on an error code and display an error message
**/

#ifndef __MBLBM_EXCEPTIONHANDLER_CUH
#define __MBLBM_EXCEPTIONHANDLER_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    namespace exceptions
    {
        /**
         * @brief Exits from the program safely by calling all destructors after sending a signal to the kill switch
         * @param signal Error signal
         * @param errorMessage Error message string to print on program exit
         * @note The errorMessage parameter is not necessary to specify
         **/
        void program_exit(const int signal, const std::string_view &errorMessage = "Undefined exception")
        {
            std::cout << "cudaLBM exiting: " << std::endl;
            std::cout << errorMessage << std::endl;
            std::exit(signal);
        }
    }
}

#endif