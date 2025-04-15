/**
Filename: inputControl.cuh
Contents: A class handling the input arguments supplied to the executable
**/

#ifndef __MBLBM_INPUTCONTROL_CUH
#define __MBLBM_INPUTCONTROL_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    class inputControl
    {
    public:
        /**
         * @brief Constructor for the inputControl class
         **/
        [[nodiscard]] inputControl(int argc, char *argv[])
            : nArgs_(nArgsCheck(argc, argv)),
              deviceList_(string::parseValue<deviceIndex_t>(parseCommandLine(argc, argv), "-GPU")) {};

        /**
         * @brief Destructor for the inputControl class
         **/
        ~inputControl() {};

        /**
         * @brief Returns the array of device indices
         * @return A read-only reference to deviceList_
         **/
        [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return deviceList_;
        }

    private:
        /**
         * @brief Number of arguments supplied at the command line
         **/
        const std::size_t nArgs_;

        /**
         * @brief Returns the number of arguments supplied at the command line
         * @return The number of arguments supplied at the command line as a std::size_t
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        [[nodiscard]] std::size_t nArgsCheck(int argc, char *argv[]) const noexcept
        {
            // Check for a bad number of supplied arguments
            if (argc < 0)
            {
                exceptions::program_exit(-1, "Bad value of argc: cannot be negative");
                return std::numeric_limits<std::size_t>::max();
            }
            else
            {
                return static_cast<std::size_t>(argc);
            }
        }

        /**
         * @brief A list (vector of int) of GPUs employed by the simulation
         * @note Must be int since cudaSetDevice works on int
         **/
        const std::vector<deviceIndex_t> deviceList_;

        /**
         * @brief Return a vector of string views of the arguments passed to the solver at the command line
         * @return A vector of string views of the arguments passed to the solver at the command line
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        [[nodiscard]] std::vector<std::string> parseCommandLine(int argc, char *argv[]) const noexcept
        {
            if (argc > 0)
            {
                std::vector<std::string> arr;
                std::size_t arrLength = 0;

                for (std::size_t i = 0; i < static_cast<std::size_t>(argc); i++)
                {
                    arr.push_back(argv[i]);
                    arrLength = arrLength + 1;
                }

                arr.resize(arrLength);
                return arr;
            }
            else
            {
                return std::vector<std::string>{""};
            }
        }
    };
}

#endif