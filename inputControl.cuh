/**
Filename: inputControl.cuh
Contents: A class handling the input arguments supplied to the executable
**/

#ifndef __MBLBM_INPUTCONTROL_CUH
#define __MBLBM_INPUTCONTROL_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    class inputControl
    {
    public:
        /**
         * @brief Constructor for the inputControl class
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        [[nodiscard]] inputControl(int argc, char *argv[]) noexcept
            : nArgs_(nArgsCheck(argc)),
              deviceList_(initialiseDeviceList(argc, argv)) {};

        /**
         * @brief Destructor for the inputControl class
         **/
        ~inputControl() noexcept {};

        /**
         * @brief Return a vector of string views of the arguments passed to the solver at the command line
         * @return A vector of string views of the arguments passed to the solver at the command line
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        [[nodiscard]] const std::vector<std::string> parseCommandLine(int argc, char *argv[]) const noexcept
        {
            if (argc > 0)
            {
                std::vector<std::string> arr;
                label_t arrLength = 0;

                for (label_t i = 0; i < static_cast<label_t>(argc); i++)
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

        [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return deviceList_;
        }

    private:
        /**
         * @brief Number of arguments supplied at the command line
         **/
        const label_t nArgs_;

        /**
         * @brief Returns the number of arguments supplied at the command line
         * @return The number of arguments supplied at the command line as a label_t
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        [[nodiscard]] label_t nArgsCheck(int argc) const
        {
            // Check for a bad number of supplied arguments
            if (argc < 0)
            {
                throw std::runtime_error("Bad value of argc: cannot be negative");
                return std::numeric_limits<label_t>::max();
            }
            else
            {
                return static_cast<label_t>(argc);
            }
        }

        /**
         * @brief A list (vector of int) of GPUs employed by the simulation
         * @note Must be int since cudaSetDevice works on int
         **/
        const std::vector<deviceIndex_t> deviceList_;

        /**
         * @brief Parses the command line for the -GPU argument, checking for valid inputs and converting to deviceList
         * @return An std::vector of deviceIndex_t representing the indices of the devices
         * @note Checks that the number of GPUs supplied on the command line is valid
         **/
        [[nodiscard]] const std::vector<deviceIndex_t> initialiseDeviceList(int argc, char *argv[]) const
        {
            const std::vector<deviceIndex_t> deviceList = string::parseValue<deviceIndex_t>(parseCommandLine(argc, argv), "-GPU");

            if (deviceList.size() > static_cast<label_t>(nAvailableDevices()) | nAvailableDevices() < 1)
            {
                throw std::runtime_error("Number of GPUs requested is greater than the number available");
            }

            return deviceList;
        }

        /**
         * @brief Checks the number of available CUDA devices
         * @return The number of avaiable CUDA devices
         **/
        [[nodiscard]] deviceIndex_t nAvailableDevices() const noexcept
        {
            deviceIndex_t deviceCount = -1;
            cudaGetDeviceCount(&deviceCount);
            return deviceCount;
        }
    };
}

#endif