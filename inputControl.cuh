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
        [[nodiscard]] inputControl(const int argc, const char *const argv[]) noexcept
            : nArgs_(nArgsCheck(argc)),
              commandLine_(parseCommandLine(argc, argv)),
              deviceList_(initialiseDeviceList()) {};

        /**
         * @brief Destructor for the inputControl class
         **/
        ~inputControl() noexcept {};

        /**
         * @brief Returns the device list as a vector of ints
         * @return The device list
         **/
        [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return deviceList_;
        }

        /**
         * @brief Verifies if an argument is present at the command line
         * @return True if the argument is present at the command line, false otherwise
         * @param name The argument to search for
         **/
        [[nodiscard]] bool isArgPresent(const std::string &name) const noexcept
        {
            for (label_t i = 0; i < commandLine_.size(); i++)
            {
                if (commandLine_[i] == name)
                {
                    return true;
                }
            }

            return false;
        }

        /**
         * @brief Returns the command line input as a vector of strings
         * @return The command line input
         **/
        __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &commandLine() const noexcept
        {
            return commandLine_;
        }

        /**
         * @brief Returns the name of the currently running executable as a string
         * @return The name of the currently running executable
         **/
        __host__ [[nodiscard]] inline constexpr const std::string &executableName() const noexcept
        {
            return commandLine_[0];
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
        [[nodiscard]] label_t nArgsCheck(const int argc) const
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
         * @brief The parsed command line
         **/
        const std::vector<std::string> commandLine_;

        /**
         * @brief Return a vector of string views of the arguments passed to the solver at the command line
         * @return A vector of string views of the arguments passed to the solver at the command line
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        [[nodiscard]] const std::vector<std::string> parseCommandLine(const int argc, const char *const argv[]) const noexcept
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

        /**
         * @brief A list (vector of int) of GPUs employed by the simulation
         * @note Must be int since cudaSetDevice works on int
         **/
        const std::vector<deviceIndex_t> deviceList_;

        /**
         * @brief Parses the command line for the -GPU argument, checking for valid inputs and converting to deviceList
         * @return An std::vector of deviceIndex_t representing the indices of the devices
         * @note Checks that the number of GPUs supplied on the command line is valid, and if the argument is "fieldConvert", the -GPU flag is not necessary
         **/
        [[nodiscard]] const std::vector<deviceIndex_t> initialiseDeviceList() const
        {
            if (isArgPresent("-GPU"))
            {
                const std::vector<deviceIndex_t> parsedList = string::parseValue<deviceIndex_t>(commandLine_, "-GPU");

                if (parsedList.size() > static_cast<label_t>(nAvailableDevices()) || nAvailableDevices() < 1)
                {
                    throw std::runtime_error("Number of GPUs requested is greater than the number available");
                }
                return parsedList;
            }
            else
            {
                if ((executableName() == "fieldConvert") | (executableName() == "fieldCalculate"))
                {
                    return {0};
                }
                else
                {
                    throw std::runtime_error("Error: The -GPU argument is mandatory for the " + executableName() + " executable.");
                }
            }
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