/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    A class handling the input arguments supplied to the executable

Namespace
    LBM

SourceFiles
    inputControl.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_INPUTCONTROL_CUH
#define __MBLBM_INPUTCONTROL_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    /**
     * @class inputControl
     * @brief Handles command-line input arguments and GPU device configuration
     *
     * @details Parses command-line arguments, validates input, and initializes
     * GPU device list based on provided arguments. Supports mandatory -GPU flag
     * for most executables with exceptions for utility tools.
     **/
    class inputControl
    {
    public:
        /**
         * @brief Constructor for the inputControl class
         * @param[in] argc First argument passed to main (argument count)
         * @param[in] argv Second argument passed to main (argument vector)
         * @throws std::runtime_error if argument count is negative
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
         * @return const std::vector<deviceIndex_t>& The device list containing GPU indices
         **/
        [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return deviceList_;
        }

        /**
         * @brief Verifies if an argument is present at the command line
         * @param[in] name The argument to search for
         * @return bool True if the argument is present, false otherwise
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
         * @return const std::vector<std::string>& The parsed command line arguments
         **/
        __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &commandLine() const noexcept
        {
            return commandLine_;
        }

        /**
         * @brief Returns the name of the currently running executable
         * @return const std::string& The executable name
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
         * @brief Validates and returns the number of command line arguments
         * @param[in] argc First argument passed to main (argument count)
         * @return label_t Validated number of arguments
         * @throws std::runtime_error if argument count is negative
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
         * @brief Parses command line arguments into a vector of strings
         * @param[in] argc First argument passed to main (argument count)
         * @param[in] argv Second argument passed to main (argument vector)
         * @return std::vector<std::string> Parsed command line arguments
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
         * @brief Initializes GPU device list based on command line arguments
         * @return std::vector<deviceIndex_t> List of GPU device indices
         * @throws std::runtime_error if:
         * - -GPU argument is missing for non-utility executables
         * - Requested GPUs exceed available devices
         * @note For "fieldConvert" and "fieldCalculate" executables, -GPU flag is optional (defaults to device 0)
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
         * @brief Queries the number of available CUDA devices
         * @return deviceIndex_t Count of available CUDA devices
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