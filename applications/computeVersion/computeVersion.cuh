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
    Function definitions and includes specific to the computeVersion executable

Namespace
    LBM

SourceFiles
    computeVersion.cu

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_COMPUTEVERSION_CUH
#define __MBLBM_COMPUTEVERSION_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/globalFunctions.cuh"
#include "../../src/inputControl.cuh"

namespace LBM
{
    /**
     * @brief Queries and returns the number of available CUDA devices.
     * @details Checks for CUDA devices and handles potential errors during device querying.
     * @return The number of CUDA devices available. Returns 0 if no devices are found.
     * @tparam throws If true, the function may throw. If not, the function returns 0 on error.
     * @throws std::runtime_error If CUDA device query fails or returns an invalid count.
     **/
    template <const bool throws>
    __host__ [[nodiscard]] deviceIndex_t countDevices()
    {
        deviceIndex_t deviceCount = 0;

        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
        {
            if constexpr (throws)
            {
                throw std::runtime_error("Error querying CUDA devices. Is the driver installed correctly?");
            }
            else
            {
                return 0;
            }
        }

        if (deviceCount < 0)
        {
            if constexpr (throws)
            {
                throw std::runtime_error("Error querying CUDA devices. Device count is negative.");
            }
            else
            {
                return 0;
            }
        }

        if (deviceCount == 0)
        {
            if constexpr (throws)
            {
                throw std::runtime_error("No CUDA devices found on the system.");
            }
            else
            {
                return 0;
            }
        }

        return deviceCount;
    }

    /**
     * @brief Converts a month abbreviation string to its corresponding index (0-11).
     * @param monthStr Three-letter month abbreviation (e.g., "Jan", "Feb").
     * @return Numerical index of the month (0 for January, 11 for December).
     * @throws std::runtime_error If the input string does not match any month abbreviation.
     **/
    __host__ [[nodiscard]] std::size_t monthIndex(const std::string &monthStr)
    {
        // Map month abbreviations to numbers
        const std::vector<std::string> months{"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

        for (std::size_t i = 0; i < 12; i++)
        {
            if (monthStr == months[i])
            {
                return i;
            }
        }

        throw std::runtime_error("Invalid month string: " + monthStr);
    }

    /**
     * @brief Generates an ISO 8601 formatted timestamp of the compilation time.
     * @details Uses the predefined __DATE__ and __TIME__ macros to determine compilation time.
     * @return Formatted timestamp string (YYYY-MM-DD HH:MM:SS).
     **/
    __host__ [[nodiscard]] const std::string compileTimestamp()
    {
        const std::string date = __DATE__;
        const std::string time = __TIME__;

        const std::string monthStr = date.substr(0, 3);
        const std::string dayStr = date.substr(4, 2);
        const std::string yearStr = date.substr(7, 4);

        // Find month number
        const std::size_t month = monthIndex(monthStr);

        // Format as ISO 8601 (YYYY-MM-DD HH:MM:SS)
        std::stringstream ss;
        ss << yearStr << "-" << std::setw(2) << std::setfill('0') << month << "-" << std::setw(2) << std::setfill('0') << std::stoi(dayStr) << " " << time;

        return ss.str();
    }

    /**
     * @brief Retrieves the value of an environment variable, returning a default value if the variable is not set.
     * @tparam verboseOutput A boolean template parameter that controls whether the retrieved environment variable and its value are printed to the console.
     * @param envVariable The name of the environment variable to retrieve.
     * @param defaultName The default value to return if the environment variable is not set.
     * @return The value of the environment variable, or the default value if it is not set.
     **/
    template <const bool verboseOutput = false>
    __host__ [[nodiscard]] const std::string getEnvironmentVariable(const std::string &envVariable, const std::string &defaultName)
    {
        const char *const env_ptr = std::getenv(envVariable.c_str());

        if (env_ptr == nullptr)
        {
            if constexpr (verboseOutput)
            {
                std::cout << envVariable << ": " << defaultName << std::endl;
            }
            return defaultName;
        }
        else
        {
            if constexpr (verboseOutput)
            {
                std::cout << envVariable << ": " << env_ptr << std::endl;
            }
            return env_ptr;
        }
    }

    /**
     * @brief Retrieves the value of an environment variable, throwing an exception if the variable is not set.
     * @tparam verboseOutput A boolean template parameter that controls whether the retrieved environment variable and its value are printed to the console.
     * @param envVariable The name of the environment variable to retrieve.
     * @return The value of the environment variable.
     * @throws std::runtime_error If the environment variable is not set.
     **/
    template <const bool verboseOutput = false>
    __host__ [[nodiscard]] const std::string getEnvironmentVariable(const std::string &envVariable)
    {
        const char *const env_ptr = std::getenv(envVariable.c_str());

        if (env_ptr == nullptr)
        {
            const std::string errorString = "Error: " + envVariable + " environment variable is not set." + "Please run:" + "  source ~/.bashrc" + "or add it to your environment.";
            throw std::runtime_error(errorString);
        }

        if constexpr (verboseOutput)
        {
            std::cout << envVariable << ": " << env_ptr << std::endl;
        }

        return env_ptr;
    }
}

#endif