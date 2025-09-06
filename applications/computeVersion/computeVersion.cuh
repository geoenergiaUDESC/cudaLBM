/**
Filename: computeVersion.cuh
Contents: Function definitions specific to the computeVersion executable
**/

#ifndef __MBLBM_COMPUTEVERSION_CUH
#define __MBLBM_COMPUTEVERSION_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/globalFunctions.cuh"

namespace LBM
{
    __host__ [[nodiscard]] deviceIndex_t countDevices()
    {
        deviceIndex_t deviceCount = 0;

        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
        {
            throw std::runtime_error("Error querying CUDA devices. Is the driver installed correctly?");
        }

        if (deviceCount < 0)
        {
            throw std::runtime_error("Error querying CUDA devices. Device count is negative.");
        }

        if (deviceCount == 0)
        {
            // throw std::runtime_error("No CUDA devices found on the system.");
            std::cout << "No CUDA devices found on the system." << std::endl;
            return 0;
        }

        return deviceCount;
    }

    [[nodiscard]] std::size_t monthIndex(const std::string &monthStr)
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

    [[nodiscard]] const std::string compileTimestamp()
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
}

#endif