/**
Filename: runTimeIO.cuh
Contents: Functions handling runtime IO
**/

#ifndef __MBLBM_RUNTIMEIO_CUH
#define __MBLBM_RUNTIMEIO_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace runTimeIO
    {
        [[nodiscard]] const std::string duration(const long long totalSeconds) noexcept
        {
            // Handle sign and absolute value conversion
            const bool isNegative = (totalSeconds < 0);
            const unsigned long long abs_seconds = isNegative ? -static_cast<unsigned long long>(totalSeconds) : static_cast<unsigned long long>(totalSeconds);

            // Calculate time components
            const unsigned long long hours = abs_seconds / 3600;
            const unsigned long long secondsRemaining = abs_seconds % 3600;
            const unsigned long long minutes = secondsRemaining / 60;
            const unsigned long long seconds = secondsRemaining % 60;

            // Format components to HH:MM:SS
            std::ostringstream oss;
            if (isNegative)
            {
                oss << "-";
            }
            oss << hours << ":" << std::setfill('0') << std::setw(2) << minutes << ":" << std::setfill('0') << std::setw(2) << seconds;

            return oss.str();
        }

        template <typename T>
        [[nodiscard]] inline constexpr T MLUPS(const host::latticeMesh &mesh, const programControl &programCtrl, const std::chrono::high_resolution_clock::time_point &start, const std::chrono::high_resolution_clock::time_point &end) noexcept
        {
            if ((programCtrl.nt() == (programCtrl.latestTime() - 1)) | mesh.nPoints() == 0)
            {
                return 0;
            }

            const uint64_t nPoints = static_cast<uint64_t>(mesh.nx()) * static_cast<uint64_t>(mesh.ny()) * static_cast<uint64_t>(mesh.nz());

            const uint64_t nTime = static_cast<uint64_t>(programCtrl.nt()) - static_cast<uint64_t>(programCtrl.latestTime()) - 1;

            const uint64_t numerator = nPoints * nTime;

            const uint64_t denominator = static_cast<uint64_t>(1000000) * static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

            return static_cast<T>(numerator) / static_cast<T>(denominator);
        }
    }
}

#endif