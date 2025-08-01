/**
Filename: fileIO.cuh
Contents: Includes needed for the file IO
**/

#ifndef __MBLBM_FILEIO_CUH
#define __MBLBM_FILEIO_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../memory/memory.cuh"

namespace LBM
{
    namespace fileIO
    {
        /**
         * @brief Determines whether or not the number string is all digits
         * @param numStr The number string
         * @return True if the string is all digits, false otherwise
         **/
        [[nodiscard]] inline bool isAllDigits(const std::string &numStr) noexcept
        {
            for (char c : numStr)
            {
                if (!std::isdigit(static_cast<unsigned char>(c)))
                {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Determines whether or not the number string is valid
         * @param numStr The number string
         * @return True if the string is valid, false otherwise
         **/
        [[nodiscard]] inline bool isValidString(const std::string &numStr) noexcept
        {
            return ((numStr.empty()) && (!isAllDigits(numStr)));
        }

        /**
         * @brief Determine whether or not the current directory contains valid .LBMBin files for a given file prefix
         * @param fileName The prefix of the case files - normally defined in caseInfo as caseName
         **/
        [[nodiscard]] bool hasIndexedFiles(const std::string &fileName)
        {
            const std::filesystem::path currentDir = std::filesystem::current_path();
            const std::string prefix = fileName + "_";

            for (const auto &entry : std::filesystem::directory_iterator(currentDir))
            {
                if (!entry.is_regular_file())
                {
                    continue;
                }

                const auto &filePath = entry.path();
                if (filePath.extension() != ".LBMBin")
                {
                    continue;
                }

                const std::string stem = filePath.stem().string();
                if (stem.size() <= prefix.size() || stem.substr(0, prefix.size()) != prefix)
                {
                    continue;
                }

                const std::string num_str = stem.substr(prefix.size());
                if (num_str.empty())
                {
                    continue;
                }

                if (isAllDigits(num_str))
                {
                    return true;
                }
            }

            return false;
        }

        /**
         * @brief Get the indices of the time step present within the current directory for a given file prefix
         * @param fileName The prefix of the case files - normally defined in caseInfo as caseName
         **/
        [[nodiscard]] const std::vector<label_t> timeIndices(const std::string &fileName)
        {
            std::vector<label_t> indices;
            const std::filesystem::path currentDir = std::filesystem::current_path();
            const std::string prefix = fileName + "_";

            for (const auto &entry : std::filesystem::directory_iterator(currentDir))
            {
                if (!entry.is_regular_file())
                {
                    continue;
                }

                const auto &filePath = entry.path();
                if (filePath.extension() != ".LBMBin")
                {
                    continue;
                }

                const std::string stem = filePath.stem().string();
                if (stem.size() <= prefix.size() || stem.substr(0, prefix.size()) != prefix)
                {
                    continue;
                }

                const std::string num_str = stem.substr(prefix.size());
                if (isValidString(num_str))
                {
                    continue;
                }

                try
                {
                    indices.push_back(std::stoull(num_str));
                }
                catch (...)
                {
                    continue;
                }
            }

            // Check that the indices are empty - if they are, it means no valid files were found
            if (indices.empty())
            {
                throw std::runtime_error("No matching files found with prefix " + fileName + " and .LBMBin extension");
            }

            std::sort(indices.begin(), indices.end());

            return indices;
        }

        /**
         * @brief Get the latest time step contained within the current directory of a given file prefix
         * @param fileName The prefix of the case files - normally defined in caseInfo as caseName
         **/
        [[nodiscard]] label_t latestTime(const std::string &fileName)
        {
            if (hasIndexedFiles(fileName))
            {
                const std::vector<label_t> indices = timeIndices(fileName);
                return indices[indices.size() - 1];
            }
            else
            {
                return 0;
            }
        }
    }
}

#include "fileOutput.cuh"
#include "fileInput.cuh"

#endif