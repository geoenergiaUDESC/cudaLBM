/**
Filename: strings.cuh
Contents: Functions employed throughout the source code to manipulate strings
**/

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "exceptionHandler.cuh"

#ifndef __MBLBM_STRINGS_CUH
#define __MBLBM_STRINGS_CUH

namespace LBM
{
    namespace string
    {
        struct functionNameLines_t
        {
        public:
            /**
             * @brief Reads a std::vector of std::strings for functionName
             * @return A functionNameLines_t object
             * @param fileString A std::vector of std::strings read from a file to be scanned
             * @param functionName The function name for which fileString is to be scanned
             **/
            [[nodiscard]] inline functionNameLines_t(const std::vector<std::string> &fileString, const std::string &functionName) noexcept
                : functionNameLine(scanFor(fileString, functionName)),
                  openBracketLine(scanFor(fileString, "{")),
                  closeBracketLine(scanFor(fileString, "}")),
                  nLines(closeBracketLine - openBracketLine - 1) {};

            /**
             * @brief The line on which the function name appears
             **/
            const label_t functionNameLine;

            /**
             * @brief The line on which the first open curly brace after functionName appears
             **/
            const label_t openBracketLine;

            /**
             * @brief The line on which the curly braces are closed
             **/
            const label_t closeBracketLine;

            /**
             * @brief The number of lines read
             **/
            const label_t nLines;

        private:
            /**
             * @brief Scans fileString for a substring of characters and returns the line on which it is found
             * @return The line on which functionName is found
             * @param fileString A std::vector of std::strings read from a file to be scanned
             * @param functionName The function name for which fileString is to be scanned
             **/
            [[nodiscard]] label_t scanFor(const std::vector<std::string> &fileString, const std::string &functionName) const noexcept
            {
                for (label_t i = 0; i < fileString.size(); i++)
                {
                    if (fileString[i].find(functionName) != std::string::npos)
                    {
                        return i;
                    }
                }
                return 0;
            }
        };

        /**
         * @brief Reads the caseInfo file in the current directory into a vector of strings
         * @return A std::vector of std::string_view objects contained within the caseInfo file
         * @note This function will cause the program to exit if caseInfo is not found in the launch directory
         **/
        [[nodiscard]] std::vector<std::string> readCaseDirectory(const std::string_view &fileName) noexcept
        {
            // Does the file even exist?
            if (!std::filesystem::exists(fileName))
            {
                exceptions::program_exit(-1, std::string(fileName) + std::string(" file not opened"));
            }

            // Read the caseInfo file contained within the directory
            std::ifstream caseInfo(std::string(fileName).c_str());
            std::vector<std::string> S;
            std::string s;

            // Count the number of lines
            label_t nLines = 0;

            // Count the number of lines
            while (std::getline(caseInfo, s))
            {
                S.push_back(s);
                nLines = nLines + 1;
            }

            S.resize(nLines);

            return S;
        }

        /**
         * @brief Checks that the input string is numeric
         * @param s The string_view object which is to be checked
         * @return True if s is numeric, false otherwise
         **/
        [[nodiscard]] bool is_number(const std::string &s) noexcept
        {
            std::string::const_iterator it = s.begin();
            while (it != s.end() && std::isdigit(*it))
            {
                ++it;
            }
            return !s.empty() && it == s.end();
        }

        /**
         * @brief Splits the string_view object s according to the delimiter delim
         * @param s The string_view object which is to be split
         * @param delim The delimiter character by which s is split, e.g. a comma, space, etc
         * @param removeWhitespace Controls the removal of whitespace; removes blank spaces from the return value if true (default true)
         * @return A std::vector of std::string_view objects split from s by delim
         * @note This function can be used to, for example, split a string by commas, spaces, etc
         **/
        [[nodiscard]] std::vector<std::string> split(const std::string_view &s, const char delim, const bool removeWhitespace = true) noexcept
        {
            std::vector<std::string> result;
            const char *left = s.begin();
            for (const char *it = left; it != s.end(); ++it)
            {
                if (*it == delim)
                {
                    result.emplace_back(&*left, it - left);
                    left = it + 1;
                }
            }
            if (left != s.end())
            {
                result.emplace_back(&*left, s.end() - left);
            }

            // Remove whitespace from the returned vector
            if (removeWhitespace)
            {
                result.erase(std::remove(result.begin(), result.end(), ""), result.end());
            }

            return result;
        }

        /**
         * @brief Searches for an entry corresponding to variableName within the vector of strings S
         * @param T The type of variable returned
         * @param S The vector of strings which is searched
         * @param name The name of the variable which is to be found and returned as type T
         * @return The value of the variable expressed as a label_t
         * @note This function can be used to, for example, read an entry of nx within caseInfo after caseInfo has been loaded into S
         * @note The line containing the definition of variableName must separate variableName and its value with a space, for instance nx 128;
         **/
        template <typename T>
        [[nodiscard]] T extractParameter(const std::vector<std::string> &S, const std::string_view &name) noexcept
        {
            // Loop over S
            for (label_t i = 0; i < S.size(); i++)
            {
                // Check if S[i] contains a substring of name
                if (S[i].find(name) != std::string_view::npos)
                {
                    // Split by space and remove whitespace
                    const std::vector<std::string> s = split(S[i], " "[0], true);

                    // Check that the last char is ;
                    // Perform the exit here if the above string is not equal to ;

                    const std::string toReturn = std::string(s[1].begin(), s[1].end() - 1);
                    // Check if T is integral type
                    if constexpr (std::is_integral_v<T>)
                    {
                        if (is_number(toReturn))
                        {
                            // Check if T is an unsigned integral type
                            if constexpr (std::is_unsigned_v<T>)
                            {
                                return static_cast<T>(std::stoul(toReturn));
                            }
                            // T must be a signed integral type
                            else
                            {
                                return static_cast<T>(std::stol(toReturn));
                            }
                        }
                    }
                    // Otherwise T must be a scalar type
                    else if constexpr (std::is_floating_point_v<T>)
                    {
                        return static_cast<T>(std::stold(toReturn));
                    }
                    else if constexpr (std::is_same_v<T, std::string>)
                    {
                        return toReturn;
                    }
                }
            }
            // Otherwise return 0
            // Should theoretically never get to this point because we have checked already that the string exists
            exceptions::program_exit(-1, "Parameter " + std::string(name) + " not found");
            return 0;
        }

        /**
         * @brief Parses a name-value pair
         * @param args The list of arguments to be searched
         * @param name The argument to be searched for
         * @return A std::string_view of the value argument corresponding to name
         **/
        [[nodiscard]] std::string_view parseNameValuePair(const std::vector<std::string> &args, const std::string_view &name) noexcept
        {
            // Loop over the input arguments and search for name
            for (label_t i = 0; i < args.size(); i++)
            {
                // The name argument exists, so handle it
                if (args[i] == name)
                {
                    // First check to see that i + 1 is not out of bounds
                    if (i + 1 < args.size())
                    {
                        return args[i + 1];
                    }
                    // Otherwise it is out of bounds: the supplied argument is the last argument and no value pair has been supplied
                    else
                    {
                        exceptions::program_exit(-1, "Input argument " + std::string(name) + std::string(" has not been supplied with a value; the correct syntax is -GPU 0,1 for example"));
                        return "";
                    }
                }
            }
            exceptions::program_exit(-1, "Input argument " + std::string(name) + std::string(" has not been supplied"));
            return "";
        }

        /**
         * @brief Parses the value of the argument following name
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         * @return A vector of integral type T
         * @note This function can be used to parse arguments passed to the executable on the command line such as -GPU 0,1
         **/
        template <typename T>
        [[nodiscard]] const std::vector<T> parseValue(const std::vector<std::string> &args, const std::string_view &name) noexcept
        {
            const std::vector<std::string> s_v = string::split(parseNameValuePair(args, name), ","[0], true);

            std::vector<T> arr;
            label_t arrLength = 0;

            for (label_t i = 0; i < s_v.size(); i++)
            {
                // Should check here if the string converts to a negative number and exit
                if constexpr (std::is_signed_v<T>)
                {
                    if (is_number(s_v[i]))
                    {
                        arr.push_back(std::stoi(std::string(s_v[i])));
                    }
                    else
                    {
                        exceptions::program_exit(-1, std::string(name) + std::string(" is not numeric"));
                    }
                }
                else
                {
                    if (is_number(s_v[i]))
                    {
                        arr.push_back(std::stoul(std::string(s_v[i])));
                    }
                    else
                    {
                        exceptions::program_exit(-1, std::string("Value supplied to argument ") + std::string(name) + std::string(" is not numeric"));
                    }
                }
                arrLength = arrLength + 1;
            }
            arr.resize(arrLength);

            return arr;
        }
    }
}

#endif