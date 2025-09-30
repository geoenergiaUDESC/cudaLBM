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
    Functions employed throughout the source code to manipulate strings

Namespace
    LBM

SourceFiles
    strings.cuh

\*---------------------------------------------------------------------------*/

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

#ifndef __MBLBM_STRINGS_CUH
#define __MBLBM_STRINGS_CUH

namespace LBM
{
    namespace string
    {
        __host__ [[nodiscard]] bool containsString(const std::vector<std::string> &vec, const std::string &target)
        {
            return std::find(vec.begin(), vec.end(), target) != vec.end();
        }

        /**
         * @brief Trims leading and trailing whitespace from a string.
         * @param str The input string to trim.
         * @return Trimmed string, or empty string if only whitespace.
         * @note Handles space, tab, newline, carriage return, form feed, and vertical tab.
         **/
        template <const bool trimSemicolon>
        [[nodiscard]] const std::string trim(const std::string &str)
        {
            const std::size_t start = str.find_first_not_of(" \t\n\r\f\v");

            if (start == std::string::npos)
            {
                return "";
            }

            if constexpr (trimSemicolon)
            {
                return str.substr(start, str.find_last_not_of(" \t\n\r\f\v;") - start + 1);
            }
            else
            {
                return str.substr(start, str.find_last_not_of(" \t\n\r\f\v") - start + 1);
            }
        }

        template <const bool trimSemicolon>
        [[nodiscard]] const std::vector<std::string> trim(const std::vector<std::string> &str)
        {
            std::vector<std::string> strTrimmed(str.size(), "");

            for (std::size_t s = 0; s < str.size(); s++)
            {
                strTrimmed[s] = trim<trimSemicolon>(str[s]);
            }

            return strTrimmed;
        }

        /**
         * @brief Removes C++-style comments from a string.
         * @param str The input string to process.
         * @return String with comments removed (everything after '//').
         * @note Only handles single-line comments starting with '//'.
         **/
        [[nodiscard]] const std::string removeComments(const std::string &str)
        {
            const std::size_t commentPos = str.find("//");
            if (commentPos != std::string::npos)
            {
                return str.substr(0, commentPos);
            }
            return str;
        }

        /**
         * @brief Checks if a string contains only whitespace characters.
         * @param str The string to check.
         * @return true if string contains only whitespace, false otherwise.
         * @note Uses std::isspace for whitespace detection.
         **/
        [[nodiscard]] bool isOnlyWhitespace(const std::string &str)
        {
            for (char c : str)
            {
                if (!std::isspace(static_cast<unsigned char>(c)))
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Finds the line number where a block declaration starts.
         * @param lines Vector of strings representing the source lines.
         * @param blockName The name of the block to find (e.g., "boundaryField").
         * @param startLine Line number to start searching from (default: 0).
         * @return Line number where the block declaration was found.
         * @throws std::runtime_error if block is not found.
         * @note Handles various declaration styles including braces and semicolons.
         **/
        [[nodiscard]] std::size_t findBlockLine(const std::vector<std::string> &lines, const std::string &blockName, const std::size_t startLine = 0)
        {
            for (std::size_t i = startLine; i < lines.size(); ++i)
            {
                const std::string processedLine = removeComments(lines[i]);
                const std::string trimmedLine = trim<false>(processedLine);

                // Check if line starts with the block name
                if (trimmedLine.compare(0, blockName.length(), blockName) == 0)
                {
                    // Extract the rest of the line after the block name
                    const std::string rest = trim<false>(trimmedLine.substr(blockName.length()));

                    // Check if the next token is empty or a brace
                    if (rest.empty() || rest == "{" || rest[0] == '{')
                    {
                        return i;
                    }

                    // Check if the next token is a semicolon (for field declarations)
                    if (rest[0] == ';')
                    {
                        return i;
                    }

                    // For cases like "internalField" which might be followed by other content
                    std::istringstream iss(rest);
                    std::string nextToken;
                    iss >> nextToken;

                    if (nextToken.empty() || nextToken == "{" || nextToken == ";")
                    {
                        return i;
                    }
                }
            }

            throw std::runtime_error("Field block \"" + blockName + "\" not defined");
        }

        /**
         * @brief Extracts a complete block (with braces) from source lines.
         * @param lines Vector of strings representing the source lines.
         * @param blockName The name of the block to extract.
         * @param startLine Line number to start searching from (default: 0).
         * @return Vector of strings containing the complete block including braces.
         * @throws std::runtime_error for malformed blocks or unbalanced braces.
         * @note Preserves original formatting including comments in the returned block.
         **/
        [[nodiscard]] const std::vector<std::string> extractBlock(
            const std::vector<std::string> &lines,
            const std::string &blockName,
            const std::size_t startLine = 0)
        {
            std::vector<std::string> result;

            // Find the block line using the helper function
            const std::size_t blockLine = findBlockLine(lines, blockName, startLine);

            // Check for non-whitespace content between block declaration and opening brace
            bool foundOpeningBrace = false;
            int braceCount = 0;
            std::size_t openingBraceLine = 0;

            // First, check if the opening brace is on the same line as the block declaration
            const std::string blockLineProcessed = removeComments(lines[blockLine]);
            std::size_t openBracePos = blockLineProcessed.find('{');
            if (openBracePos != std::string::npos)
            {
                // Opening brace is on the same line as block declaration
                braceCount = 1;
                result.push_back(lines[blockLine]);
                foundOpeningBrace = true;
                openingBraceLine = blockLine;
            }
            else
            {
                // Check subsequent lines for the opening brace
                for (std::size_t i = blockLine + 1; i < lines.size(); ++i)
                {
                    const std::string processedLine = removeComments(lines[i]);
                    const std::string trimmedLine = trim<false>(processedLine);

                    // Check for closing brace before opening brace
                    if (processedLine.find('}') != std::string::npos)
                    {
                        throw std::runtime_error("Found closing brace before opening brace for block '" + blockName + "'");
                    }

                    // Check for non-whitespace content
                    if (!isOnlyWhitespace(trimmedLine) && trimmedLine.find('{') == std::string::npos)
                    {
                        throw std::runtime_error("Non-whitespace content found between block declaration and opening brace for block '" + blockName + "'");
                    }

                    // Check for opening brace
                    openBracePos = processedLine.find('{');
                    if (openBracePos != std::string::npos)
                    {
                        braceCount = 1;
                        result.push_back(lines[i]);
                        foundOpeningBrace = true;
                        openingBraceLine = i;
                        break;
                    }

                    // If we reach here, the line contains only whitespace or comments
                    // We don't add these lines to the result yet
                }
            }

            if (!foundOpeningBrace)
            {
                throw std::runtime_error("Opening brace not found for block '" + blockName + "'");
            }

            // Continue processing from the line after the opening brace
            for (std::size_t i = openingBraceLine + 1; i < lines.size(); ++i)
            {
                // Process each line to count braces, but keep the original line in the result
                const std::string processedLineInner = removeComments(lines[i]);
                for (char c : processedLineInner)
                {
                    if (c == '{')
                    {
                        braceCount++;
                    }
                    else if (c == '}')
                    {
                        braceCount--;
                    }
                }
                result.push_back(lines[i]);

                if (braceCount == 0)
                {
                    return result;
                }
            }

            // If we reach here, the braces are unbalanced
            throw std::runtime_error("Unbalanced braces for block '" + blockName + "'");
        }

        /**
         * @brief Extracts a field-specific block using a combined key-field identifier.
         * @param lines Vector of strings representing the source lines.
         * @param fieldName The field name (e.g., "p" for pressure).
         * @param key The block type key (e.g., "internalField").
         * @return Vector of strings containing the complete block.
         * @note Convenience wrapper for extractBlock(lines, key + " " + fieldName).
         **/
        [[nodiscard]] const std::vector<std::string> extractBlock(
            const std::vector<std::string> &lines,
            const std::string &fieldName,
            const std::string &key)
        {
            return extractBlock(lines, key + " " + fieldName);
        }

        /**
         * @brief Reads the caseInfo file in the current directory into a vector of strings
         * @return A std::vector of std::string_view objects contained within the caseInfo file
         * @note This function will cause the program to exit if caseInfo is not found in the launch directory
         **/
        [[nodiscard]] const std::vector<std::string> readFile(const std::string_view &fileName)
        {
            // Does the file even exist?
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error(std::string(fileName) + std::string(" file not opened"));
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
         * @brief Checks that the input string is numeric.
         * @param s The string_view object which is to be checked.
         * @return True if s is a valid number, false otherwise.
         * @note A valid number can optionally start with a '+' or '-' sign and may contain one decimal point.
         **/
        [[nodiscard]] bool is_number(const std::string &s) noexcept
        {
            if (s.empty())
            {
                return false;
            }

            std::string::const_iterator it = s.begin();

            // Check for optional sign
            if (*it == '+' || *it == '-')
            {
                ++it;
                // If string is just a sign, it's not a valid number
                if (it == s.end())
                {
                    return false;
                }
            }

            bool has_digits = false;
            bool has_decimal = false;

            // Process each character
            while (it != s.end())
            {
                if (std::isdigit(*it))
                {
                    has_digits = true;
                    ++it;
                }
                else if (*it == '.')
                {
                    // Only one decimal point allowed
                    if (has_decimal)
                    {
                        return false;
                    }
                    has_decimal = true;
                    ++it;
                }
                else
                {
                    // Invalid character found
                    return false;
                }
            }

            // Must have at least one digit and if there's a decimal point,
            // there must be digits after it (handled by the iteration)
            return has_digits;
        }

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
         * @brief Splits the string_view object s according to the delimiter delim
         * @param s The string_view object which is to be split
         * @param delim The delimiter character by which s is split, e.g. a comma, space, etc
         * @param removeWhitespace Controls the removal of whitespace; removes blank spaces from the return value if true (default true)
         * @return A std::vector of std::string_view objects split from s by delim
         * @note This function can be used to, for example, split a string by commas, spaces, etc
         **/
        [[nodiscard]] const std::vector<std::string> split(const std::string_view &s, const char delim, const bool removeWhitespace = true) noexcept
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

        [[nodiscard]] const std::string extractParameterLine(const std::vector<std::string> &S, const std::string_view &name)
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

                    return std::string(s[1].begin(), s[1].end() - 1);
                }
            }

            // Otherwise return 0
            // Should theoretically never get to this point because we have checked already that the string exists

            throw std::runtime_error("Parameter " + std::string(name) + " not found");

            return "";
        }

        /**
         * @brief Searches for an entry corresponding to variableName within the vector of strings S
         * @param T The type of variable returned
         * @param S The vector of strings which is searched
         * @param name The name of the variable which is to be found and returned as type T
         * @return The value of the variable expressed as a type T
         * @note This function can be used to, for example, read an entry of nx within caseInfo after caseInfo has been loaded into S
         * @note The line containing the definition of variableName must separate variableName and its value with a space, for instance nx 128;
         **/
        template <typename T>
        [[nodiscard]] T extractParameter(const std::vector<std::string> &S, const std::string_view &name)
        {
            // First get the parameter line string
            const std::string toReturn = extractParameterLine(S, name);

            // Is it supposed an integral value?
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
            // Is it supposed a floating ponit value?
            else if constexpr (std::is_floating_point_v<T>)
            {
                return static_cast<T>(std::stold(toReturn));
            }
            // Is it supposed a string?
            else if constexpr (std::is_same_v<T, std::string>)
            {
                return toReturn;
            }

            return 0;
        }

        template <typename T>
        [[nodiscard]] T extractParameter(const std::string &toReturn)
        {
            // Is it supposed an integral value?
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
            // Is it supposed a floating ponit value?
            else if constexpr (std::is_floating_point_v<T>)
            {
                return static_cast<T>(std::stold(toReturn));
            }
            // Is it supposed a string?
            else if constexpr (std::is_same_v<T, std::string>)
            {
                return toReturn;
            }

            return 0;
        }

        /**
         * @brief Parses a name-value pair
         * @param args The list of arguments to be searched
         * @param name The argument to be searched for
         * @return A std::string_view of the value argument corresponding to name
         **/
        [[nodiscard]] std::string_view parseNameValuePair(const std::vector<std::string> &args, const std::string_view &name)
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
                        throw std::runtime_error("Input argument " + std::string(name) + std::string(" has not been supplied with a value; the correct syntax is -GPU 0,1 for example"));
                        return "";
                    }
                }
            }
            throw std::runtime_error("Input argument " + std::string(name) + std::string(" has not been supplied"));
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
        [[nodiscard]] const std::vector<T> parseValue(const std::vector<std::string> &args, const std::string_view &name)
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
                        throw std::runtime_error(std::string(name) + std::string(" is not numeric"));
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
                        throw std::runtime_error(std::string("Value supplied to argument ") + std::string(name) + std::string(" is not numeric"));
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