#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"
#include "../collision/collision.cuh"
#include "../blockHalo/blockHalo.cuh"
#include "../fileIO/fileIO.cuh"
#include "../runTimeIO/runTimeIO.cuh"
#include "../postProcess.cuh"

#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>

// Helper function to trim whitespace from a string
[[nodiscard]] const std::string trim(const std::string &str)
{
    const std::size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos)
    {
        return "";
    }

    const std::size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

// Helper function to remove comments from a string
[[nodiscard]] const std::string removeComments(const std::string &str)
{
    const std::size_t commentPos = str.find("//");
    if (commentPos != std::string::npos)
    {
        return str.substr(0, commentPos);
    }
    return str;
}

// Helper function to check if a string contains only whitespace
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

// Helper function to find the line number of a block declaration
[[nodiscard]] std::size_t findBlockLine(const std::vector<std::string> &lines, const std::string &blockName, const std::size_t startLine = 0)
{
    for (std::size_t i = startLine; i < lines.size(); ++i)
    {
        const std::string processedLine = removeComments(lines[i]);
        const std::string trimmedLine = trim(processedLine);

        // Check if line starts with the block name
        if (trimmedLine.compare(0, blockName.length(), blockName) == 0)
        {
            // Extract the rest of the line after the block name
            const std::string rest = trim(trimmedLine.substr(blockName.length()));

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

// General function to extract a block by name
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
            const std::string trimmedLine = trim(processedLine);

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

// Specialized function for field blocks (for backward compatibility)
[[nodiscard]] const std::vector<std::string> extractBlock(
    const std::vector<std::string> &lines,
    const std::string &fieldName,
    const std::string &key)
{
    return extractBlock(lines, key + " " + fieldName);
}