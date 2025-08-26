/**
Filename: fileInput.cuh
Contents: Implementation of reading solution variables encoded in binary format
**/

#ifndef __MBLBM_FILEINPUT_CUH
#define __MBLBM_FILEINPUT_CUH

namespace LBM
{
    namespace fileIO
    {
        /**
         * @brief  Structure to hold parsed header information
         **/
        struct fieldFileHeader
        {
            const bool isLittleEndian;                 // Endianness of the binary data
            const std::size_t scalarSize;              // 4 or 8 bytes
            const std::size_t nx;                      // Grid dimensions
            const std::size_t ny;                      // Grid dimensions
            const std::size_t nz;                      // Grid dimensions
            const std::size_t nVars;                   // Number of variables
            const std::size_t dataStartPos;            // File position of binary data start
            const std::vector<std::string> fieldNames; // Names of the field
        };

        /**
         * @brief Trims whitespace from a string
         * @param str The string to be trimmed
         * @return The trimmed string
         **/
        [[nodiscard]] const std::string trim(const std::string &str)
        {
            const std::size_t start = str.find_first_not_of(" \t\r\n");
            const std::size_t end = str.find_last_not_of(" \t\r\n;");
            return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
        }

        /**
         * @brief Parse header metadata from file
         * @param fileName The name of the file to be parsed
         * @return The parsed header information
         **/
        [[nodiscard]] const fieldFileHeader parseFieldFileHeader(const std::string &fileName)
        {
            // Check if file exists and is accessible
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error("File does not exist: " + fileName);
            }

            std::ifstream in(fileName, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Get file size for validation
            in.seekg(0, std::ios::end);
            const auto fileSizePos = in.tellg();
            if (fileSizePos == -1)
            {
                throw std::runtime_error("Cannot determine file size");
            }

            // Safe cast: file size should be non-negative
            if (fileSizePos < 0)
            {
                throw std::runtime_error("Invalid file size (negative)");
            }

            const std::size_t fileSize = static_cast<std::size_t>(fileSizePos);
            in.seekg(0, std::ios::beg);

            std::string line;
            bool inSystemInfo = false;
            bool inFieldData = false;
            bool inFieldInfo = false;
            bool inFieldNames = false;
            bool isLittleEndian = false;

            // Variables to store parsed data
            std::size_t scalarSize = 0;
            std::size_t nx = 0;
            std::size_t ny = 0;
            std::size_t nz = 0;
            std::size_t nVars = 0;
            std::size_t totalPoints = 0;
            std::size_t dataStartPos = 0;
            std::vector<std::string> fieldNamesVec;
            std::size_t expectedFieldCount = 0;
            bool foundFieldData = false;
            bool foundSystemInfo = false;
            bool foundFieldInfo = false;

            // Track which sections we've already seen to detect duplicates
            bool systemInfoSeen = false;
            bool fieldDataSeen = false;
            bool fieldInfoSeen = false;

            // Track line number for better error messages
            std::size_t lineNumber = 0;

            while (std::getline(in, line))
            {
                lineNumber++;
                line = trim(line);
                if (line.empty())
                {
                    continue;
                }

                // Detect sections (regardless of order)
                if (line == "systemInformation")
                {
                    if (systemInfoSeen)
                    {
                        throw std::runtime_error("Duplicate systemInformation section at line " + std::to_string(lineNumber));
                    }
                    inSystemInfo = true;
                    foundSystemInfo = true;
                    systemInfoSeen = true;
                    continue;
                }
                else if (line == "fieldData")
                {
                    if (fieldDataSeen)
                    {
                        throw std::runtime_error("Duplicate fieldData section at line " + std::to_string(lineNumber));
                    }
                    inFieldData = true;
                    fieldDataSeen = true;
                    continue;
                }
                else if (line == "fieldInformation")
                {
                    if (fieldInfoSeen)
                    {
                        throw std::runtime_error("Duplicate fieldInformation section at line " + std::to_string(lineNumber));
                    }
                    inFieldInfo = true;
                    foundFieldInfo = true;
                    fieldInfoSeen = true;
                    continue;
                }

                // Parse systemInformation section
                if (inSystemInfo)
                {
                    if (line == "}")
                    {
                        inSystemInfo = false;
                    }
                    else if (line.find("binaryType") != std::string::npos)
                    {
                        isLittleEndian = (line.find("littleEndian") != std::string::npos);
                    }
                    else if (line.find("scalarType") != std::string::npos)
                    {
                        if (line.find("32 bit") != std::string::npos)
                        {
                            scalarSize = 4;
                        }
                        else if (line.find("64 bit") != std::string::npos)
                        {
                            scalarSize = 8;
                        }
                        else
                        {
                            throw std::runtime_error("Invalid scalarType at line " + std::to_string(lineNumber));
                        }
                    }
                }

                // Parse fieldData section for dimensions
                if (inFieldData && !foundFieldData)
                {
                    if (line == "}")
                    {
                        inFieldData = false;
                    }
                    else if (line.find("field[") != std::string::npos)
                    {
                        // Extract dimensions from pattern: field[total][nx][ny][nz][nVars]
                        std::vector<std::size_t> dims;
                        std::size_t pos = 0;

                        while ((pos = line.find('[', pos)) != std::string::npos)
                        {
                            const std::size_t end = line.find(']', pos);
                            if (end == std::string::npos)
                            {
                                throw std::runtime_error("Unclosed bracket at line " + std::to_string(lineNumber));
                            }

                            try
                            {
                                const std::string dimStr = line.substr(pos + 1, end - pos - 1);
                                const unsigned long long dimValue = std::stoull(dimStr);

                                // Check for overflow before casting to size_t
                                if (dimValue > std::numeric_limits<std::size_t>::max())
                                {
                                    throw std::runtime_error("Dimension value too large at line " + std::to_string(lineNumber));
                                }

                                dims.push_back(static_cast<std::size_t>(dimValue));
                            }
                            catch (const std::out_of_range &)
                            {
                                throw std::runtime_error("Dimension value out of range at line " + std::to_string(lineNumber));
                            }
                            catch (...)
                            {
                                throw std::runtime_error("Invalid dimension format at line " + std::to_string(lineNumber));
                            }
                            pos = end + 1;
                        }

                        if (dims.size() < 5)
                        {
                            throw std::runtime_error("Invalid field dimensions at line " + std::to_string(lineNumber));
                        }

                        totalPoints = dims[0];
                        nVars = dims[1];
                        nz = dims[2];
                        ny = dims[3];
                        nx = dims[4];

                        // Validate dimensions
                        if (nx == 0 || ny == 0 || nz == 0 || nVars == 0)
                        {
                            throw std::runtime_error("Invalid grid dimensions: nx, ny, nz, nVars cannot be zero at line " + std::to_string(lineNumber));
                        }

                        // Check for potential overflow in multiplication
                        if (nx > std::numeric_limits<std::size_t>::max() / ny / nz / nVars)
                        {
                            throw std::runtime_error("Dimension product would overflow at line " + std::to_string(lineNumber));
                        }

                        if (totalPoints != nx * ny * nz * nVars)
                        {
                            throw std::runtime_error("Dimension mismatch at line " + std::to_string(lineNumber) + ": total points (" + std::to_string(totalPoints) + ") != nx * ny * nz * nVars (" + std::to_string(nx * ny * nz * nVars) + ")");
                        }

                        // Skip next line (contains "{")
                        if (!std::getline(in, line))
                        {
                            throw std::runtime_error("Unexpected end of file after field declaration");
                        }
                        lineNumber++;

                        // Record start position of binary data with safety check
                        const auto dataPos = in.tellg();
                        if (dataPos == -1)
                        {
                            throw std::runtime_error("Error getting file position");
                        }

                        if (dataPos < 0)
                        {
                            throw std::runtime_error("Invalid file position (negative)");
                        }

                        // Check for overflow before casting to size_t
                        if (static_cast<unsigned long long>(dataPos) > std::numeric_limits<std::size_t>::max())
                        {
                            throw std::runtime_error("File position too large for size_t");
                        }

                        dataStartPos = static_cast<std::size_t>(dataPos);

                        // Check if data start position is within file bounds
                        if (dataStartPos > fileSize)
                        {
                            throw std::runtime_error("Data start position exceeds file size");
                        }

                        foundFieldData = true;
                        inFieldData = false;
                    }
                }

                // Parse fieldInformation section for field names
                if (inFieldInfo)
                {
                    if (line == "}")
                    {
                        inFieldInfo = false;
                    }
                    else if (line.find("fieldNames[") != std::string::npos)
                    {
                        // Extract expected number of field names
                        const std::size_t startBracket = line.find('[');
                        const std::size_t endBracket = line.find(']');
                        if (startBracket == std::string::npos || endBracket == std::string::npos)
                        {
                            throw std::runtime_error("Invalid fieldNames format at line " + std::to_string(lineNumber));
                        }
                        try
                        {
                            const unsigned long long count = std::stoull(
                                line.substr(startBracket + 1, endBracket - startBracket - 1));

                            // Check for overflow before casting to size_t
                            if (count > std::numeric_limits<std::size_t>::max())
                            {
                                throw std::runtime_error("Field names count too large at line " + std::to_string(lineNumber));
                            }

                            expectedFieldCount = static_cast<std::size_t>(count);
                            if (expectedFieldCount == 0)
                            {
                                throw std::runtime_error("Field names count cannot be zero at line " + std::to_string(lineNumber));
                            }
                        }
                        catch (const std::out_of_range &)
                        {
                            throw std::runtime_error("Field names count out of range at line " + std::to_string(lineNumber));
                        }
                        catch (...)
                        {
                            throw std::runtime_error("Invalid fieldNames count at line " + std::to_string(lineNumber));
                        }
                        // Enter fieldNames block
                        inFieldNames = true;

                        // Skip the opening brace line
                        if (!std::getline(in, line))
                        {
                            throw std::runtime_error("Unexpected end of file after fieldNames declaration");
                        }
                        lineNumber++;
                        line = trim(line);
                        if (line != "{")
                        {
                            throw std::runtime_error("Expected opening brace after fieldNames declaration at line " + std::to_string(lineNumber));
                        }
                    }
                    else if (inFieldNames)
                    {
                        if (line == "}")
                        {
                            inFieldNames = false;
                            inFieldInfo = false;
                        }
                        else if (line != "{") // Skip the opening brace if we encounter it
                        {
                            // Remove trailing semicolon and trim to get field name
                            if (line.back() == ';')
                            {
                                line.pop_back();
                            }
                            const std::string fieldName = trim(line);

                            // Validate field name
                            if (fieldName.empty())
                            {
                                throw std::runtime_error("Empty field name at line " + std::to_string(lineNumber));
                            }

                            // Check for duplicate field names
                            if (std::find(fieldNamesVec.begin(), fieldNamesVec.end(), fieldName) != fieldNamesVec.end())
                            {
                                throw std::runtime_error("Duplicate field name '" + fieldName + "' at line " + std::to_string(lineNumber));
                            }

                            fieldNamesVec.push_back(fieldName);

                            // Check if we've exceeded the expected number of field names
                            if (fieldNamesVec.size() > expectedFieldCount)
                            {
                                throw std::runtime_error("Too many field names at line " + std::to_string(lineNumber) + ". Expected: " + std::to_string(expectedFieldCount));
                            }
                        }
                    }
                }

                // Early exit if we've collected all necessary information
                if (scalarSize > 0 && foundFieldData && fieldNamesVec.size() == expectedFieldCount && expectedFieldCount > 0)
                {
                    break;
                }
            }

            // Final validation checks
            if (!foundSystemInfo)
            {
                throw std::runtime_error("Missing systemInformation section");
            }

            if (!foundFieldInfo)
            {
                throw std::runtime_error("Missing fieldInformation section");
            }

            if (!foundFieldData)
            {
                throw std::runtime_error("Missing fieldData section");
            }

            if (scalarSize == 0)
            {
                throw std::runtime_error("Missing or invalid scalarType in systemInformation");
            }

            if (fieldNamesVec.size() != nVars)
            {
                throw std::runtime_error("Field names count (" + std::to_string(fieldNamesVec.size()) + ") does not match nVars (" + std::to_string(nVars) + ")");
            }

            if (fieldNamesVec.size() != expectedFieldCount)
            {
                throw std::runtime_error("Field names count (" + std::to_string(fieldNamesVec.size()) + ") does not match declared count (" + std::to_string(expectedFieldCount) + ")");
            }

            // Check if binary data size matches expectations
            // Check for potential overflow in multiplication
            if (totalPoints > std::numeric_limits<std::size_t>::max() / scalarSize)
            {
                throw std::runtime_error("Data size calculation would overflow");
            }

            const std::size_t expectedDataSize = totalPoints * scalarSize;
            if (fileSize - dataStartPos < expectedDataSize)
            {
                throw std::runtime_error("Insufficient data in file. Expected " + std::to_string(expectedDataSize) + " bytes, but only " + std::to_string(fileSize - dataStartPos) + " bytes available");
            }

            return {isLittleEndian, scalarSize, nx, ny, nz, nVars, dataStartPos, fieldNamesVec};
        }

        /**
         * @brief Swap endianness for a single value
         * @param value The value to be swapped
         **/
        template <typename T>
        void swapEndian(T &value)
        {
            char *bytes = reinterpret_cast<char *>(&value);
            for (std::size_t i = 0; i < sizeof(T) / 2; ++i)
            {
                std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
            }
        }

        /**
         * @brief Swap endianness for all values in a vector
         * @param data The vector to be swapped
         **/
        template <typename T>
        void swapEndianVector(std::vector<T> &data)
        {
            for (T &value : data)
            {
                swapEndian(value);
            }
        }

        /**
         * @brief Read a file and return the data contained
         * @param fileName The name of the file to be read
         **/
        template <typename T>
        [[nodiscard]] const std::vector<T> readFieldFile(const std::string &fileName)
        {
            static_assert(std::is_floating_point_v<T>, "T must be floating point");
            static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big, "System must be little or big endian");

            // Check that the file exists - if it doesn't, throw an exception
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error("File does not exist: " + fileName);
            }

            // Parse header metadata
            const fieldFileHeader header = parseFieldFileHeader(fileName);

            // Validate scalar size
            if (sizeof(T) != header.scalarSize)
            {
                throw std::runtime_error("Scalar size mismatch between file and template type");
            }

            // Calculate expected data size
            const std::size_t totalPoints = header.nx * header.ny * header.nz;
            const std::size_t totalDataCount = totalPoints * header.nVars;

            // Open file and jump to binary data
            std::ifstream in(fileName, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Safe conversion for seekg
            if (header.dataStartPos > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()))
            {
                throw std::runtime_error("File position overflow");
            }
            in.seekg(static_cast<std::streamoff>(header.dataStartPos));

            // Read binary data
            std::vector<T> data(totalDataCount);
            const std::size_t byteCount = totalDataCount * sizeof(T);

            // Check for streamsize overflow
            if (byteCount > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Data size exceeds maximum stream size");
            }
            in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(byteCount));

            if (!in.good() || in.gcount() != static_cast<std::streamsize>(byteCount))
            {
                throw std::runtime_error("Error reading binary data");
            }

            // Handle endianness conversion if needed
            const bool systemIsLittle = (std::endian::native == std::endian::little);
            if (systemIsLittle != header.isLittleEndian)
            {
                swapEndianVector(data);
            }

            return data;
        }

        template <typename T>
        [[nodiscard]] const std::vector<T> readFieldByName(const std::string &fileName, const std::string &fieldName)
        {
            static_assert(std::is_floating_point_v<T>, "T must be floating point");
            static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big, "System must be little or big endian");

            // Check if file exists
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error("File does not exist: " + fileName);
            }

            // Get file size for validation
            std::ifstream sizeCheck(fileName, std::ios::binary | std::ios::ate);
            if (!sizeCheck)
            {
                throw std::runtime_error("Cannot open file for size check: " + fileName);
            }
            const std::size_t fileSize = static_cast<std::size_t>(sizeCheck.tellg());
            sizeCheck.close();

            // Parse header to get file structure and field names
            const fieldFileHeader header = parseFieldFileHeader(fileName);

            // Validate scalar size
            if (sizeof(T) != header.scalarSize)
            {
                throw std::runtime_error("Scalar size mismatch between file (" + std::to_string(header.scalarSize) + " bytes) and template type (" + std::to_string(sizeof(T)) + " bytes)");
            }

            // Find the requested field name
            const auto it = std::find(header.fieldNames.begin(), header.fieldNames.end(), fieldName);
            if (it == header.fieldNames.end())
            {
                // Create a list of available field names for better error message
                std::string availableFields;
                for (const auto &name : header.fieldNames)
                {
                    if (!availableFields.empty())
                    {
                        availableFields += ", ";
                    }
                    availableFields += "'" + name + "'";
                }
                throw std::runtime_error("Field name '" + fieldName + "' not found in file. Available fields: " + availableFields);
            }

            // Calculate field index and data position - FIXED sign conversion issue
            const std::ptrdiff_t signedFieldIndex = std::distance(header.fieldNames.begin(), it);
            if (signedFieldIndex < 0)
            {
                throw std::runtime_error("Internal error: Negative field index");
            }

            const std::size_t fieldIndex = static_cast<std::size_t>(signedFieldIndex);
            const std::size_t pointsPerField = header.nx * header.ny * header.nz;

            // Check for potential overflow in calculations
            if (pointsPerField == 0)
            {
                throw std::runtime_error("Invalid field dimensions: points per field is zero");
            }

            // Check if fieldIndex is valid
            if (fieldIndex >= header.nVars)
            {
                throw std::runtime_error("Field index out of range");
            }

            // Check for overflow in fieldOffset calculation
            if (fieldIndex > (std::numeric_limits<std::size_t>::max() / pointsPerField / sizeof(T)))
            {
                throw std::runtime_error("Field offset calculation would overflow");
            }

            const std::size_t fieldOffset = fieldIndex * pointsPerField * sizeof(T);

            // Check if fieldOffset would exceed file bounds
            if (fieldOffset > (std::numeric_limits<std::size_t>::max() - header.dataStartPos))
            {
                throw std::runtime_error("Field start position calculation would overflow");
            }

            const std::size_t fieldStartPos = header.dataStartPos + fieldOffset;

            // Check if field data would extend beyond file end
            if (fieldStartPos > fileSize)
            {
                throw std::runtime_error("Field start position is beyond file end");
            }

            const std::size_t fieldByteSize = pointsPerField * sizeof(T);

            // Check if field data would extend beyond file end
            if (fieldStartPos > (std::numeric_limits<std::size_t>::max() - fieldByteSize))
            {
                throw std::runtime_error("Field end position calculation would overflow");
            }

            if (fieldStartPos + fieldByteSize > fileSize)
            {
                throw std::runtime_error("Field data extends beyond file end");
            }

            // Open file and jump to field data
            std::ifstream in(fileName, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            // Check for position overflow
            if (fieldStartPos > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()))
            {
                throw std::runtime_error("Field position overflow");
            }

            in.seekg(static_cast<std::streamoff>(fieldStartPos));
            if (!in.good())
            {
                throw std::runtime_error("Failed to seek to field position");
            }

            // Read field data
            std::vector<T> fieldData(pointsPerField);
            const std::size_t byteCount = fieldByteSize;

            // Check for streamsize overflow
            if (byteCount > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Field data size exceeds maximum stream size");
            }

            in.read(reinterpret_cast<char *>(fieldData.data()), static_cast<std::streamsize>(byteCount));

            if (!in.good())
            {
                throw std::runtime_error("Error reading field data: stream not good after read");
            }

            if (in.gcount() != static_cast<std::streamsize>(byteCount))
            {
                throw std::runtime_error("Incomplete field data read. Expected " + std::to_string(byteCount) + " bytes, got " + std::to_string(in.gcount()) + " bytes");
            }

            // Handle endianness conversion if needed
            const bool systemIsLittle = (std::endian::native == std::endian::little);
            if (systemIsLittle != header.isLittleEndian)
            {
                swapEndianVector(fieldData);
            }

            return fieldData;
        }

        /**
         * @brief De-interleave an array of structures (AoS) into a structure of arrays (SoA).
         * @param fMom The array of structures to be de-interleaved (AoS).
         * @param mesh The mesh
         * @return A vector of vectors, where each inside vector contains all the values for a unique variable (SoA).
         **/
        template <typename T, class M>
        [[nodiscard]] const std::vector<std::vector<T>> deinterleaveAoSOptimized(const std::vector<T> &fMom, const M &mesh)
        {
            const std::size_t nNodes = static_cast<std::size_t>(mesh.nx()) * mesh.ny() * mesh.nz();

            // Safety check for the size of fMom and nPoints
            if (fMom.size() % nNodes != 0)
            {
                throw std::invalid_argument("fMom size (" + std::to_string(fMom.size()) + ") is not divisible by mesh points (" + std::to_string(nNodes) + ")");
            }

            const std::size_t nFields = fMom.size() / nNodes;

            std::vector<std::vector<T>> soa(nFields, std::vector<scalar_t>(nNodes, 0));

            for (label_t bz = 0; bz < mesh.nzBlocks(); bz++)
            {
                for (label_t by = 0; by < mesh.nyBlocks(); by++)
                {
                    for (label_t bx = 0; bx < mesh.nxBlocks(); bx++)
                    {
                        for (label_t tz = 0; tz < block::nz(); tz++)
                        {
                            for (label_t ty = 0; ty < block::ny(); ty++)
                            {
                                for (label_t tx = 0; tx < block::nx(); tx++)
                                {
                                    const label_t x = (bx * block::nx()) + tx;
                                    const label_t y = (by * block::ny()) + ty;
                                    const label_t z = (bz * block::nz()) + tz;

                                    const label_t idxGlobal = host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny());
                                    const label_t idx = host::idx(tx, ty, tz, bx, by, bz, mesh);

                                    for (label_t field = 0; field < nFields; field++)
                                    {
                                        soa[field][idxGlobal] = fMom[idx + (field * mesh.nPoints())];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return soa;
        }

    }

}

#endif