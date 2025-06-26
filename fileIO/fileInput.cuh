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
            const bool isLittleEndian;
            const std::size_t scalarSize;   // 4 or 8 bytes
            const std::size_t nx, ny, nz;   // Grid dimensions
            const std::size_t nVars;        // Number of variables
            const std::size_t dataStartPos; // File position of binary data start
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
            std::ifstream in(fileName, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Cannot open file: " + fileName);
            }

            std::string line;
            bool inSystemInfo = false;
            bool inFieldData = false;
            bool isLittleEndian = false;

            // Dummy local variables
            std::size_t scalarSize = 0;
            std::size_t nx = 0;
            std::size_t ny = 0;
            std::size_t nz = 0;
            std::size_t nVars = 0;
            std::size_t dataStartPos = 0;

            while (std::getline(in, line))
            {
                line = trim(line);
                if (line.empty())
                {
                    continue;
                }

                // Detect sections
                if (line == "systemInformation")
                {
                    inSystemInfo = true;
                    continue;
                }
                if (line == "fieldData")
                {
                    inFieldData = true;
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
                        scalarSize = (line.find("32 bit") != std::string::npos) ? 4 : 8;
                    }
                }

                // Parse fieldData section for dimensions
                if (inFieldData)
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
                                break;
                            }

                            try
                            {
                                dims.push_back(std::stoull(line.substr(pos + 1, end - pos - 1)));
                            }
                            catch (...)
                            {
                                throw std::runtime_error("Invalid dimension format");
                            }
                            pos = end + 1;
                        }

                        if (dims.size() < 5)
                        {
                            throw std::runtime_error("Invalid field dimensions");
                        }

                        nx = dims[1];
                        ny = dims[2];
                        nz = dims[3];
                        nVars = dims[4];

                        // Skip next line (contains "{")
                        std::getline(in, line);

                        // Record start position of binary data
                        dataStartPos = static_cast<std::size_t>(in.tellg());
                        break;
                    }
                }
            }

            if (scalarSize == 0 || nVars == 0)
            {
                throw std::runtime_error("Incomplete header information");
            }

            return {isLittleEndian, scalarSize, nx, ny, nz, nVars, dataStartPos};
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

        /**
         * @brief De-interleave an array of structures into a structure of arrays
         * @param fMom The array of structures to be de-interleaved
         * @param mesh The mesh
         **/
        template <typename T, class M>
        [[nodiscard]] std::vector<std::vector<T>> deinterleaveAoS(const std::vector<T> &fMom, const M &mesh)
        {
            return {
                host::save<index::rho()>(fMom.data(), mesh),
                host::save<index::u()>(fMom.data(), mesh),
                host::save<index::v()>(fMom.data(), mesh),
                host::save<index::w()>(fMom.data(), mesh),
                host::save<index::xx()>(fMom.data(), mesh),
                host::save<index::xy()>(fMom.data(), mesh),
                host::save<index::xz()>(fMom.data(), mesh),
                host::save<index::yy()>(fMom.data(), mesh),
                host::save<index::yz()>(fMom.data(), mesh),
                host::save<index::zz()>(fMom.data(), mesh)};
        }
    }
}

#endif