/**
Filename: fileOutput.cuh
Contents: Implementation of writing solution variables encoded in binary format
**/

#ifndef __MBLBM_FILEOUTPUT_CUH
#define __MBLBM_FILEOUTPUT_CUH

namespace LBM
{
    namespace fileIO
    {
        /**
         * @brief Implementation of the writing of the binary file
         * @param fileName Name of the file to be written
         * @param mesh The mesh
         * @param varNames The names of the solution variables
         * @param fields The solution variables encoded in interleaved AoS format
         * @param timeStep The current time step
         **/
        template <typename T, class M>
        __host__ void writeFile(
            const std::string &filename,
            const M &mesh,
            const std::vector<std::string> &varNames,
            const std::vector<T> &fields,
            const std::size_t timeStep)
        {
            static_assert(std::is_floating_point<T>::value, "T must be floating point");

            static_assert(std::endian::native == std::endian::little | std::endian::native == std::endian::big, "File system must be either little or big endian");

            static_assert(sizeof(scalar_t) == 4 | sizeof(scalar_t) == 8, "Error writing file: scalar_t must be either 32 or 64 bit");

            const std::size_t nVars = varNames.size();
            const std::size_t nPoints = static_cast<std::size_t>(mesh.nx()) * static_cast<std::size_t>(mesh.ny()) * static_cast<std::size_t>(mesh.nz());
            const std::size_t expectedSize = nPoints * nVars;

            if (fields.size() != expectedSize)
            {
                throw std::invalid_argument("Data vector size mismatch");
            }

            std::ofstream out(filename, std::ios::binary);
            if (!out)
            {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            // Write the system information: binary endianness
            out << "systemInformation" << std::endl;
            out << "{" << std::endl;
            if constexpr (std::endian::native == std::endian::little)
            {
                out << "\tbinaryType\tlittleEndian;" << std::endl;
            }
            else if constexpr (std::endian::native == std::endian::big)
            {
                out << "\tbinaryType\tbigEndian;" << std::endl;
            }
            out << std::endl;
            if constexpr (sizeof(scalar_t) == 4)
            {
                out << "\tscalarType\t32 bit;" << std::endl;
            }
            else if constexpr (sizeof(scalar_t) == 8)
            {
                out << "\tscalarType\t64 bit;" << std::endl;
            }
            out << "};" << std::endl;
            out << std::endl;

            // Write the field information: instantaneous or time-averaged, field names
            out << "fieldInformation" << std::endl;
            out << "{" << std::endl;
            out << "\ttimeStep\t" << timeStep << ";" << std::endl;
            out << std::endl;
            // For now, only writing instantaneous fields
            out << "\ttimeType\tinstantaneous;" << std::endl;
            out << std::endl;
            out << "\tfieldNames[" << nVars << "]" << std::endl;
            out << "\t{" << std::endl;
            for (const auto &name : varNames)
            {
                out << "\t\t" << name << ";" << std::endl;
            }
            out << "\t};" << std::endl;
            out << "};" << std::endl;
            out << std::endl;

            // Write binary data with safe size conversion
            const std::size_t byteSize = fields.size() * sizeof(T);

            if (byteSize > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::runtime_error("Data size exceeds maximum stream size");
            }

            out << "fieldData" << std::endl;
            out << "{" << std::endl;
            out << "\tfieldType\tnonUniform;" << std::endl;
            out << std::endl;
            out << "\tfield[" << expectedSize << "][" << mesh.nx() << "][" << mesh.ny() << "][" << mesh.nz() << "][" << nVars << "]" << std::endl;
            out << "\t{" << std::endl;
            // out.flush();
            out.write(reinterpret_cast<const char *>(fields.data()), static_cast<std::streamsize>(byteSize));
            out << std::endl;
            out << "\t};" << std::endl;
            out << "};" << std::endl;
        }
    }
}

#endif