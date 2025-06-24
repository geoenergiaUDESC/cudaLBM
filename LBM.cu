#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
#include "array/array.cuh"
#include "programControl.cuh"
#include "latticeMesh/latticeMesh.cuh"
#include "moments/moments.cuh"
#include "collision.cuh"
#include "postProcess.cuh"
#include "momentBasedD3Q19.cuh"
#include "fieldAverage.cuh"
#include "cavity.cuh"
#include "fileIO/fileIO.cuh"

using namespace LBM;

// Structure to hold parsed header information
struct FieldFileHeader
{
    bool isLittleEndian;
    size_t scalarSize;   // 4 or 8 bytes
    size_t nx, ny, nz;   // Grid dimensions
    size_t nVars;        // Number of variables
    size_t dataStartPos; // File position of binary data start
};

// Helper: Trim whitespace from string
std::string trim(const std::string &str)
{
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n;");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// Parse header metadata from file
FieldFileHeader parseFieldFileHeader(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file: " + filename);

    FieldFileHeader header;
    std::string line;
    bool inSystemInfo = false;
    bool inFieldData = false;

    while (std::getline(in, line))
    {
        line = trim(line);
        if (line.empty())
            continue;

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
                header.isLittleEndian = (line.find("littleEndian") != std::string::npos);
            }
            else if (line.find("scalarType") != std::string::npos)
            {
                header.scalarSize = (line.find("32 bit") != std::string::npos) ? 4 : 8;
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
                std::vector<size_t> dims;
                size_t pos = 0;

                while ((pos = line.find('[', pos)) != std::string::npos)
                {
                    size_t end = line.find(']', pos);
                    if (end == std::string::npos)
                        break;

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

                header.nx = dims[1];
                header.ny = dims[2];
                header.nz = dims[3];
                header.nVars = dims[4];

                // Skip next line (contains "{")
                std::getline(in, line);

                // Record start position of binary data
                header.dataStartPos = static_cast<size_t>(in.tellg());
                break;
            }
        }
    }

    if (header.scalarSize == 0 || header.nVars == 0)
    {
        throw std::runtime_error("Incomplete header information");
    }

    return header;
}

// Swap endianness for a single value
template <typename T>
void swapEndian(T &value)
{
    char *bytes = reinterpret_cast<char *>(&value);
    for (size_t i = 0; i < sizeof(T) / 2; ++i)
    {
        std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
    }
}

// Swap endianness for all values in a vector
template <typename T>
void swapEndianVector(std::vector<T> &data)
{
    for (T &value : data)
    {
        swapEndian(value);
    }
}

// Main reader function
template <typename T>
std::vector<T> readFieldFile(const std::string &filename)
{
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(std::endian::native == std::endian::little ||
                      std::endian::native == std::endian::big,
                  "System must be little or big endian");

    // Parse header metadata
    FieldFileHeader header = parseFieldFileHeader(filename);

    // Validate scalar size
    if (sizeof(T) != header.scalarSize)
    {
        throw std::runtime_error("Scalar size mismatch between file and template type");
    }

    // Calculate expected data size
    const size_t totalPoints = header.nx * header.ny * header.nz;
    const size_t totalDataCount = totalPoints * header.nVars;

    // Open file and jump to binary data
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file: " + filename);

    // Safe conversion for seekg
    if (header.dataStartPos > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()))
    {
        throw std::runtime_error("File position overflow");
    }
    in.seekg(static_cast<std::streamoff>(header.dataStartPos));

    // Read binary data
    std::vector<T> data(totalDataCount);
    const size_t byteCount = totalDataCount * sizeof(T);

    // Check for streamsize overflow
    if (byteCount > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
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
[[nodiscard]] std::vector<std::vector<T>> deinterleaveAoS(
    const std::vector<T> &fMom,
    const host::latticeMesh &mesh)
{
    return {
        postProcess::save<index::rho()>(fMom.data(), mesh),
        postProcess::save<index::u()>(fMom.data(), mesh),
        postProcess::save<index::v()>(fMom.data(), mesh),
        postProcess::save<index::w()>(fMom.data(), mesh),
        postProcess::save<index::xx()>(fMom.data(), mesh),
        postProcess::save<index::xy()>(fMom.data(), mesh),
        postProcess::save<index::xz()>(fMom.data(), mesh),
        postProcess::save<index::yy()>(fMom.data(), mesh),
        postProcess::save<index::yz()>(fMom.data(), mesh),
        postProcess::save<index::zz()>(fMom.data(), mesh)};
}

namespace LBM
{
    namespace host
    {
        template <typename T>
        class array
        {
        public:
            [[nodiscard]] array(const std::string &fileName)
                : arr_(readFieldFile<T>(fileName)) {};
            ~array() {};

        private:
            const std::vector<T> arr_;
        };
    }
}

int main(int argc, char *argv[])
{
    const host::latticeMesh mesh;

    const programControl programCtrl(argc, argv);

    // Set cuda device
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform device memory allocation
    device::array<scalar_t> moments(
        host::moments(mesh, programCtrl.u_inf()),
        {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        mesh);

    device::array<scalar_t> momentsMean(
        host::moments(mesh, programCtrl.u_inf()),
        {"rhoMean", "uMean", "vMean", "wMean", "m_xxMean", "m_xyMean", "m_xzMean", "m_yyMean", "m_yzMean", "m_zzMean"},
        mesh);

    const device::array<nodeType_t> nodeTypes(host::nodeType(mesh), {"nodeTypes"}, mesh);

    device::halo blockHalo(host::moments(mesh, programCtrl.u_inf()), mesh);

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy symbols to device
    mesh.copyDeviceSymbols();
    programCtrl.copyDeviceSymbols(mesh.nx());

    for (label_t timeStep = 0; timeStep < programCtrl.nt(); timeStep++)
    {
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << std::endl;
        }

        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, 0>>>(
            moments.ptr(),
            nodeTypes.ptr(),
            blockHalo);

        fieldAverage::calculate<<<mesh.gridBlock(), mesh.threadBlock(), 0, 0>>>(
            moments.ptr(),
            momentsMean.ptr(),
            nodeTypes.ptr(),
            timeStep);

        blockHalo.swap();

        if (programCtrl.save(timeStep))
        {
            host::write(
                "latticeMesh_" + std::to_string(timeStep) + ".LBMBin",
                moments,
                timeStep);
        }
    }

    std::cout << "End" << std::endl;

    return 0;
}