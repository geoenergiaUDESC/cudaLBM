#include "computeVersion.cuh"

using namespace LBM;

template <const bool verboseOutput = false>
[[nodiscard]] const std::string getEnvironmentVariable(const std::string &envVariable, const std::string &defaultName)
{
    const char *const env_ptr = std::getenv(envVariable.c_str());

    if (env_ptr == nullptr)
    {
        if constexpr (verboseOutput)
        {
            std::cout << envVariable << ": " << defaultName << std::endl;
        }
        return defaultName;
    }
    else
    {
        if constexpr (verboseOutput)
        {
            std::cout << envVariable << ": " << env_ptr << std::endl;
        }
        return env_ptr;
    }
}

template <const bool verboseOutput = false>
[[nodiscard]] const std::string getEnvironmentVariable(const std::string &envVariable)
{
    const char *const env_ptr = std::getenv(envVariable.c_str());

    if (env_ptr == nullptr)
    {
        const std::string errorString = "Error: " + envVariable + " environment variable is not set." + "Please run:" + "  source ~/.bashrc" + "or add it to your environment.";
        throw std::runtime_error(errorString);
    }

    if constexpr (verboseOutput)
    {
        std::cout << envVariable << ": " << env_ptr << std::endl;
    }

    return env_ptr;
}

int main()
{
    const deviceIndex_t deviceCount = countDevices();

    const std::string CUDALBM_ARCHITECTURE_DETECTION = getEnvironmentVariable("CUDALBM_ARCHITECTURE_DETECTION", "Automatic");
    const std::string CUDALBM_ARCHITECTURE_VERSION = getEnvironmentVariable("CUDALBM_ARCHITECTURE_VERSION");
    const std::string CUDALBM_BUILD_DIR = getEnvironmentVariable("CUDALBM_BUILD_DIR");
    const std::string CUDALBM_BIN_DIR = getEnvironmentVariable("CUDALBM_BIN_DIR");
    const std::string CUDALBM_INCLUDE_DIR = getEnvironmentVariable("CUDALBM_INCLUDE_DIR");

    // Now handle the file path setup
    // If the path doesn't exist, create it
    if (!std::filesystem::exists(CUDALBM_BUILD_DIR))
    {
        std::filesystem::create_directory(CUDALBM_BUILD_DIR);
    }
    if (!std::filesystem::exists(CUDALBM_INCLUDE_DIR))
    {
        std::filesystem::create_directory(CUDALBM_INCLUDE_DIR);
    }
    if (!std::filesystem::exists(CUDALBM_BIN_DIR))
    {
        std::filesystem::create_directory(CUDALBM_BIN_DIR);
    }

    const std::filesystem::path outputFilePath = CUDALBM_INCLUDE_DIR + "/hardware.info";
    std::ofstream outputFile(outputFilePath);

    const time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    outputFile << "# Hardware information file automatically generated" << std::endl;
    outputFile << "# Compiled on: " << compileTimestamp() << std::endl;
    outputFile << "# Executed on: " << std::put_time(std::localtime(&time_now), "%Y-%m-%d %H:%M:%S") << std::endl;
    outputFile << std::endl;

    if (CUDALBM_ARCHITECTURE_DETECTION == "Manual")
    {
        std::cout << "Doing manual detection" << std::endl;
        const std::string all_arch_flags = "-gencode arch=compute_" + CUDALBM_ARCHITECTURE_VERSION + ",code=sm_" + CUDALBM_ARCHITECTURE_VERSION;
        const std::string all_lto_flags = "-gencode arch=compute_" + CUDALBM_ARCHITECTURE_VERSION + ",code=lto_" + CUDALBM_ARCHITECTURE_VERSION;
        outputFile << "# Consolidated architecture flags for all found GPUs (no duplicates)" << std::endl;
        outputFile << "NVCXX_ALL_ARCHFLAGS = " << all_arch_flags << " " << all_lto_flags << std::endl;
    }
    else
    {
        std::cout << "Doing automatic detection" << std::endl;
        std::string all_arch_flags = "";

        for (int i = 0; i < deviceCount; ++i)
        {
            const cudaDeviceProp props = getDeviceProperties(i);

            const std::string current_arch_flag =
                "-gencode arch=compute_" +
                std::to_string(props.major) +
                std::to_string(props.minor) +
                ",code=sm_" +
                std::to_string(props.major) +
                std::to_string(props.minor);

            const std::string current_lto_flag =
                " -gencode arch=compute_" +
                std::to_string(props.major) +
                std::to_string(props.minor) +
                ",code=lto_" +
                std::to_string(props.major) +
                std::to_string(props.minor);

            if (all_arch_flags.find(current_arch_flag) == std::string::npos)
            {
                all_arch_flags += current_arch_flag;
            }

            if (all_arch_flags.find(current_lto_flag) == std::string::npos)
            {
                all_arch_flags += current_lto_flag;
            }

            outputFile << "# Properties for CUDA Device ID: " << i << std::endl;
            outputFile << "GPU_NAME_" << i << " = " << props.name << std::endl;
            outputFile << "GPU_ARCH_MAJOR_" << i << " = " << props.major << std::endl;
            outputFile << "GPU_ARCH_MINOR_" << i << " = " << props.minor << std::endl;
            outputFile << "GPU_GLOBAL_MEM_MB_" << i << " = " << props.totalGlobalMem / (1024 * 1024) << std::endl;
            outputFile << "GPU_SHARED_MEM_PER_BLOCK_KB_" << i << " = " << props.sharedMemPerBlock / 1024 << std::endl;
            outputFile << "GPU_REGS_PER_BLOCK_" << i << " = " << props.regsPerBlock << std::endl;
            outputFile << "GPU_MAX_THREADS_PER_BLOCK_" << i << " = " << props.maxThreadsPerBlock << std::endl;
            outputFile << "GPU_MULTIPROCESSOR_COUNT_" << i << " = " << props.multiProcessorCount << std::endl;
            outputFile << std::endl;

            outputFile << "# Consolidated architecture flags for all found GPUs (no duplicates)" << std::endl;
            outputFile << "NVCXX_ALL_ARCHFLAGS = " << all_arch_flags << std::endl;
        }
    }

    outputFile.close();

    std::cout << "Successfully generated '" << outputFilePath.string() << "' with info for " << deviceCount << " CUDA device(s)." << std::endl;

    return 0;
}