#include "computeVersion.cuh"

using namespace LBM;

int main()
{
    const deviceIndex_t deviceCount = countDevices();

    const std::filesystem::path outputFilePath = "hardware.info";
    std::ofstream outputFile(outputFilePath);

    const time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    outputFile << "# Hardware information file automatically generated" << std::endl;
    outputFile << "# Compiled on: " << compileTimestamp() << std::endl;
    outputFile << "# Executed on: " << std::put_time(std::localtime(&time_now), "%Y-%m-%d %H:%M:%S") << std::endl;
    outputFile << std::endl;

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

        if (all_arch_flags.find(current_arch_flag) == std::string::npos)
        {
            all_arch_flags += current_arch_flag;
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
    }

    outputFile << "# Consolidated architecture flags for all found GPUs (no duplicates)" << std::endl;
    outputFile << "NVCXX_ALL_ARCHFLAGS = " << all_arch_flags << std::endl;

    outputFile.close();

    std::cout << "Successfully generated '" << outputFilePath.string() << "' with info for " << deviceCount << " CUDA device(s)." << std::endl;

    return 0;
}