/**
Filename: latticeMesh.cuh
Contents: A class holding information about the solution grid
**/

#ifndef __MBLBM_LATTICEMESH_CUH
#define __MBLBM_LATTICEMESH_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"
#include "../programControl.cuh"

namespace LBM
{
    namespace host
    {
        class latticeMesh
        {
        public:
            /**
             * @brief Default-constructs a latticeMesh object
             * @return A latticeMesh object constructed from the caseInfo file
             * @note This constructor requires that a mesh file be present in the working directory
             * @note This constructor reads from the caseInfo file and is used primarily to construct the global mesh
             **/
            [[nodiscard]] latticeMesh() noexcept
                : nx_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "nx")),
                  ny_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "ny")),
                  nz_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "nz")),
                  nPoints_(nx_ * ny_ * nz_)
            {
                // #ifdef VERBOSE
                std::cout << "Allocated global latticeMesh object:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << nx_ << ";" << std::endl;
                std::cout << "    ny = " << ny_ << ";" << std::endl;
                std::cout << "    nz = " << nz_ << ";" << std::endl;
                std::cout << "};" << std::endl;
                std::cout << std::endl;
                // #endif

                // Allocate symbols on the GPU, temporary workaround
                checkCudaErrors(cudaMemcpyToSymbol(d_nx, &nx_, sizeof(label_t)));
                checkCudaErrors(cudaMemcpyToSymbol(d_ny, &ny_, sizeof(label_t)));
                checkCudaErrors(cudaMemcpyToSymbol(d_nz, &nz_, sizeof(label_t)));

                const label_t d_NUM_BLOCK_X_tmp = nx_ / block::nx();
                const label_t d_NUM_BLOCK_Y_tmp = ny_ / block::ny();
                const label_t d_NUM_BLOCK_Z_tmp = nz_ / block::nz();

                checkCudaErrors(cudaMemcpyToSymbol(d_NUM_BLOCK_X, &d_NUM_BLOCK_X_tmp, sizeof(label_t)));
                checkCudaErrors(cudaMemcpyToSymbol(d_NUM_BLOCK_Y, &d_NUM_BLOCK_Y_tmp, sizeof(label_t)));
                checkCudaErrors(cudaMemcpyToSymbol(d_NUM_BLOCK_Z, &d_NUM_BLOCK_Z_tmp, sizeof(label_t)));
            };

            /**
             * @brief Copy the defined symbols to the device constant memory
             **/
            void copyDeviceSymbols() const noexcept
            {
                const label_t d_NUM_BLOCK_X_tmp = nx_ / block::nx();
                const label_t d_NUM_BLOCK_Y_tmp = ny_ / block::ny();
                const label_t d_NUM_BLOCK_Z_tmp = nz_ / block::nz();

                // Allocate symbols on the GPU, temporary workaround
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(d_nx, &nx_, sizeof(label_t)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(d_ny, &ny_, sizeof(label_t)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(d_nz, &nz_, sizeof(label_t)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(d_NUM_BLOCK_X, &d_NUM_BLOCK_X_tmp, sizeof(label_t)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(d_NUM_BLOCK_Y, &d_NUM_BLOCK_Y_tmp, sizeof(label_t)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(d_NUM_BLOCK_Z, &d_NUM_BLOCK_Z_tmp, sizeof(label_t)));
                cudaDeviceSynchronize();
            }

            /**
             * @brief Returns the number of lattices in the x, y and z directions
             * @return Number of lattices as a label_t
             **/
            __device__ __host__ [[nodiscard]] inline constexpr label_t nx() const noexcept
            {
                return nx_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t ny() const noexcept
            {
                return ny_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nz() const noexcept
            {
                return nz_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nPoints() const noexcept
            {
                return nPoints_;
            }

            /**
             * @brief Returns the number of CUDA blocks in the x, y and z directions
             * @return Number of CUDA blocks as a label_t
             **/
            __device__ __host__ [[nodiscard]] inline constexpr label_t nxBlocks() const noexcept
            {
                return nx_ / block::nx();
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nyBlocks() const noexcept
            {
                return ny_ / block::ny();
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nzBlocks() const noexcept
            {
                return nz_ / block::nz();
            }
            __device__ __host__ [[nodiscard]] inline constexpr label_t nBlocks() const noexcept
            {
                return (nx_ / block::nx()) * (ny_ / block::ny()) * (nz_ / block::nz());
            }

            /**
             * @brief Returns the dimensions of a thread block in the x, y and z directions
             * @return Dimensions of a thread block as a dim3
             **/
            __device__ __host__ [[nodiscard]] inline constexpr dim3 threadBlock() const noexcept
            {
                return {block::nx(), block::ny(), block::nz()};
            }

            /**
             * @brief Returns the number of thread blocks in the x, y and z directions
             * @return Number of thread blocks as a dim3
             **/
            __device__ __host__ [[nodiscard]] inline constexpr dim3 gridBlock() const noexcept
            {
                return {static_cast<uint32_t>(nx_ / block::nx()), static_cast<uint32_t>(ny_ / block::ny()), static_cast<uint32_t>(nz_ / block::nz())};
            }

        private:
            /**
             * @brief The number of lattices in the x, y and z directions
             **/
            const label_t nx_;
            const label_t ny_;
            const label_t nz_;
            const label_t nPoints_;
        };
    }
}

#endif