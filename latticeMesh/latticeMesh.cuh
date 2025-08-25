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
            [[nodiscard]] latticeMesh(const programControl &programCtrl) noexcept
                : nx_(string::extractParameter<label_t>(string::readFile("caseInfo"), "nx")),
                  ny_(string::extractParameter<label_t>(string::readFile("caseInfo"), "ny")),
                  nz_(string::extractParameter<label_t>(string::readFile("caseInfo"), "nz")),
                  nPoints_(nx_ * ny_ * nz_),
                  L_(programCtrl.L())
            {
#ifdef VERBOSE
                std::cout << "Allocated global latticeMesh object:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << nx_ << ";" << std::endl;
                std::cout << "    ny = " << ny_ << ";" << std::endl;
                std::cout << "    nz = " << nz_ << ";" << std::endl;
                std::cout << "};" << std::endl;
                std::cout << std::endl;
#endif

                // Perform a block dimensions safety check
                {
                    if (!(block::nx() * nxBlocks() == nx_))
                    {
                        errorHandler(ERR_SIZE, "block::nx() * mesh.nxBlocks() not equal to mesh.nx()\nMesh dimensions should be multiples of 32");
                    }
                    if (!(block::ny() * nyBlocks() == ny_))
                    {
                        errorHandler(ERR_SIZE, "block::ny() * mesh.nyBlocks() not equal to mesh.ny()\nMesh dimensions should be multiples of 32");
                    }
                    if (!(block::nz() * nzBlocks() == nz_))
                    {
                        errorHandler(ERR_SIZE, "block::nz() * mesh.nzBlocks() not equal to mesh.nz()\nMesh dimensions should be multiples of 32");
                    }
                    if (!(block::nx() * nxBlocks() * block::ny() * nyBlocks() * block::nz() * nzBlocks() == nx_ * ny_ * nz_))
                    {
                        errorHandler(ERR_SIZE, "block::nx() * nxBlocks() * block::ny() * nyBlocks() * block::nz() * nzBlocks() not equal to mesh.nPoints()\nMesh dimensions should be multiples of 32");
                    }
                }

                const scalar_t ReTemp = programCtrl.Re();
                const scalar_t u_infTemp = programCtrl.u_inf();
                const scalar_t viscosityTemp = programCtrl.u_inf() * static_cast<scalar_t>(nx_ - 1) / programCtrl.Re();
                const scalar_t tauTemp = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTemp;
                const scalar_t omegaTemp = static_cast<scalar_t>(1.0) / tauTemp;
                const scalar_t t_omegaVarTemp = static_cast<scalar_t>(1) - omegaTemp;
                const scalar_t omegaVar_d2Temp = omegaTemp * static_cast<scalar_t>(0.5);

                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::Re, &ReTemp, sizeof(device::Re)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::u_inf, &u_infTemp, sizeof(device::u_inf)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::tau, &tauTemp, sizeof(device::tau)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::omega, &omegaTemp, sizeof(device::omega)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::t_omegaVar, &t_omegaVarTemp, sizeof(device::t_omegaVar)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::omegaVar_d2, &omegaVar_d2Temp, sizeof(device::omegaVar_d2)));
                cudaDeviceSynchronize();

                const label_t nxBlocksTemp = nxBlocks();
                const label_t nyBlocksTemp = nyBlocks();
                const label_t nzBlocksTemp = nzBlocks();

                // Allocate symbols on the GPU
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::nx, &nx_, sizeof(device::nx)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::ny, &ny_, sizeof(device::ny)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::nz, &nz_, sizeof(device::nz)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::NUM_BLOCK_X, &nxBlocksTemp, sizeof(device::NUM_BLOCK_X)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::NUM_BLOCK_Y, &nyBlocksTemp, sizeof(device::NUM_BLOCK_Y)));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpyToSymbol(device::NUM_BLOCK_Z, &nzBlocksTemp, sizeof(device::NUM_BLOCK_Z)));
                cudaDeviceSynchronize();
            };

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
            __device__ __host__ [[nodiscard]] inline consteval dim3 threadBlock() const noexcept
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

            __host__ [[nodiscard]] inline constexpr const pointVector &L() const noexcept
            {
                return L_;
            }

        private:
            /**
             * @brief The number of lattices in the x, y and z directions
             **/
            const label_t nx_;
            const label_t ny_;
            const label_t nz_;
            const label_t nPoints_;

            const pointVector L_;
        };
    }
}

#endif