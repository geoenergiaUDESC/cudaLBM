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
    File containing a list of all valid function object names

Namespace
    LBM::host

SourceFiles
    strainRateTensor.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_STRAINRATETENSOR_CUH
#define __MBLBM_STRAINRATETENSOR_CUH

namespace LBM
{
    namespace functionObjects
    {
        namespace StrainRateTensor
        {
            /**
             * @brief Calculates the strain rate tensor component
             * @param[in] uAlpha Velocity component in alpha direction
             * @param[in] uBeta Velocity component in beta direction
             * @param[in] mAlphaBeta Second order moment component
             * @return The calculated strain rate tensor component
             **/
            template <typename T>
            __device__ [[nodiscard]] inline constexpr T calculate(const T uAlpha, const T uBeta, const T mAlphaBeta) noexcept
            {
                return velocitySet::as2<T>() * ((uAlpha * uBeta) - mAlphaBeta) / (static_cast<T>(2) * device::tau);
            }

            /**
             * @brief CUDA kernel for calculating time-averaged strain rate tensor components
             * @param[in] u_Alpha Velocity field in alpha direction
             * @param[in] u_Beta Velocity field in beta direction
             * @param[in] m_AlphaBeta Second order moment field
             * @param[in,out] S_AlphaBetaMean Mean strain rate tensor component to update
             * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
             **/
            launchBounds __global__ void calculateMean_kernel(
                const scalar_t *const ptrRestrict u_Alpha,
                const scalar_t *const ptrRestrict u_Beta,
                const scalar_t *const ptrRestrict m_AlphaBeta,
                scalar_t *const ptrRestrict S_AlphaBetaMean,
                const scalar_t invNewCount)
            {
                // Calculate the index
                const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

                // Read from global memory
                const scalar_t uAlpha = u_Alpha[idx];
                const scalar_t uBeta = u_Beta[idx];
                const scalar_t mAlphaBeta = m_AlphaBeta[idx];
                const scalar_t S_AlphaBetaMean_ = S_AlphaBetaMean[idx];

                // Update the mean
                const scalar_t S_Mean = timeAverage(S_AlphaBetaMean_, calculate(uAlpha, uBeta, mAlphaBeta), invNewCount);

                // Write back to global
                S_AlphaBetaMean[idx] = S_Mean;
            }

            /**
             * @brief CUDA kernel for calculating instantaneous strain rate tensor components
             * @param[in] u_Alpha Velocity field in alpha direction
             * @param[in] u_Beta Velocity field in beta direction
             * @param[in] m_AlphaBeta Second order moment field
             * @param[out] SAlphaBeta Strain rate tensor component to update
             **/
            launchBounds __global__ void calculate_kernel(
                const scalar_t *const ptrRestrict u_Alpha,
                const scalar_t *const ptrRestrict u_Beta,
                const scalar_t *const ptrRestrict m_AlphaBeta,
                scalar_t *const ptrRestrict SAlphaBeta)
            {
                // Calculate the index
                const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

                // Read from global memory
                const scalar_t uAlpha = u_Alpha[idx];
                const scalar_t uBeta = u_Beta[idx];
                const scalar_t mAlphaBeta = m_AlphaBeta[idx];

                // Update the mean
                const scalar_t S = calculate(uAlpha, uBeta, mAlphaBeta);

                // Write back to global
                SAlphaBeta[idx] = S;
            }

            /**
             * @brief Class for managing strain rate tensor calculations in LBM simulations
             * @tparam VelocitySet The velocity set type used in LBM
             * @tparam N The number of streams (compile-time constant)
             **/
            template <class VelocitySet, const label_t N>
            class strainRateTensor
            {
            public:
                /**
                 * @brief Constructs a strain rate tensor object
                 * @param[in] mesh Reference to lattice mesh
                 * @param[in] devPtrs Device pointer collection for memory access
                 * @param[in] streamsLBM Stream handler for CUDA operations
                 **/
                __host__ [[nodiscard]] strainRateTensor(
                    const host::latticeMesh &mesh,
                    const device::ptrCollection<10, scalar_t> &devPtrs,
                    const streamHandler<N> &streamsLBM) noexcept
                    : mesh_(mesh),
                      devPtrs_(devPtrs),
                      streamsLBM_(streamsLBM),
                      calculate_(initialiserSwitch("S")),
                      calculateMean_(initialiserSwitch("SMean")),
                      xx_(objectAllocator<VelocitySet, time::instantaneous>("S_xx", mesh, calculate_)),
                      xy_(objectAllocator<VelocitySet, time::instantaneous>("S_xy", mesh, calculate_)),
                      xz_(objectAllocator<VelocitySet, time::instantaneous>("S_xz", mesh, calculate_)),
                      yy_(objectAllocator<VelocitySet, time::instantaneous>("S_yy", mesh, calculate_)),
                      yz_(objectAllocator<VelocitySet, time::instantaneous>("S_yz", mesh, calculate_)),
                      zz_(objectAllocator<VelocitySet, time::instantaneous>("S_zz", mesh, calculate_)),
                      xxMean_(objectAllocator<VelocitySet, time::timeAverage>("S_xxMean", mesh, calculateMean_)),
                      xyMean_(objectAllocator<VelocitySet, time::timeAverage>("S_xyMean", mesh, calculateMean_)),
                      xzMean_(objectAllocator<VelocitySet, time::timeAverage>("S_xzMean", mesh, calculateMean_)),
                      yyMean_(objectAllocator<VelocitySet, time::timeAverage>("S_yyMean", mesh, calculateMean_)),
                      yzMean_(objectAllocator<VelocitySet, time::timeAverage>("S_yzMean", mesh, calculateMean_)),
                      zzMean_(objectAllocator<VelocitySet, time::timeAverage>("S_zzMean", mesh, calculateMean_))
                {
                    // Set the cache config to prefer L1
                    checkCudaErrors(cudaFuncSetCacheConfig(calculate_kernel, cudaFuncCachePreferL1));
                    checkCudaErrors(cudaFuncSetCacheConfig(calculateMean_kernel, cudaFuncCachePreferL1));
                };

                /**
                 * @brief Default destructor
                 **/
                ~strainRateTensor() {};

                /**
                 * @brief Check if instantaneous calculation is enabled
                 * @return True if instantaneous calculation is enabled
                 **/
                __host__ inline constexpr bool calculate() const noexcept
                {
                    return calculate_;
                }

                /**
                 * @brief Check if mean calculation is enabled
                 * @return True if mean calculation is enabled
                 **/
                __host__ inline constexpr bool calculateMean() const noexcept
                {
                    return calculateMean_;
                }

                /**
                 * @brief Getter methods for instantaneous components
                 **/
                __host__ __device__ inline scalar_t *xx() noexcept { return xx_.ptr(); }
                __host__ __device__ inline scalar_t *xy() noexcept { return xy_.ptr(); }
                __host__ __device__ inline scalar_t *xz() noexcept { return xz_.ptr(); }
                __host__ __device__ inline scalar_t *yy() noexcept { return yy_.ptr(); }
                __host__ __device__ inline scalar_t *yz() noexcept { return yz_.ptr(); }
                __host__ __device__ inline scalar_t *zz() noexcept { return zz_.ptr(); }

                /**
                 * @brief Getter methods for mean components
                 **/
                __host__ __device__ inline scalar_t *xxMean() noexcept { return xxMean_.ptr(); }
                __host__ __device__ inline scalar_t *xyMean() noexcept { return xyMean_.ptr(); }
                __host__ __device__ inline scalar_t *xzMean() noexcept { return xzMean_.ptr(); }
                __host__ __device__ inline scalar_t *yyMean() noexcept { return yyMean_.ptr(); }
                __host__ __device__ inline scalar_t *yzMean() noexcept { return yzMean_.ptr(); }
                __host__ __device__ inline scalar_t *zzMean() noexcept { return zzMean_.ptr(); }

                /**
                 * @brief Calculate instantaneous strain rate tensor components
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculate([[maybe_unused]] const label_t timeStep) noexcept
                {
                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            StrainRateTensor::calculate_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<1>(), devPtrs_.ptr<1>(), devPtrs_.ptr<4>(), xx_.ptr());

                            StrainRateTensor::calculate_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<1>(), devPtrs_.ptr<2>(), devPtrs_.ptr<5>(), xy_.ptr());

                            StrainRateTensor::calculate_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<1>(), devPtrs_.ptr<3>(), devPtrs_.ptr<6>(), xz_.ptr());

                            StrainRateTensor::calculate_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<2>(), devPtrs_.ptr<2>(), devPtrs_.ptr<7>(), yy_.ptr());

                            StrainRateTensor::calculate_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<2>(), devPtrs_.ptr<3>(), devPtrs_.ptr<8>(), yz_.ptr());

                            StrainRateTensor::calculate_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<3>(), devPtrs_.ptr<3>(), devPtrs_.ptr<9>(), zz_.ptr());
                        });
                }

                /**
                 * @brief Calculate time-averaged strain rate tensor components
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateMean(const label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(timeStep + 1);

                    // Calculate the mean
                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            StrainRateTensor::calculateMean_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<1>(), devPtrs_.ptr<1>(), devPtrs_.ptr<4>(), xxMean_.ptr(), invNewCount);

                            StrainRateTensor::calculateMean_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<1>(), devPtrs_.ptr<2>(), devPtrs_.ptr<5>(), xyMean_.ptr(), invNewCount);

                            StrainRateTensor::calculateMean_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<1>(), devPtrs_.ptr<3>(), devPtrs_.ptr<6>(), xzMean_.ptr(), invNewCount);

                            StrainRateTensor::calculateMean_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<2>(), devPtrs_.ptr<2>(), devPtrs_.ptr<7>(), yyMean_.ptr(), invNewCount);

                            StrainRateTensor::calculateMean_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<2>(), devPtrs_.ptr<3>(), devPtrs_.ptr<8>(), yzMean_.ptr(), invNewCount);

                            StrainRateTensor::calculateMean_kernel<<<mesh_.gridBlock(), {block::nx(), block::ny(), block::nz()}, 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_.ptr<3>(), devPtrs_.ptr<3>(), devPtrs_.ptr<9>(), zzMean_.ptr(), invNewCount);
                        });
                }

                /**
                 * @brief Get the field name for instantaneous components
                 * @return Field name string
                 **/
                __host__ __device__ [[nodiscard]] inline constexpr const std::string &fieldName() const noexcept
                {
                    return fieldName_;
                }

                /**
                 * @brief Get the field name for mean components
                 * @return Field name string
                 **/
                __host__ __device__ [[nodiscard]] inline constexpr const std::string &fieldNameMean() const noexcept
                {
                    return fieldNameMean_;
                }

                /**
                 * @brief Get the component names for instantaneous tensor
                 * @return Vector of component names
                 **/
                __host__ __device__ [[nodiscard]] inline constexpr const std::vector<std::string> &componentNames() const noexcept
                {
                    return componentNames_;
                }

                /**
                 * @brief Get the component names for mean tensor
                 * @return Vector of component names
                 **/
                __host__ __device__ [[nodiscard]] inline constexpr const std::vector<std::string> &componentNamesMean() const noexcept
                {
                    return componentNamesMean_;
                }

            private:
                /**
                 * @brief Reference to lattice mesh
                 **/
                const host::latticeMesh &mesh_;
                /**
                 * @brief Device pointer collection
                 **/
                const device::ptrCollection<10, scalar_t> &devPtrs_;

                /**
                 * @brief Stream handler for CUDA operations
                 **/
                const streamHandler<N> &streamsLBM_;

                /**
                 * @brief Flag for instantaneous calculation
                 **/
                const bool calculate_;

                /**
                 * @brief Flag for mean calculation
                 **/
                const bool calculateMean_;

                /**
                 * @brief Instantaneous strain rate tensor components
                 **/
                device::array<scalar_t, VelocitySet, time::instantaneous> xx_;
                device::array<scalar_t, VelocitySet, time::instantaneous> xy_;
                device::array<scalar_t, VelocitySet, time::instantaneous> xz_;
                device::array<scalar_t, VelocitySet, time::instantaneous> yy_;
                device::array<scalar_t, VelocitySet, time::instantaneous> yz_;
                device::array<scalar_t, VelocitySet, time::instantaneous> zz_;

                /**
                 * @brief Time-averaged strain rate tensor components
                 **/
                device::array<scalar_t, VelocitySet, time::timeAverage> xxMean_;
                device::array<scalar_t, VelocitySet, time::timeAverage> xyMean_;
                device::array<scalar_t, VelocitySet, time::timeAverage> xzMean_;
                device::array<scalar_t, VelocitySet, time::timeAverage> yyMean_;
                device::array<scalar_t, VelocitySet, time::timeAverage> yzMean_;
                device::array<scalar_t, VelocitySet, time::timeAverage> zzMean_;

                /**
                 * @brief Field name for instantaneous components
                 **/
                const std::string fieldName_ = "S";

                /**
                 * @brief Field name for mean components
                 **/
                const std::string fieldNameMean_ = "SMean";

                /**
                 * @brief Instantaneous component names
                 **/
                const std::vector<std::string> componentNames_ = {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"};

                /**
                 * @brief Mean component names
                 **/
                const std::vector<std::string> componentNamesMean_ = {"S_xxMean", "S_xyMean", "S_xzMean", "S_yyMean", "S_yzMean", "S_zzMean"};

                /**
                 * @brief Initializes calculation switches based on function object configuration
                 * @param[in] objectName Name of the function object to check
                 * @return True if the object is enabled in configuration
                 **/
                __host__ [[nodiscard]] bool initialiserSwitch(const std::string &objectName)
                {
                    return string::containsString(string::trim<true>(string::eraseBraces(string::extractBlock(string::readFile("functionObjects"), "functionObjectList"))), objectName);
                }
            };
        }
    }
}

#endif