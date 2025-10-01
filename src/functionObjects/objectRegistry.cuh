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
    objectRegistry.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_OBJECTREGISTRY_CUH
#define __MBLBM_OBJECTREGISTRY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../strings.cuh"
#include "functionObjects.cuh"
#include "strainRateTensor.cuh"

namespace LBM
{
    /**
     * @brief Registry for managing function objects and their calculations
     * @tparam VelocitySet The velocity set type used in LBM simulations
     * @tparam N The number of streams (as a compile-time constant)
     **/
    template <class VelocitySet, const label_t N>
    class objectRegistry
    {
    public:
        /**
         * @brief Constructs an objectRegistry with mesh, device pointers and streams
         * @param[in] mesh Reference to the lattice mesh
         * @param[in] devPtrs Device pointer collection for memory management
         * @param[in] streamsLBM Stream handler for LBM operations
         **/
        objectRegistry(
            const host::latticeMesh &mesh,
            const device::ptrCollection<10, scalar_t> &devPtrs,
            const streamHandler<N> &streamsLBM)
            : mesh_(mesh),
              S_(
                  mesh,
                  devPtrs,
                  streamsLBM),
              functionVector_(functionObjectCallInitialiser(S_)),
              saveVector_(functionObjectSaveInitialiser(S_)) {};

        /**
         * @brief Default destructor
         **/
        ~objectRegistry() {};

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        inline void calculate(const label_t timeStep) noexcept
        {
            for (const auto &func : functionVector_)
            {
                func(timeStep); // Call each function with the timeStep
            }
        }

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        inline void save(const label_t timeStep) noexcept
        {
            for (const auto &save : saveVector_)
            {
                save(timeStep); // Call each function with the timeStep
            }
        }

    private:
        /**
         * @brief Reference to lattice mesh
         **/
        const host::latticeMesh &mesh_;

        /**
         * @brief Strain rate tensor function object
         **/
        functionObjects::StrainRateTensor::strainRateTensor<VelocitySet, N> S_;

        /**
         * @brief Registry of function objects to invoke
         **/
        const std::vector<std::function<void(const label_t)>> functionVector_;

        /**
         * @brief Initializes function calls based on strain rate tensor configuration
         * @param[in] S Reference to strain rate tensor object
         * @return Vector of function objects to be executed
         **/
        __host__ [[nodiscard]] const std::vector<std::function<void(const label_t)>> functionObjectCallInitialiser(
            functionObjects::StrainRateTensor::strainRateTensor<VelocitySet, N> &S) const noexcept
        {
            std::vector<std::function<void(const label_t)>> calls;

            if ((S.calculate()) && (S.calculateMean()))
            {
                calls.push_back(
                    [&S](const label_t label)
                    { S.calculateInstantaneousAndMean(label); });
            }
            else
            {
                if (S.calculate())
                {
                    calls.push_back(
                        [&S](const label_t label)
                        { S.calculateInstantaneous(label); });
                }

                if (S.calculateMean())
                {
                    calls.push_back(
                        [&S](const label_t label)
                        { S.calculateMean(label); });
                }
            }

            return calls;
        }

        /**
         * @brief Registry of function objects to save
         **/
        const std::vector<std::function<void(const label_t)>> saveVector_;

        /**
         * @brief Initializes save calls based on strain rate tensor configuration
         * @param[in] S Reference to strain rate tensor object
         * @return Vector of function objects to be executed
         **/
        __host__ [[nodiscard]] const std::vector<std::function<void(const label_t)>> functionObjectSaveInitialiser(
            functionObjects::StrainRateTensor::strainRateTensor<VelocitySet, N> &S) const noexcept
        {
            std::vector<std::function<void(const label_t)>> calls;

            if (S.calculate())
            {
                calls.push_back(
                    [&S](const label_t label)
                    { S.saveInstantaneous(label); });
            }

            if (S.calculateMean())
            {
                calls.push_back(
                    [&S](const label_t label)
                    { S.saveMean(label); });
            }

            return calls;
        }
    };
}

#endif