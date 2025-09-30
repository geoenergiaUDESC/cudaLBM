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
    template <class VelocitySet>
    class objectRegistry
    {
    public:
        objectRegistry(const host::latticeMesh &mesh)
            : mesh_(mesh),
              S_(
                  mesh,
                  string::containsString(string::trim<true>(eraseBraces(string::extractBlock(string::readFile("functionObjects"), "functionObjectList"))), "S"),
                  string::containsString(string::trim<true>(eraseBraces(string::extractBlock(string::readFile("functionObjects"), "functionObjectList"))), "SMean"))
        {
            if (string::containsString(string::trim<true>(eraseBraces(string::extractBlock(string::readFile("functionObjects"), "functionObjectList"))), "S"))
            {
                std::cout << "Allocated S" << std::endl;
            }
            else
            {
                std::cout << "Did not allocate S" << std::endl;
            }

            if (string::containsString(string::trim<true>(eraseBraces(string::extractBlock(string::readFile("functionObjects"), "functionObjectList"))), "SMean"))
            {
                std::cout << "Allocated SMean" << std::endl;
            }
            else
            {
                std::cout << "Did not allocate SMean" << std::endl;
            }
        };

        ~objectRegistry() {};

        template <const label_t N>
        inline void calculate(const device::ptrCollection<10, scalar_t> &devPtrs, const label_t timeStep, const streamHandler<N> &streamsLBM) noexcept
        {
            S_.calculate(devPtrs, timeStep, streamsLBM);
        }

        __host__ [[nodiscard]] inline const functionObjects::StrainRateTensor::strainRateTensor<VelocitySet> &S() const noexcept
        {
            return S_;
        }

        __host__ [[nodiscard]] inline functionObjects::StrainRateTensor::strainRateTensor<VelocitySet> &S() noexcept
        {
            return S_;
        }

        // inline void save(const device::ptrCollection<6, scalar_t> &SPtrs, const device::ptrCollection<6, scalar_t> &SMeanPtrs, const label_t timeStep) const noexcept
        // {
        //     if (S_.calculate())
        //     {
        //         fileIO::writeFile<time::instantaneous>(
        //             S_.fieldName() + "_" + std::to_string(timeStep) + ".LBMBin",
        //             mesh_,
        //             S_.componentNames(),
        //             host::toHost(SPtrs, mesh_),
        //             timeStep);
        //     }

        //     if (S_.calculateMean())
        //     {
        //         fileIO::writeFile<time::timeAverage>(
        //             S_.fieldNameMean() + "_" + std::to_string(timeStep) + ".LBMBin",
        //             mesh_,
        //             S_.componentNamesMean(),
        //             host::toHost(SMeanPtrs, mesh_),
        //             timeStep);
        //     }
        // }

    private:
        const host::latticeMesh &mesh_;

        functionObjects::StrainRateTensor::strainRateTensor<VelocitySet> S_;

        __host__ [[nodiscard]] const std::string catenate(const std::vector<std::string> S) noexcept
        {
            std::string s;
            for (std::size_t line = 0; line < S.size(); line++)
            {
                s = s + S[line] + "\n";
            }
            return s;
        }

        __host__ [[nodiscard]] const std::vector<std::string> eraseBraces(const std::vector<std::string> lines) noexcept
        {
            if (!(lines.size() > 2))
            {
                errorHandler(-1, "Lines must have at least 2 entries: opening bracket and closing bracket. Problematic entry:" + catenate(lines));
            }

            // Need to check that lines has > 2 elements, i.e. more than just empty brackets
            std::vector<std::string> newLines(lines.size() - 2);

            for (std::size_t line = 1; line < lines.size() - 1; line++)
            {
                newLines[line - 1] = lines[line];
            }

            return newLines;
        }
    };
}

#endif