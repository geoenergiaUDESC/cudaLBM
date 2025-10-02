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
    Function definitions and includes specific to the fieldConvert executable

Namespace
    LBM

SourceFiles
    fieldConvert.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FIELDCONVERT_CUH
#define __MBLBM_FIELDCONVERT_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/array/array.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/postProcess/postProcess.cuh"
#include "../../src/inputControl.cuh"
#include "../../src/functionObjects/objectRegistry.cuh"

namespace LBM
{
    using VelocitySet = D3Q19;

    using WriterFunction = void (*)(
        const std::vector<std::vector<scalar_t>> &,
        const std::string &,
        const host::latticeMesh &,
        const std::vector<std::string> &);

    /**
     * @brief Unordered map of the writer types to the appropriate functions
     **/
    const std::unordered_map<std::string, WriterFunction> writers = {
        {"vtu", postProcess::writeVTU},
        {"vts", postProcess::writeVTS},
        {"tecplot", postProcess::writeTecplot}};

    /**
     * @brief Creates an error message for invalid writer types
     * @param[in] writerNames Unordered map of the writer types to the appropriate functions
     * @param[in] conversion The invalid conversion type provided by the user
     * @return A formatted error message listing the supported formats
     **/
    __host__ [[nodiscard]] const std::string invalidWriter(const std::unordered_map<std::string, WriterFunction> &writerNames, const std::string &conversion) noexcept
    {
        std::vector<std::string> supportedFormats;
        for (const auto &pair : writerNames)
        {
            supportedFormats.push_back(pair.first);
        }

        // Sort them alphabetically
        std::sort(supportedFormats.begin(), supportedFormats.end());

        // Create the error message with supported formats
        std::string errorMsg = "Unsupported conversion format: " + conversion + "\nSupported formats are: ";
        for (std::size_t i = 0; i < supportedFormats.size(); ++i)
        {
            if (i != 0)
            {
                errorMsg += ", ";
            }
            errorMsg += supportedFormats[i];
        }

        return errorMsg;
    }

    /**
     * @brief Returns the field names based on the provided prefix and whether a custom field is specified
     * @param[in] fileNamePrefix The prefix for the field names
     * @param[in] doCustomField Boolean indicating if a custom field is specified
     * @return A reference to a vector of field names
     * @throws std::runtime_error if an invalid field name is provided
     **/
    __host__ [[nodiscard]] host::arrayCollection<scalar_t, ctorType::MUST_READ, velocitySet> initialiseArrays(
        const std::string &fileNamePrefix,
        const programControl &programCtrl,
        const std::vector<std::string> &fieldNames,
        const label_t timeStep,
        const bool doCustomField)
    {
        // Construct from a custom field name
        if (doCustomField)
        {
            return host::arrayCollection<scalar_t, ctorType::MUST_READ, velocitySet>(fileNamePrefix, fieldNames, timeStep);
        }
        // Otherwise construct from default field names
        else
        {
            return host::arrayCollection<scalar_t, ctorType::MUST_READ, velocitySet>(programCtrl, fieldNames, timeStep);
        }
    }

    /**
     * @brief Returns the field names based on the provided prefix and whether a custom field is specified
     * @param[in] fileNamePrefix The prefix for the field names
     * @param[in] doCustomField Boolean indicating if a custom field is specified
     * @return A reference to a vector of field names
     * @throws std::runtime_error if an invalid field name is provided
     **/
    __host__ [[nodiscard]] const std::vector<std::string> &getFieldNames(
        const std::string &fileNamePrefix,
        const bool doCustomField)
    {
        if (!doCustomField)
        {
            return functionObjects::solutionVariableNames;
        }
        else
        {
            const std::unordered_map<std::string, std::vector<std::string>>::const_iterator namesIterator = functionObjects::fieldComponentsMap.find(fileNamePrefix);
            const bool foundField = namesIterator != functionObjects::fieldComponentsMap.end();
            if (!foundField)
            {
                // Throw an exception: invalid field name
                throw std::runtime_error("Invalid argument passed to -fieldName");
            }
            else
            {
                return namesIterator->second;
            }
        }
    }
}

#endif