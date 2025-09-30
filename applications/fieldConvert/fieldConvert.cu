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
    Post-processing utility to convert saved moment fields to other formats
    Supported formats: VTK (.vtu) and Tecplot (.dat)

Namespace
    LBM

SourceFiles
    fieldConvert.cu

\*---------------------------------------------------------------------------*/

#include "fieldConvert.cuh"

using namespace LBM;

// Define here a mapping of std::strings to std::vector<std::string>
// that maps names like StrainRateTensor to S_xx, S_xy, S_xz... etc
// Call it something like the functionObjectRegistry

// const std::unordered_map<std::string, std::vector<std::string>> fieldNamesMap = {
//     {"S", {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"}}};

// const std::vector<std::string> defaultFieldNames{"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"};

[[nodiscard]] const std::vector<std::string> &FieldNames(
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

[[nodiscard]] host::arrayCollection<scalar_t, ctorType::MUST_READ, velocitySet> initialiseArrays(
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

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    // If we have supplied a -fieldName argument, replace programCtrl.caseName() with the fieldName
    const bool doCustomField = programCtrl.input().isArgPresent("-fieldName");
    const std::string fileNamePrefix = doCustomField ? programCtrl.getArgument("-fieldName") : programCtrl.caseName();

    // Now get the std::vector of std::strings corresponding to the prefix
    const std::vector<std::string> &fieldNames = FieldNames(fileNamePrefix, doCustomField);

    // Get the time indices
    const std::vector<label_t> fileNameIndices = fileIO::timeIndices(fileNamePrefix);

    // Get the conversion type
    const std::string conversion = programCtrl.getArgument("-type");

    // Leave unchanged
    const std::unordered_map<std::string, WriterFunction>::const_iterator it = writers.find(conversion);

    // Check if the writer is valid
    if (it != writers.end())
    {
        const WriterFunction writer = it->second;

        for (label_t timeStep = fileIO::getStartIndex(fileNamePrefix, programCtrl); timeStep < fileNameIndices.size(); timeStep++)
        {
            // Get the file name at the present time step
            const std::string filename = fileNamePrefix + "_" + std::to_string(fileNameIndices[timeStep]);

            const host::arrayCollection<scalar_t, ctorType::MUST_READ, velocitySet> hostMoments = initialiseArrays(
                fileNamePrefix,
                programCtrl,
                fieldNames,
                timeStep,
                doCustomField);

            writer(
                fileIO::deinterleaveAoSOptimized(hostMoments.arr(), mesh),
                filename,
                mesh,
                hostMoments.varNames());
        }
    }
    else
    {
        // Throw
        throw std::runtime_error(invalidWriter(writers, conversion));
    }

    return 0;
}