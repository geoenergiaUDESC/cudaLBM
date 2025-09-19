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

namespace LBM
{
    using VelocitySet = D3Q19;

    using WriterFunction = void (*)(
        const std::vector<std::vector<scalar_t>> &,
        const std::string &,
        const host::latticeMesh &,
        const std::vector<std::string> &);

    /**
     * @brief Veriefies if the command line has the argument -type
     * @return A string representing the convertion type passed at the command line
     * @param argc First argument passed to main
     * @param argv Second argument passed to main
     **/
    __host__ [[nodiscard]] const std::string getConversionType(const programControl &programCtrl)
    {
        if (programCtrl.input().isArgPresent("-type"))
        {
            for (label_t arg = 0; arg < programCtrl.commandLine().size(); arg++)
            {
                if (programCtrl.commandLine()[arg] == "-type")
                {
                    if (arg + 1 == programCtrl.commandLine().size())
                    {
                        throw std::runtime_error("Conversion type not specified: the correct syntax is -type T");
                        return 0;
                    }
                    else
                    {
                        return programCtrl.commandLine()[arg + 1];
                    }
                }
            }
        }

        throw std::runtime_error("Mandatory parameter -type not specified: the correct syntax is -type T");
    }

    /**
     * @brief Unordered map of the writer types to the appropriate functions
     **/
    const std::unordered_map<std::string, WriterFunction> writers = {
        {"vtu", postProcess::writeVTU},
        {"vts", postProcess::writeVTS},
        {"tecplot", postProcess::writeTecplot}};

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
}

#endif