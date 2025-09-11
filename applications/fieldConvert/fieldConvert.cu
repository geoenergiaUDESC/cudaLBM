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

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    const std::vector<label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

    const std::string conversion = getConversionType(programCtrl);

    for (label_t timeStep = fileIO::getStartIndex(programCtrl); timeStep < fileNameIndices.size(); timeStep++)
    {
        const host::arrayCollection<scalar_t, ctorType::MUST_READ, velocitySet> hostMoments(
            programCtrl,
            {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
            timeStep);

        const std::vector<std::vector<scalar_t>> soa = fileIO::deinterleaveAoSOptimized(hostMoments.arr(), mesh);

        auto it = writers.find(conversion);

        if (it != writers.end())
        {
            const std::string extension = (conversion == "vtu") ? ".vtu" : ".dat";

            const std::string filename = programCtrl.caseName() + "_" + std::to_string(fileNameIndices[timeStep]) + extension;

            const WriterFunction writer = it->second;

            writer(soa, filename, mesh, hostMoments.varNames(), "Title");
        }
        else
        {
            throw std::runtime_error("Unsupported conversion format: " + conversion);
        }
    }

    return 0;
}