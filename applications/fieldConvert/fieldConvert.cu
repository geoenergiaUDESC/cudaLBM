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

__host__ [[nodiscard]] inline constexpr std::size_t findCharPosition(const std::string &str, const char (&c)[2])
{
    return str.find(c[0]);
}

// namespace time
// {
//     typedef enum Enum : int
//     {
//         instantaneous = 0,
//         timeAverage = 1
//     } type;
// }
// template <const time::::type T>
// using timeType = const std::integral_constant<time::::type, T>;

namespace direction
{
    typedef enum Enum : label_t
    {
        x = 0,
        y = 1,
        z = 2,
        UNDEFINED = 3
    } cardinal;
}

__host__ [[nodiscard]] direction::cardinal cutPlaneDirection(const programControl &programCtrl) noexcept
{
    const std::string cutPlanePrefix = programCtrl.getArgument("-cutPlane");

    // Need to check that j = 1 because the first character before the = symbol should be x, y or z and nothing else
    if (!(findCharPosition(cutPlanePrefix, "=") == 1))
    {
        return direction::UNDEFINED;
    }

    if (cutPlanePrefix[0] == "x"[0])
    {
        return direction::x;
    }

    if (cutPlanePrefix[0] == "y"[0])
    {
        return direction::y;
    }

    if (cutPlanePrefix[0] == "z"[0])
    {
        return direction::z;
    }

    return direction::UNDEFINED;
}

__host__ [[nodiscard]] inline host::latticeMesh meshSlice(const host::latticeMesh &mesh, const direction::cardinal dir) noexcept
{
    if (dir == direction::x)
    {
        return host::latticeMesh(1, mesh.ny(), mesh.nz());
    }

    if (dir == direction::y)
    {
        return host::latticeMesh(mesh.nx(), 1, mesh.nz());
    }

    if (dir == direction::z)
    {
        return host::latticeMesh(mesh.nx(), mesh.ny(), 1);
    }

    return host::latticeMesh(1, 1, 1);
}

__host__ [[nodiscard]] std::vector<std::vector<scalar_t>> initialiseSlice(
    const host::latticeMesh &mesh,
    const label_t nFields,
    const direction::cardinal dir)
{
    if (dir == direction::x)
    {
        return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(mesh.ny() * mesh.nz(), 0));
    }

    if (dir == direction::y)
    {
        return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(mesh.nx() * mesh.nz(), 0));
    }

    if (dir == direction::z)
    {
        return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(mesh.nx() * mesh.ny(), 0));
    }

    return std::vector<std::vector<scalar_t>>(nFields, std::vector<scalar_t>(1, 0));
}

__host__ [[nodiscard]] scalar_t indexCoordinate(const host::latticeMesh &mesh, const scalar_t pointCoordinate, const direction::cardinal dir)
{
    switch (dir)
    {
    case direction::x:
    {
        return static_cast<scalar_t>(mesh.nx() - 1) * (pointCoordinate * mesh.L().x);
    }
    case direction::y:
    {
        return static_cast<scalar_t>(mesh.ny() - 1) * (pointCoordinate * mesh.L().y);
    }
    case direction::z:
    {
        return static_cast<scalar_t>(mesh.nz() - 1) * (pointCoordinate * mesh.L().z);
    }
    default:
    {
        return static_cast<scalar_t>(0);
    }
    }
}

__host__ [[nodiscard]] const std::vector<std::vector<scalar_t>> extractCutPlane(
    const std::vector<std::vector<scalar_t>> &fields,
    const host::latticeMesh &mesh,
    const direction::cardinal dir,
    const scalar_t pointCoordinate)
{
    // Initialise an appropriately sized container based on the mesh dimensions
    std::vector<std::vector<scalar_t>> cutPlane = initialiseSlice(mesh, fields.size(), dir);

    // Get the "index" coordinate
    const scalar_t i = indexCoordinate(mesh, pointCoordinate, dir);
    // const scalar_t i = static_cast<scalar_t>(mesh.nz() - 1) * (pointCoordinate * mesh.L().z());
    const label_t im1 = static_cast<label_t>(std::floor(i));
    const label_t ip1 = static_cast<label_t>(std::ceil(i));

    // Case that it is a x oriented plane
    for (std::size_t field = 0; field < fields.size(); field++)
    {
        for (std::size_t z = 0; z < mesh.nz(); z++)
        {
            for (std::size_t y = 0; y < mesh.ny(); y++)
            {
                const scalar_t fm1 = fields[field][host::idxScalarGlobal(im1, y, z, mesh.nx(), mesh.ny())];
                const scalar_t fp1 = fields[field][host::idxScalarGlobal(ip1, y, z, mesh.nx(), mesh.ny())];
                cutPlane[field][y + (z * mesh.ny())] = (fm1 * (static_cast<scalar_t>(ip1) - i)) + (fp1 * (i - static_cast<scalar_t>(im1)));
            }
        }
    }

    // Case that it is a y oriented plane
    for (std::size_t field = 0; field < fields.size(); field++)
    {
        for (std::size_t z = 0; z < mesh.nz(); z++)
        {
            for (std::size_t x = 0; x < mesh.nx(); x++)
            {
                const scalar_t fm1 = fields[field][host::idxScalarGlobal(x, im1, z, mesh.nx(), mesh.ny())];
                const scalar_t fp1 = fields[field][host::idxScalarGlobal(x, ip1, z, mesh.nx(), mesh.ny())];
                cutPlane[field][x + (z * mesh.nx())] = (fm1 * (static_cast<scalar_t>(ip1) - i)) + (fp1 * (i - static_cast<scalar_t>(im1)));
            }
        }
    }

    // Case that it is a z oriented plane
    for (std::size_t field = 0; field < fields.size(); field++)
    {
        for (std::size_t y = 0; y < mesh.ny(); y++)
        {
            for (std::size_t x = 0; x < mesh.nx(); x++)
            {
                const scalar_t fm1 = fields[field][host::idxScalarGlobal(x, y, im1, mesh.nx(), mesh.ny())];
                const scalar_t fp1 = fields[field][host::idxScalarGlobal(x, y, ip1, mesh.nx(), mesh.ny())];
                cutPlane[field][x + (y * mesh.nx())] = (fm1 * (static_cast<scalar_t>(ip1) - i)) + (fp1 * (i - static_cast<scalar_t>(im1)));
            }
        }
    }

    return cutPlane;
}

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    // If we have supplied a -fieldName argument, replace programCtrl.caseName() with the fieldName
    const bool doCustomField = programCtrl.input().isArgPresent("-fieldName");
    const std::string fileNamePrefix = doCustomField ? programCtrl.getArgument("-fieldName") : programCtrl.caseName();

    // If we have supplied the -cutPlane argument, set the flag to true
    const bool doCutPlane = programCtrl.input().isArgPresent("-cutPlane");
    const std::string cutPlanePrefix = doCutPlane ? programCtrl.getArgument("-cutPlane") : "";

    // Check that size() - 1 isn't = 2
    const std::string value = cutPlanePrefix.substr(2, cutPlanePrefix.size() - 1);
    std::cout << "Value string: " << value << std::endl;
    const scalar_t value_v = static_cast<scalar_t>(std::stold(value));
    std::cout << "Value: " << value_v << std::endl;

    // if (doCutPlane)
    // {
    //     std::cout << cutPlaneDirection(programCtrl) << std::endl;
    // }

    // Now get the std::vector of std::strings corresponding to the prefix
    const std::vector<std::string> &fieldNames = getFieldNames(fileNamePrefix, doCustomField);

    // Get the time indices
    const std::vector<label_t> fileNameIndices = fileIO::timeIndices(fileNamePrefix);

    // Get the conversion type
    const std::string conversion = programCtrl.getArgument("-fileType");

    // Get the writer function
    const std::unordered_map<std::string, WriterFunction>::const_iterator it = writers.find(conversion);

    // Check if the writer is valid
    if (it != writers.end())
    {
        const WriterFunction writer = it->second;

        for (label_t timeStep = fileIO::getStartIndex(fileNamePrefix, programCtrl); timeStep < fileNameIndices.size(); timeStep++)
        {
            const host::arrayCollection<scalar_t, ctorType::MUST_READ, velocitySet> hostMoments = initialiseArrays(
                fileNamePrefix,
                programCtrl,
                fieldNames,
                timeStep,
                doCustomField);

            if (doCutPlane)
            {
                // Get the file name at the present time step
                const std::string fileName = fileNamePrefix + "CutPlane_" + cutPlanePrefix + "_" + std::to_string(fileNameIndices[timeStep]);

                const direction::cardinal dir = cutPlaneDirection(programCtrl);

                const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoS(hostMoments.arr(), mesh);

                const host::latticeMesh slice = meshSlice(mesh, dir);

                const std::vector<std::vector<scalar_t>> cutPlane = extractCutPlane(fields, mesh, dir, value_v);

                writer(
                    {cutPlane},
                    fileName,
                    slice,
                    fieldNames);
            }
            else
            {
                // Get the file name at the present time step
                const std::string fileName = fileNamePrefix + "_" + std::to_string(fileNameIndices[timeStep]);

                writer(
                    fileIO::deinterleaveAoS(hostMoments.arr(), mesh),
                    fileName,
                    mesh,
                    hostMoments.varNames());
            }
        }
    }
    else
    {
        // Throw
        throw std::runtime_error(invalidWriter(writers, conversion));
    }

    return 0;
}