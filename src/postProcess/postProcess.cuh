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
    Top-level header file for the post processing routines

Namespace
    LBM::postProcess

SourceFiles
    postProcess.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_POSTPROCESS_CUH
#define __MBLBM_POSTPROCESS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Calculates physical coordinates of lattice points
         * @tparam T Coordinate data type (typically scalar_t or double)
         * @param[in] mesh Lattice mesh providing dimensions and physical size
         * @return Vector of coordinates in interleaved format [x0, y0, z0, x1, y1, z1, ...]
         *
         * This function converts lattice indices to physical coordinates using
         * the domain dimensions stored in the mesh. Coordinates are normalized
         * to the physical domain size and distributed evenly across the lattice.
         */
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> meshCoordinates(const host::latticeMesh &mesh)
        {
            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            const pointVector &L = mesh.L();

            const label_t numNodes = nx * ny * nz;
            std::vector<T> coords(numNodes * 3);

            for (label_t k = 0; k < nz; ++k)
            {
                for (label_t j = 0; j < ny; ++j)
                {
                    for (label_t i = 0; i < nx; ++i)
                    {
                        const label_t idx = k * ny * nx + j * nx + i;
                        // Do the conversion in double, then cast to the desired type
                        coords[3 * idx + 0] = static_cast<T>((static_cast<double>(L.x) * static_cast<double>(i)) / static_cast<double>(nx - 1));
                        coords[3 * idx + 1] = static_cast<T>((static_cast<double>(L.y) * static_cast<double>(j)) / static_cast<double>(ny - 1));
                        coords[3 * idx + 2] = static_cast<T>((static_cast<double>(L.z) * static_cast<double>(k)) / static_cast<double>(nz - 1));
                    }
                }
            }

            return coords;
        }

        /**
         * @brief Calculates element connectivity for visualization
         * @tparam one_based If true, uses 1-based indexing; if false, uses 0-based
         * @param[in] mesh Lattice mesh providing dimensions
         * @return Vector of element connectivity in VTK hexahedron order
         *
         * This function generates connectivity information for hexahedral elements
         * that make up the lattice mesh. The connectivity follows the standard
         * VTK ordering for hexahedrons (voxels).
         *
         * @note The one_based template parameter determines whether node indices
         *       start at 1 (for some file formats like Tecplot) or 0 (for VTK)
         */
        template <const bool one_based>
        __host__ [[nodiscard]] const std::vector<label_t> meshConnectivity(const host::latticeMesh &mesh)
        {
            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            const label_t numElements = (nx - 1) * (ny - 1) * (nz - 1);

            std::vector<label_t> connectivity(numElements * 8);
            label_t cell_idx = 0;
            constexpr const label_t offset = one_based ? 1 : 0;

            for (label_t k = 0; k < nz - 1; ++k)
            {
                for (label_t j = 0; j < ny - 1; ++j)
                {
                    for (label_t i = 0; i < nx - 1; ++i)
                    {
                        const label_t base = k * nx * ny + j * nx + i;
                        const label_t stride_y = nx;
                        const label_t stride_z = nx * ny;

                        connectivity[cell_idx * 8 + 0] = base + offset;
                        connectivity[cell_idx * 8 + 1] = base + 1 + offset;
                        connectivity[cell_idx * 8 + 2] = base + stride_y + 1 + offset;
                        connectivity[cell_idx * 8 + 3] = base + stride_y + offset;
                        connectivity[cell_idx * 8 + 4] = base + stride_z + offset;
                        connectivity[cell_idx * 8 + 5] = base + stride_z + 1 + offset;
                        connectivity[cell_idx * 8 + 6] = base + stride_z + stride_y + 1 + offset;
                        connectivity[cell_idx * 8 + 7] = base + stride_z + stride_y + offset;
                        ++cell_idx;
                    }
                }
            }

            return connectivity;
        }

        /**
         * @brief Calculates offset pointers for unstructured grid data
         * @param[in] mesh Lattice mesh providing dimensions
         * @return Vector of offset pointers for VTK unstructured grid format
         *
         * This function generates offset information for VTK unstructured grid
         * format, where each offset indicates the cumulative number of points
         * up to that element. For hexahedral elements, each element has 8 points,
         * so offsets increase by 8 for each element.
         */
        __host__ [[nodiscard]] const std::vector<label_t> meshOffsets(const host::latticeMesh &mesh)
        {
            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            const label_t numElements = (nx - 1) * (ny - 1) * (nz - 1);

            std::vector<label_t> offsets(numElements);

            for (label_t i = 0; i < numElements; ++i)
            {
                offsets[i] = (i + 1) * 8;
            }

            return offsets;
        }
    }
}

#include "Tecplot.cuh"
#include "VTU.cuh"

#endif