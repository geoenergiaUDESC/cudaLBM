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
    Top-level header file for the halo class

Namespace
    LBM::device

SourceFiles
    blockHalo.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BLOCKHALO_CUH
#define __MBLBM_BLOCKHALO_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../globalFunctions.cuh"
#include "../velocitySet/velocitySet.cuh"
#include "../latticeMesh/latticeMesh.cuh"

namespace LBM
{
    namespace device
    {
        namespace haloFaces
        {
            /**
             * @brief Consteval functions used to distinguish between halo face normal directions
             * @return An unsigned integer corresponding to the correct direction
             **/
            [[nodiscard]] static inline consteval label_t x() noexcept { return 0; }
            [[nodiscard]] static inline consteval label_t y() noexcept { return 1; }
            [[nodiscard]] static inline consteval label_t z() noexcept { return 2; }
        }
    }
}

#include "haloFace.cuh"
#include "halo.cuh"

#endif