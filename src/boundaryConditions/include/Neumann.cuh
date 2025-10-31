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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    along with this program. If not, see <https://www.gnu.org/licenses/>.

Description
    Temporary Neumann boundary implementation used as a placeholder for development and testing

SourceFiles
    Neumann.cuh

Notes
    Copies second-order moments from the interior node, which should instead 
    be computed according to the IRBC formulation (see IRBCNeumann.cuh).

    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

case normalVector::WEST():            
case normalVector::EAST():            
case normalVector::SOUTH():           
case normalVector::NORTH():            
case normalVector::FRONT():            
case normalVector::WEST_SOUTH():       
case normalVector::WEST_NORTH():       
case normalVector::WEST_FRONT():       
case normalVector::EAST_SOUTH():      
case normalVector::EAST_NORTH():      
case normalVector::EAST_FRONT():       
case normalVector::SOUTH_FRONT():     
case normalVector::NORTH_FRONT():      
case normalVector::WEST_SOUTH_FRONT(): 
case normalVector::WEST_NORTH_FRONT(): 
case normalVector::EAST_SOUTH_FRONT(): 
case normalVector::EAST_NORTH_FRONT(): 
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    device::constexpr_for<0, NUMBER_MOMENTS()>(
        [&](const auto moment)
        {
            const label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
            moments[moment] = shared_buffer[ID];
        });

    already_handled = true;

    return;
}
