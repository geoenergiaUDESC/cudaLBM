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
Authors: Nathan Duggins, Vinicius Czarnobay, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Applies IRBC Neumann-type boundary conditions

SourceFiles
    IRBCNeumann.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

// Faces
case normalVector::FRONT(): 
{
    const scalar_t mxz_I = FRONT_mxz_I(pop, inv_rho_I);
    const scalar_t myz_I = FRONT_myz_I(pop, inv_rho_I);
    const scalar_t mxx_I = ((pop[label_constant<1>()]) + (pop[label_constant<2>()]) + (pop[label_constant<7>()]) + (pop[label_constant<8>()]) + (pop[label_constant<9>()]) + (pop[label_constant<13>()]) + (pop[label_constant<14>()]) + (pop[label_constant<16>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();
    const scalar_t mxy_I = ((pop[label_constant<7>()]) + (pop[label_constant<8>()]) - (pop[label_constant<13>()]) - (pop[label_constant<14>()])) * inv_rho_I;
    const scalar_t myy_I = ((pop[label_constant<3>()]) + (pop[label_constant<4>()]) + (pop[label_constant<7>()]) + (pop[label_constant<8>()]) + (pop[label_constant<11>()]) + (pop[label_constant<13>()]) + (pop[label_constant<14>()]) + (pop[label_constant<18>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("FRONT", offset);

    // Classic Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = -((-static_cast<scalar_t>(4) * mxx_I * rho_I + static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mxx
    moments[label_constant<5>()] = (mxy_I * rho_I) / rho; // mxy
    moments[label_constant<6>()] = -((-static_cast<scalar_t>(6) * mxz_I * rho_I + moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxz
    moments[label_constant<7>()] = -((static_cast<scalar_t>(4) * mxx_I * rho_I - static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(6) * rho)); // myy
    moments[label_constant<8>()] = -((-static_cast<scalar_t>(6) * myz_I * rho_I + moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho));// myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}

// Corners
case normalVector::WEST_FRONT(): 
{
    const scalar_t mxy_I = ((pop[label_constant<8>()]) - (pop[label_constant<14>()])) * inv_rho_I;
    const scalar_t myz_I = ((pop[label_constant<11>()]) - (pop[label_constant<18>()])) * inv_rho_I;

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("WEST_FRONT", offset);

    // Classic Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I + moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::EAST_FRONT(): 
{
    const scalar_t mxy_I = ((pop[label_constant<7>()]) - (pop[label_constant<13>()])) * inv_rho_I;
    const scalar_t myz_I = ((pop[label_constant<11>()]) - (pop[label_constant<18>()])) * inv_rho_I;

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("EAST_FRONT", offset);

    // Classic Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I - moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz  
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::SOUTH_FRONT(): 
{
    const scalar_t mxy_I = ((pop[label_constant<8>()]) - (pop[label_constant<13>()])) * inv_rho_I;
    const scalar_t mxz_I = ((pop[label_constant<9>()]) - (pop[label_constant<16>()])) * inv_rho_I;

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("SOUTH_FRONT", offset);

    // Classic Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I + moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy  
    moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::NORTH_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("NORTH_FRONT", offset);

    const scalar_t mxy_I = ((pop[label_constant<7>()]) - (pop[label_constant<14>()])) * inv_rho_I;
    const scalar_t mxz_I = ((pop[label_constant<9>()]) - (pop[label_constant<16>()])) * inv_rho_I;

    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I - moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy
    moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;
    return;
}

// Edges
case normalVector::WEST_SOUTH_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("WEST_SOUTH_FRONT", offset);

    // Classic Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::WEST_NORTH_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("WEST_NORTH_FRONT", offset);
    
    // Classic Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann 
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::EAST_SOUTH_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("EAST_SOUTH_FRONT", offset);

    // Classic Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::EAST_NORTH_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    printThreadMapping("EAST_NORTH_FRONT", offset);

    // Classic Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}