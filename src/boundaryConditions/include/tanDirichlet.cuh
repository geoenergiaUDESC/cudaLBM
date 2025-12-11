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
    Lateral boundaries are assigned a z-oriented prescribed velocity
    equal to 1% of the maximum axial velocity

SourceFiles
    tanDirichlet.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

case normalVector::WEST():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments[m_i<0>()] = rho;     // rho
    moments[m_i<1>()] = ux;      // ux
    moments[m_i<2>()] = uy;      // uy
    moments[m_i<3>()] = uz;      // uz
    moments[m_i<4>()] = ux * ux; // mxx
    moments[m_i<5>()] = ux * uy; // mxy
    moments[m_i<6>()] = ux * uz; // mxz
    moments[m_i<7>()] = uy * uy; // myy
    moments[m_i<8>()] = uy * uz; // myz
    moments[m_i<9>()] = uz * uz; // mzz

    return;
}
case normalVector::EAST():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments[m_i<0>()] = rho;     // rho
    moments[m_i<1>()] = ux;      // ux
    moments[m_i<2>()] = uy;      // uy
    moments[m_i<3>()] = uz;      // uz
    moments[m_i<4>()] = ux * ux; // mxx
    moments[m_i<5>()] = ux * uy; // mxy
    moments[m_i<6>()] = ux * uz; // mxz
    moments[m_i<7>()] = uy * uy; // myy
    moments[m_i<8>()] = uy * uz; // myz
    moments[m_i<9>()] = uz * uz; // mzz

    return;
}
case normalVector::SOUTH():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments[m_i<0>()] = rho;     // rho
    moments[m_i<1>()] = ux;      // ux
    moments[m_i<2>()] = uy;      // uy
    moments[m_i<3>()] = uz;      // uz
    moments[m_i<4>()] = ux * ux; // mxx
    moments[m_i<5>()] = ux * uy; // mxy
    moments[m_i<6>()] = ux * uz; // mxz
    moments[m_i<7>()] = uy * uy; // myy
    moments[m_i<8>()] = uy * uz; // myz
    moments[m_i<9>()] = uz * uz; // mzz

    return;
}
case normalVector::NORTH():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments[m_i<0>()] = rho;     // rho
    moments[m_i<1>()] = ux;      // ux
    moments[m_i<2>()] = uy;      // uy
    moments[m_i<3>()] = uz;      // uz
    moments[m_i<4>()] = ux * ux; // mxx
    moments[m_i<5>()] = ux * uy; // mxy
    moments[m_i<6>()] = ux * uz; // mxz
    moments[m_i<7>()] = uy * uy; // myy
    moments[m_i<8>()] = uy * uz; // myz
    moments[m_i<9>()] = uz * uz; // mzz

    return;
}
// case normalVector::WEST_SOUTH():
// {
//     const scalar_t rho = rho0<scalar_t>();

//     const scalar_t ux = static_cast<scalar_t>(0);
//     const scalar_t uy = static_cast<scalar_t>(0);
//     const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

//     moments[m_i<0>()] = rho;     // rho
//     moments[m_i<1>()] = ux;      // ux
//     moments[m_i<2>()] = uy;      // uy
//     moments[m_i<3>()] = uz;      // uz
//     moments[m_i<4>()] = ux * ux; // mxx
//     moments[m_i<5>()] = ux * uy; // mxy
//     moments[m_i<6>()] = ux * uz; // mxz
//     moments[m_i<7>()] = uy * uy; // myy
//     moments[m_i<8>()] = uy * uz; // myz
//     moments[m_i<9>()] = uz * uz; // mzz

//     return;
// }
// case normalVector::WEST_NORTH():
// {
//     const scalar_t rho = rho0<scalar_t>();

//     const scalar_t ux = static_cast<scalar_t>(0);
//     const scalar_t uy = static_cast<scalar_t>(0);
//     const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

//     moments[m_i<0>()] = rho;     // rho
//     moments[m_i<1>()] = ux;      // ux
//     moments[m_i<2>()] = uy;      // uy
//     moments[m_i<3>()] = uz;      // uz
//     moments[m_i<4>()] = ux * ux; // mxx
//     moments[m_i<5>()] = ux * uy; // mxy
//     moments[m_i<6>()] = ux * uz; // mxz
//     moments[m_i<7>()] = uy * uy; // myy
//     moments[m_i<8>()] = uy * uz; // myz
//     moments[m_i<9>()] = uz * uz; // mzz

//     return;
// }
// case normalVector::EAST_SOUTH():
// {
//     const scalar_t rho = rho0<scalar_t>();

//     const scalar_t ux = static_cast<scalar_t>(0);
//     const scalar_t uy = static_cast<scalar_t>(0);
//     const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

//     moments[m_i<0>()] = rho;     // rho
//     moments[m_i<1>()] = ux;      // ux
//     moments[m_i<2>()] = uy;      // uy
//     moments[m_i<3>()] = uz;      // uz
//     moments[m_i<4>()] = ux * ux; // mxx
//     moments[m_i<5>()] = ux * uy; // mxy
//     moments[m_i<6>()] = ux * uz; // mxz
//     moments[m_i<7>()] = uy * uy; // myy
//     moments[m_i<8>()] = uy * uz; // myz
//     moments[m_i<9>()] = uz * uz; // mzz

//     return;
// }
// case normalVector::EAST_NORTH():
// {
//     const scalar_t rho = rho0<scalar_t>();

//     const scalar_t ux = static_cast<scalar_t>(0);
//     const scalar_t uy = static_cast<scalar_t>(0);
//     const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

//     moments[m_i<0>()] = rho;     // rho
//     moments[m_i<1>()] = ux;      // ux
//     moments[m_i<2>()] = uy;      // uy
//     moments[m_i<3>()] = uz;      // uz
//     moments[m_i<4>()] = ux * ux; // mxx
//     moments[m_i<5>()] = ux * uy; // mxy
//     moments[m_i<6>()] = ux * uz; // mxz
//     moments[m_i<7>()] = uy * uy; // myy
//     moments[m_i<8>()] = uy * uz; // myz
//     moments[m_i<9>()] = uz * uz; // mzz

//     return;
// }