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
    Temporary boundary implementation used as a placeholder for development and testing

SourceFiles
    tanDirichlet.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

// Faces
case normalVector::WEST():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = ux;      // ux
    moments(label_constant<2>()) = uy;      // uy
    moments(label_constant<3>()) = uz;      // uz
    moments(label_constant<4>()) = ux * ux; // mxx
    moments(label_constant<5>()) = ux * uy; // mxy
    moments(label_constant<6>()) = ux * uz; // mxz
    moments(label_constant<7>()) = uy * uy; // myy
    moments(label_constant<8>()) = uy * uz; // myz
    moments(label_constant<9>()) = uz * uz; // mzz

    return;
}
case normalVector::EAST():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = ux;      // ux
    moments(label_constant<2>()) = uy;      // uy
    moments(label_constant<3>()) = uz;      // uz
    moments(label_constant<4>()) = ux * ux; // mxx
    moments(label_constant<5>()) = ux * uy; // mxy
    moments(label_constant<6>()) = ux * uz; // mxz
    moments(label_constant<7>()) = uy * uy; // myy
    moments(label_constant<8>()) = uy * uz; // myz
    moments(label_constant<9>()) = uz * uz; // mzz

    return;
}
case normalVector::SOUTH():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = ux;      // ux
    moments(label_constant<2>()) = uy;      // uy
    moments(label_constant<3>()) = uz;      // uz
    moments(label_constant<4>()) = ux * ux; // mxx
    moments(label_constant<5>()) = ux * uy; // mxy
    moments(label_constant<6>()) = ux * uz; // mxz
    moments(label_constant<7>()) = uy * uy; // myy
    moments(label_constant<8>()) = uy * uz; // myz
    moments(label_constant<9>()) = uz * uz; // mzz

    return;
}
case normalVector::NORTH():
{
    const scalar_t rho = rho0<scalar_t>();

    const scalar_t ux = static_cast<scalar_t>(0);
    const scalar_t uy = static_cast<scalar_t>(0);
    const scalar_t uz = static_cast<scalar_t>(0.01) * device::u_inf;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = ux;      // ux
    moments(label_constant<2>()) = uy;      // uy
    moments(label_constant<3>()) = uz;      // uz
    moments(label_constant<4>()) = ux * ux; // mxx
    moments(label_constant<5>()) = ux * uy; // mxy
    moments(label_constant<6>()) = ux * uz; // mxz
    moments(label_constant<7>()) = uy * uy; // myy
    moments(label_constant<8>()) = uy * uz; // myz
    moments(label_constant<9>()) = uz * uz; // mzz

    return;
}

// Edges
// case normalVector::WEST_SOUTH():
// {
//     printOnce(normalVector::WEST_SOUTH(), "WEST_SOUTH");

//     const scalar_t mxy_I = WEST_SOUTH_mxy_I(pop, inv_rho_I);

//     const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
//     const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

//     moments(label_constant<0>()) = rho;
//     moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
//     moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
//     moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
//     moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
//     moments(label_constant<5>()) = mxy;                      // mxy
//     moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
//     moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
//     moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
//     moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

//     return;
// }
// case normalVector::WEST_NORTH():
// {
//     printOnce(normalVector::WEST_NORTH(), "WEST_NORTH");

//     const scalar_t mxy_I = WEST_NORTH_mxy_I(pop, inv_rho_I);

//     const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
//     const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

//     moments(label_constant<0>()) = rho;
//     moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
//     moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
//     moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
//     moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
//     moments(label_constant<5>()) = mxy;                      // mxy
//     moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
//     moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
//     moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
//     moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
//     return;
// }

// // Corners
// case normalVector::EAST_SOUTH():
// {
//     printOnce(normalVector::EAST_SOUTH(), "EAST_SOUTH");

//     const scalar_t mxy_I = EAST_SOUTH_mxy_I(pop, inv_rho_I);

//     const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
//     const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

//     moments(label_constant<0>()) = rho;
//     moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
//     moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
//     moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
//     moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
//     moments(label_constant<5>()) = mxy;                      // mxy
//     moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
//     moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
//     moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
//     moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

//     return;
// }
// case normalVector::EAST_NORTH():
// {
//     printOnce(normalVector::EAST_NORTH(), "EAST_NORTH");

//     const scalar_t mxy_I = EAST_NORTH_mxy_I(pop, inv_rho_I);

//     const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
//     const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

//     moments(label_constant<0>()) = rho;
//     moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
//     moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
//     moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
//     moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
//     moments(label_constant<5>()) = mxy;                      // mxy
//     moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
//     moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
//     moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
//     moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
//     return;
// }