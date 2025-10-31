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
    Fallback handler for uncovered boundary cases

SourceFiles
    fallback.cuh

Notes
    This file is invoked inside the 'default' branch of the boundary-condition
    switch statement and is additionally protected by the '!already_handled' flag,
    ensuring that fallback routines are executed only when no prior condition
    has been applied to the current boundary node

\*---------------------------------------------------------------------------*/

// Static faces
case normalVector::WEST():
{
    printOnce(normalVector::WEST(), "WEST");
    
    const scalar_t mxy_I = WEST_mxy_I(pop, inv_rho_I);
    const scalar_t mxz_I = WEST_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST():
{
    printOnce(normalVector::EAST(), "EAST");
    
    const scalar_t mxy_I = EAST_mxy_I(pop, inv_rho_I);
    const scalar_t mxz_I = EAST_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH():
{
    printOnce(normalVector::SOUTH(), "SOUTH");
    
    const scalar_t mxy_I = SOUTH_mxy_I(pop, inv_rho_I);
    const scalar_t myz_I = SOUTH_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH():
{
    printOnce(normalVector::NORTH(), "NORTH");
    
    const scalar_t mxy_I = NORTH_mxy_I(pop, inv_rho_I);
    const scalar_t myz_I = NORTH_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::BACK():
{
    printOnce(normalVector::BACK(), "BACK");

    const scalar_t mxz_I = BACK_mxz_I(pop, inv_rho_I);
    const scalar_t myz_I = BACK_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
    const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = static_cast<scalar_t>(0); // ux
    moments[label_constant<2>()] = static_cast<scalar_t>(0); // uy
    moments[label_constant<3>()] = static_cast<scalar_t>(0); // uz
    moments[label_constant<4>()] = static_cast<scalar_t>(0); // mxx
    moments[label_constant<5>()] = static_cast<scalar_t>(0); // mxy
    moments[label_constant<6>()] = mxz;                      // mxz
    moments[label_constant<7>()] = static_cast<scalar_t>(0); // myy
    moments[label_constant<8>()] = myz;                      // myz
    moments[label_constant<9>()] = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::FRONT():
{
    printOnce(normalVector::FRONT(), "FRONT");

    const scalar_t mxz_I = FRONT_mxz_I(pop, inv_rho_I);
    const scalar_t myz_I = FRONT_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
    const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = static_cast<scalar_t>(0); // ux
    moments[label_constant<2>()] = static_cast<scalar_t>(0); // uy
    moments[label_constant<3>()] = static_cast<scalar_t>(0); // uz
    moments[label_constant<4>()] = static_cast<scalar_t>(0); // mxx
    moments[label_constant<5>()] = static_cast<scalar_t>(0); // mxy
    moments[label_constant<6>()] = mxz;                      // mxz
    moments[label_constant<7>()] = static_cast<scalar_t>(0); // myy
    moments[label_constant<8>()] = myz;                      // myz
    moments[label_constant<9>()] = static_cast<scalar_t>(0); // mzz

    return;
}

// Static edges
case normalVector::WEST_SOUTH():
{
    printOnce(normalVector::WEST_SOUTH(), "WEST_SOUTH");
    
    const scalar_t mxy_I = SOUTH_WEST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::WEST_NORTH():
{
    printOnce(normalVector::WEST_NORTH(), "WEST_NORTH");
    
    const scalar_t mxy_I = NORTH_WEST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}
case normalVector::WEST_BACK():
{
    printOnce(normalVector::WEST_BACK(), "WEST_BACK");
    
    const scalar_t mxz_I = WEST_BACK_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::WEST_FRONT():
{
    printOnce(normalVector::WEST_FRONT(), "WEST_FRONT");
    
    const scalar_t mxz_I = WEST_FRONT_mxz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_SOUTH():
{
    printOnce(normalVector::EAST_SOUTH(), "EAST_SOUTH");
    
    const scalar_t mxy_I = SOUTH_EAST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_NORTH():
{
    printOnce(normalVector::EAST_NORTH(), "EAST_NORTH");
    
    const scalar_t mxy_I = NORTH_EAST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}
case normalVector::EAST_BACK():
{
    printOnce(normalVector::EAST_BACK(), "EAST_BACK");
    
    const scalar_t mxz_I = EAST_BACK_mxz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_FRONT():
{
    printOnce(normalVector::EAST_FRONT(), "EAST_FRONT");
    
    const scalar_t mxz_I = EAST_FRONT_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_BACK():
{
    printOnce(normalVector::SOUTH_BACK(), "SOUTH_BACK");
    
    const scalar_t myz_I = SOUTH_BACK_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_FRONT():
{
    printOnce(normalVector::SOUTH_FRONT(), "SOUTH_FRONT");
    
    const scalar_t myz_I = SOUTH_FRONT_myz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH_BACK():
{
    printOnce(normalVector::NORTH_BACK(), "NORTH_BACK");
    
    const scalar_t myz_I = NORTH_BACK_myz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}
case normalVector::NORTH_FRONT():
{
    printOnce(normalVector::NORTH_FRONT(), "NORTH_FRONT");
    
    const scalar_t myz_I = NORTH_FRONT_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}

// Static corners
case normalVector::WEST_SOUTH_BACK():
{
    printOnce(normalVector::WEST_SOUTH_BACK(), "WEST_SOUTH_BACK");

    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::WEST_SOUTH_FRONT():
{
    printOnce(normalVector::WEST_SOUTH_FRONT(), "WEST_SOUTH_FRONT");

    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::WEST_NORTH_BACK():
{
    printOnce(normalVector::WEST_NORTH_BACK(), "WEST_NORTH_BACK");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::WEST_NORTH_FRONT():
{
    printOnce(normalVector::WEST_NORTH_FRONT(), "WEST_NORTH_FRONT");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_SOUTH_BACK():
{
    printOnce(normalVector::EAST_SOUTH_BACK(), "EAST_SOUTH_BACK");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_SOUTH_FRONT():
{
    printOnce(normalVector::EAST_SOUTH_FRONT(), "EAST_SOUTH_FRONT");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_NORTH_BACK():
{
    printOnce(normalVector::EAST_NORTH_BACK(), "EAST_NORTH_BACK");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_NORTH_FRONT():
{
    printOnce(normalVector::EAST_NORTH_FRONT(), "EAST_NORTH_FRONT");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}

