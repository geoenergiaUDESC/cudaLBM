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
    Post-processing utility to calculate derived fields from saved moment fields
    Supported calculations: velocity magnitude, velocity divergence, vorticity,
    vorticity magnitude, integrated vorticity

Namespace
    LBM

SourceFiles
    testExecutable.cu

\*---------------------------------------------------------------------------*/

#include "testExecutable.cuh"

using namespace LBM;

using VelocitySet = D3Q19;

template <typename T, const label_t N>
__host__ __device__ [[nodiscard]] inline consteval label_t number_indices_equal(const thread::array<T, N> &arr, const T val) noexcept
{
    label_t j = 0;

    for (label_t i = 0; i < N; i++)
    {
        if (arr[i] == val)
        {
            j++;
        }
    }

    return j;
}

template <const label_t NReturn, typename T, const label_t N>
__host__ __device__ [[nodiscard]] inline consteval thread::array<label_t, NReturn> indices_equal(const thread::array<T, N> &arr, const T val) noexcept
{
    thread::array<label_t, NReturn> indices;

    label_t j = 0;

    for (label_t i = 0; i < N; i++)
    {
        if (arr[i] == val)
        {
            indices[j] = i;
            j++;
        }
    }

    return indices;
}

int main()
{
    static constexpr const label_t N = number_indices_equal(VelocitySet::cx<int>(), -1);

    static_assert(N == VelocitySet::QF());

    static constexpr const thread::array<label_t, N> indices = indices_equal<N>(VelocitySet::cx<int>(), -1);

    // thread::array<label_t, VelocitySet::QF()> indices;

    // constexpr const thread::array<int, VelocitySet::Q()> cx = VelocitySet::cx<int>();

    // label_t j = 0;
    // for (label_t i = 0; i < VelocitySet::Q(); i++)
    // {
    //     if (cx[i] == -1)
    //     {
    //         indices[j] = i;
    //         j++;
    //     }
    // }

    for (label_t i = 0; i < VelocitySet::QF(); i++)
    {
        std::cout << indices[i] << std::endl;
    }

    // std::cout << "This executable is used for testing purposes only" << std::endl;

    return 0;
}