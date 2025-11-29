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
    fieldCalculate.cu

\*---------------------------------------------------------------------------*/

#include "testExecutable.cuh"

using namespace LBM;

using VelocitySet = D3Q27;

template <const label_t ReturnSize, typename T, const label_t N>
__host__ __device__ [[nodiscard]] inline consteval const thread::array<label_t, ReturnSize> non_zero_indices(const thread::array<T, N> &C)
{
    thread::array<label_t, ReturnSize> arr;

    label_t j = 0;

    for (label_t i = 0; i < N; i++)
    {
        if (!(C[i] == 0))
        {
            arr[j] = i;
            j++;
        }
    }

    return arr;
}

int main()
{
    constexpr const thread::array<int, VelocitySet::Q()> C_ = VelocitySet::cx<int>() * VelocitySet::cy<int>();

    constexpr const label_t N = number_non_zero(C_);

    constexpr const thread::array<int, N> C = extract_non_zero<N>(C_);

    for (label_t i = 0; i < C_.size(); i++)
    {
        std::cout << C_[i] << std::endl;
    }

    std::cout << std::endl;

    for (label_t i = 0; i < C.size(); i++)
    {
        std::cout << C[i] << std::endl;
    }

    // constexpr const thread::array<int, 27> C = dot_product(D3Q27::cx<int>(), D3Q27::cy<int>());

    // constexpr const label_t N = number_non_zero(dot_product(D3Q27::cx<int>(), D3Q27::cy<int>()));

    // std::cout << N << std::endl;

    return 0;
}