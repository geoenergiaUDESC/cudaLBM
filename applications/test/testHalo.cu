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
    Implementation of the moment representation with the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    momentBasedD3Q19.cu

\*---------------------------------------------------------------------------*/

#include "testHalo.cuh"

using namespace LBM;

__host__ [[nodiscard]] inline consteval label_t NStreams() noexcept { return 1; }

__host__ [[nodiscard]] inline consteval label_t n_warps() noexcept { return block::size() / 32; }

template <const label_t cycle>
__host__ [[nodiscard]] const std::string transactionName(const label_t warpID) noexcept
{
    static_assert(cycle < 4, "Cycle must be 0, 1, 2 or 3");

    if constexpr (cycle == 0)
    {
        if (warpID < 10)
        {
            return "West";
        }
        else
        {
            return "East";
        }
    }
    else if constexpr (cycle == 1)
    {
        if (warpID < 4)
        {
            return "East";
        }
        else if (warpID < 14)
        {
            return "South";
        }
        else
        {
            return "North";
        }
    }
    else if constexpr (cycle == 2)
    {
        if (warpID < 8)
        {
            return "North";
        }
        else
        {
            return "Back";
        }
    }
    else if constexpr (cycle == 3)
    {
        if (warpID < 2)
        {
            return "Back";
        }
        else if (warpID < 12)
        {
            return "Front";
        }
        else
        {
            return "Null";
        }
    }
    else
    {
        return "Null";
        // return {"null"};
    }
}

template <const label_t cycle>
__host__ [[nodiscard]] inline constexpr label_t popID(const label_t warp) noexcept
{
    static_assert(cycle < 4, "Cycle must be 0, 1, 2 or 3");
    if constexpr (cycle == 0)
    {
        return (warp / 2) - (static_cast<label_t>(warp > 9) * 5);
    }
    else if constexpr (cycle == 1)
    {
        if (warp < 4)
        {
            return (warp / 2) + (3);
        }
        else if (warp < 14)
        {
            return (warp / 2) - 2;
        }
        else
        {
            return (warp / 2) - 7;
        }
        // return (warp / 2) + ((warp < 4) * 2);
    }
    else if constexpr (cycle == 2)
    {
        if (warp < 8)
        {
            return (warp / 2) + 1;
        }
        else
        {
            return (warp / 2) - 4;
        }
    }
    else if constexpr (cycle == 3)
    {
        if (warp < 2)
        {
            return (warp / 2) + 4;
        }
        else if (warp < 12)
        {
            return (warp / 2) - 1;
        }
        else
        {
            return static_cast<label_t>(-1);
        }
    }

    return 0;
}

__host__ [[nodiscard]] inline constexpr label_t idxPopBlock(const label_t tx, const label_t ty, const label_t tz, const label_t pop) noexcept
{
    return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (pop)));
}

template <const label_t cycleID>
__host__ [[nodiscard]] inline constexpr std::array<label_t, block::size()> s_buffer_initialise() noexcept
{
    std::array<label_t, block::size()> s_buffer;

    for (std::size_t i = 0; i < s_buffer.size(); i++)
    {
        s_buffer[i] = 0;
    }

    // First cycle
    for (std::size_t z = 0; z < block::nz(); z++)
    {
        for (std::size_t y = 0; y < block::ny(); y++)
        {
            for (std::size_t x = 0; x < block::nx(); x++)
            {
                const label_t warpID = device::idxBlock(x, y, z) / 32;
                s_buffer[device::idxBlock(x, y, z)] = popID<cycleID>(warpID) + 1;
            }
        }
    }

    return s_buffer;
}

// int main(const int argc, const char *const argv[])
int main(void)
{

    host::constexpr_for<0, 4>(
        [&](const auto cycleID)
        {
            constexpr const std::array<label_t, block::size()> s_buffer = s_buffer_initialise<cycleID>();

            if constexpr (cycleID > 0)
            {
                std::cout << std::endl;
            }
            std::cout << "CYCLE " << cycleID << std::endl;
            std::cout << std::endl;

            for (std::size_t i = 0; i < s_buffer.size(); i++)
            {
                if (s_buffer[i] > 0)
                {
                    const label_t warpID = i / 32;
                    const label_t remainder = warpID % 2;

                    // Calculate the thread ID within the face
                    const label_t idx_in_face = i - (warpID * 32) + (remainder * 32);

                    // Calculate the first index
                    const label_t J = idx_in_face / (block::nx());
                    const label_t I = idx_in_face - (J * block::nx());

                    std::cout << "s_buffer[" << i << "] = " << transactionName<cycleID>(warpID) << "[" << s_buffer[i] - 1 << "], idxFace = " << idx_in_face << ", i = " << I << ", j = " << J << std::endl;
                }
            }
        });

    return 0;
}