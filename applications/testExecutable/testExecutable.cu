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

using VelocitySet = D3Q19;

// static constexpr const label_t device_nx = 128;
// static constexpr const label_t device_ny = 128;
// static constexpr const label_t device_nz = 128;

/**
 * @brief Check if current thread is at western block boundary
 * @param[in] x Global x-coordinate
 * @return True if at western boundary but not domain edge
 **/
__host__ [[nodiscard]] static inline bool West(const label_t tx) noexcept
{
    return (tx == 0);
}

/**
 * @brief Check if current thread is at eastern block boundary
 * @param[in] x Global x-coordinate
 * @return True if at eastern boundary but not domain edge
 **/
__host__ [[nodiscard]] static inline bool East(const label_t tx) noexcept
{
    return (tx == (block::nx() - 1));
}

/**
 * @brief Check if current thread is at southern block boundary
 * @param[in] y Global y-coordinate
 * @return True if at southern boundary but not domain edge
 **/
__host__ [[nodiscard]] static inline bool South(const label_t ty) noexcept
{
    return (ty == 0);
}

/**
 * @brief Check if current thread is at northern block boundary
 * @param[in] y Global y-coordinate
 * @return True if at northern boundary but not domain edge
 **/
__host__ [[nodiscard]] static inline bool North(const label_t ty) noexcept
{
    return (ty == (block::ny() - 1));
}

/**
 * @brief Check if current thread is at back (z-min) block boundary
 * @param[in] z Global z-coordinate
 * @return True if at back boundary but not domain edge
 **/
__host__ [[nodiscard]] static inline bool Back(const label_t tz) noexcept
{
    return (tz == 0);
}

/**
 * @brief Check if current thread is at front (z-max) block boundary
 * @param[in] z Global z-coordinate
 * @return True if at front boundary but not domain edge
 **/
__host__ [[nodiscard]] static inline bool Front(const label_t tz) noexcept
{
    return (tz == (block::nz() - 1));
}

// static constexpr const label_t x_ =99;

static constexpr const label_t warp_size = 32;

__host__ [[nodiscard]] inline label_t idx_block(const label_t tx, const label_t ty, const label_t tz) noexcept
{
    return tx + block::nx() * (ty + block::ny() * tz);
}

__host__ [[nodiscard]] inline label_t warp_idx(const label_t tx, const label_t ty, const label_t tz) noexcept
{
    return idx_block(tx, ty, tz) / warp_size;
}

static constexpr const label_t n_cycles = 4;

int main()
{
    thread::array<int, block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS()>()> s_buffer(0);
    // for (label_t i = 0; i < s_buffer.size(); i++)
    // {
    //     s_buffer[i] = 0;
    // }

    // Put the populations into the shared memory
    for (label_t z = 0; z < block::nz(); z++)
    {
        for (label_t y = 0; y < block::ny(); y++)
        {
            for (label_t x = 0; x < block::nx(); x++)
            {
                // const label_t warpID = warp_idx(x, y, z);
                // std::cout << warpID << std::endl;
                constexpr const label_t z_size = block::nx() * block::ny();
                constexpr const label_t y_size = block::nx() * block::nz();
                constexpr const label_t x_size = block::ny() * block::nz();
                // constexpr const label_t yz_nWarps = x_size / warp_size;

                if (West(x))
                {                                                          // w
                    s_buffer[(z + (y * block::ny())) + (0 * x_size)] = 2;  // pop[q_i<2>()];
                    s_buffer[(z + (y * block::ny())) + (1 * x_size)] = 8;  // pop[q_i<8>()];
                    s_buffer[(z + (y * block::ny())) + (2 * x_size)] = 10; // pop[q_i<10>()];
                    s_buffer[(z + (y * block::ny())) + (3 * x_size)] = 14; // pop[q_i<14>()];
                    s_buffer[(z + (y * block::ny())) + (4 * x_size)] = 16; // pop[q_i<16>()];
                }

                if (East(x))
                {                                                          // e
                    s_buffer[(z + (y * block::ny())) + (5 * x_size)] = 1;  // pop[q_i<1>()];
                    s_buffer[(z + (y * block::ny())) + (6 * x_size)] = 7;  // pop[q_i<7>()];
                    s_buffer[(z + (y * block::ny())) + (7 * x_size)] = 9;  // pop[q_i<9>()];
                    s_buffer[(z + (y * block::ny())) + (8 * x_size)] = 13; // pop[q_i<13>()];
                    s_buffer[(z + (y * block::ny())) + (9 * x_size)] = 5;  // pop[q_i<15>()];
                }

                if (South(y))
                {                                                                          // s
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (0 * y_size)] = 4;  // pop[q_i<4>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (1 * y_size)] = 8;  // pop[q_i<8>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (2 * y_size)] = 12; // pop[q_i<12>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (3 * y_size)] = 13; // pop[q_i<13>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (4 * y_size)] = 18; // pop[q_i<18>()];
                }

                if (North(y))
                {                                                                          // n
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (5 * y_size)] = 3;  // pop[q_i<3>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (6 * y_size)] = 7;  // pop[q_i<7>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (7 * y_size)] = 11; // pop[q_i<11>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (8 * y_size)] = 14; // pop[q_i<14>()];
                    s_buffer[(x + (z * block::nx())) + (10 * x_size) + (9 * y_size)] = 17; // pop[q_i<17>()];
                }

                if (Back(z))
                {                                                                                          // b
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (0 * z_size)] = 6;  // pop[q_i<6>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (1 * z_size)] = 10; // pop[q_i<10>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (2 * z_size)] = 12; // pop[q_i<12>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (3 * z_size)] = 15; // pop[q_i<15>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (4 * z_size)] = 17; // pop[q_i<17>()];
                }

                if (Front(z))
                {
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (5 * z_size)] = 5;  // pop[q_i<5>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (6 * z_size)] = 8;  // pop[q_i<9>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (7 * z_size)] = 11; // pop[q_i<11>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (8 * z_size)] = 16; // pop[q_i<16>()];
                    s_buffer[(x + (y * block::nx())) + (10 * x_size) + (10 * y_size) + (9 * z_size)] = 18; // pop[q_i<18>()];
                }
            }
        }
    }

    // for (label_t cycle = 0; cycle < n_cycles; cycle++)
    for (label_t cycle = 0; cycle < 1; cycle++)
    {
        std::cout << "Cycle " << cycle << std::endl;
        for (label_t z = 0; z < block::nz(); z++)
        {
            for (label_t y = 0; y < block::ny(); y++)
            {
                for (label_t x = 0; x < block::nx(); x++)
                {
                    const label_t idx_in_warp = idx_block(x, y, z) % warp_size;
                    const label_t ID = idx_block(x, y, z) + (cycle * block::size());
                    if ((!s_buffer[ID] == 0))
                    {
                        std::cout << "pop[" << s_buffer[ID] << "], warpID[" << warp_idx(x, y, z) << "][" << idx_in_warp << "], threadID{" << x << ", " << y << ", " << z << "};" << std::endl;
                    }
                }
            }
        }
        std::cout << std::endl;
    }

    // for (label_t z = 0; z < block::nz(); z++)
    // {
    //     for (label_t y = 0; y < block::ny(); y++)
    //     {
    //         for (label_t x = 0; x < block::nx(); x++)
    //         {
    //             const label_t idx_in_warp = idx_block(x, y, z) % warp_size;
    //             std::cout << idx_in_warp << std::endl;
    //         }
    //     }
    // }
    // constexpr const thread::array<int, VelocitySet::Q()> C_ = VelocitySet::cx<int>() * VelocitySet::cy<int>();

    // constexpr const thread::array<int, number_non_zero(C_)> C = non_zero_values<number_non_zero(C_)>(C_);

    // for (label_t i = 0; i < C.size(); i++)
    // {
    //     std::cout << "cxcy[" << (i < 10 ? "0" : "") << i << "] = " << (C[i] == -1 ? "-1" : "+1") << std::endl;
    // }

    return 0;
}