/**
Filename: cavity.cuh
Contents: Case setup specific to the lid driven cavity case
**/

#ifndef __MBLBM_CAVITY_CUH
#define __MBLBM_CAVITY_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

#include "velocitySet/velocitySet.cuh"
#include "boundaryConditions.cuh"

namespace LBM
{
    namespace host
    {
        __host__ [[nodiscard]] const std::vector<scalar_t> moments(const host::latticeMesh &mesh, const scalar_t u_inf)
        {
            const label_t nBlockx = mesh.nx() / block::nx();
            const label_t nBlocky = mesh.ny() / block::ny();
            std::vector<scalar_t> fMom(mesh.nx() * mesh.ny() * mesh.nz() * NUMBER_MOMENTS(), 0);

            // Loop over all grid points
            for (label_t x = 0; x < mesh.nx(); x++)
            {
                for (label_t y = 0; y < mesh.ny(); y++)
                {
                    for (label_t z = 0; z < mesh.nz(); z++)
                    {
                        // Default: no-slip (zero velocity) on all boundaries
                        constexpr const scalar_t ux = 0.0;
                        constexpr const scalar_t uy = 0.0;
                        constexpr const scalar_t uz = 0.0;

                        // Zeroth moment (density fluctuation and velocity)
                        fMom[idxMom<index::rho()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0; // rho - rho0() = 0
                        // Override for the top wall (y = mesh.ny()-1): ux = U_MAX
                        if (y == mesh.ny() - 1)
                        {
                            fMom[idxMom<index::u()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_i() * u_inf;
                        }
                        else
                        {
                            fMom[idxMom<index::u()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;
                        }
                        fMom[idxMom<index::v()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;
                        fMom[idxMom<index::w()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;

                        // Second moments: compute equilibrium populations
                        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::F_eq(ux, uy, uz);

                        // Compute second-order moments (reused terms)
                        const scalar_t pop_7_8 = pop[7] + pop[8];
                        const scalar_t pop_9_10 = pop[9] + pop[10];
                        const scalar_t pop_13_14 = pop[13] + pop[14];
                        const scalar_t pop_15_16 = pop[15] + pop[16];
                        const scalar_t pop_11_12 = pop[11] + pop[12];
                        const scalar_t pop_17_18 = pop[17] + pop[18];

                        const scalar_t pixx = (pop[1] + pop[2] + pop_7_8 + pop_9_10 + pop_13_14 + pop_15_16) - VelocitySet::velocitySet::cs2();
                        const scalar_t pixy = (pop_7_8 - pop_13_14);
                        const scalar_t pixz = (pop_9_10 - pop_15_16);
                        const scalar_t piyy = (pop[3] + pop[4] + pop_7_8 + pop_11_12 + pop_13_14 + pop_17_18) - VelocitySet::velocitySet::cs2();
                        const scalar_t piyz = (pop_11_12 - pop_17_18);
                        const scalar_t pizz = (pop[5] + pop[6] + pop_9_10 + pop_11_12 + pop_15_16 + pop_17_18) - VelocitySet::velocitySet::cs2();

                        // Store second-order moments
                        fMom[idxMom<index::xx()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_ii() * pixx;
                        fMom[idxMom<index::xy()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_ij() * pixy;
                        fMom[idxMom<index::xz()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_ij() * pixz;
                        fMom[idxMom<index::yy()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_ii() * piyy;
                        fMom[idxMom<index::yz()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_ij() * piyz;
                        fMom[idxMom<index::zz()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_ii() * pizz;
                    }
                }
            }

            return fMom;
        }

        // __host__ [[nodiscard]] const std::vector<momentArray> moments_v2(const host::latticeMesh &mesh, const scalar_t u_inf)
        // {
        //     const label_t nBlockx = mesh.nx() / block::nx();
        //     const label_t nBlocky = mesh.ny() / block::ny();
        //     std::vector<momentArray> fMom(mesh.nx() * mesh.ny() * mesh.nz(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

        //     // Loop over all grid points
        //     for (label_t x = 0; x < mesh.nx(); x++)
        //     {
        //         for (label_t y = 0; y < mesh.ny(); y++)
        //         {
        //             for (label_t z = 0; z < mesh.nz(); z++)
        //             {
        //                 // Default: no-slip (zero velocity) on all boundaries
        //                 // constexpr const scalar_t ux = 0.0;
        //                 // constexpr const scalar_t uy = 0.0;
        //                 // constexpr const scalar_t uz = 0.0;

        //                 // Zeroth moment (density fluctuation and velocity)
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].rho = 0; // rho - rho0() = 0
        //                 // Override for the top wall (y = mesh.ny()-1): ux = U_MAX
        //                 if (y == mesh.ny() - 1)
        //                 {
        //                     fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].u = VelocitySet::velocitySet::scale_i() * u_inf;
        //                 }
        //                 else
        //                 {
        //                     fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].u = 0;
        //                 }
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].w = 0;
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].v = 0;

        //                 // Second moments: compute equilibrium populations
        //                 const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::F_eq(
        //                     fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].u,
        //                     fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].v,
        //                     fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].w);

        //                 // Compute second-order moments (reused terms)
        //                 const scalar_t pop_7_8 = pop[7] + pop[8];
        //                 const scalar_t pop_9_10 = pop[9] + pop[10];
        //                 const scalar_t pop_13_14 = pop[13] + pop[14];
        //                 const scalar_t pop_15_16 = pop[15] + pop[16];
        //                 const scalar_t pop_11_12 = pop[11] + pop[12];
        //                 const scalar_t pop_17_18 = pop[17] + pop[18];

        //                 const scalar_t pixx = (pop[1] + pop[2] + pop_7_8 + pop_9_10 + pop_13_14 + pop_15_16) - VelocitySet::velocitySet::cs2();
        //                 const scalar_t pixy = (pop_7_8 - pop_13_14);
        //                 const scalar_t pixz = (pop_9_10 - pop_15_16);
        //                 const scalar_t piyy = (pop[3] + pop[4] + pop_7_8 + pop_11_12 + pop_13_14 + pop_17_18) - VelocitySet::velocitySet::cs2();
        //                 const scalar_t piyz = (pop_11_12 - pop_17_18);
        //                 const scalar_t pizz = (pop[5] + pop[6] + pop_9_10 + pop_11_12 + pop_15_16 + pop_17_18) - VelocitySet::velocitySet::cs2();

        //                 // Store second-order moments
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].m_xx = VelocitySet::velocitySet::scale_ii() * pixx;
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].m_xy = VelocitySet::velocitySet::scale_ij() * pixy;
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].m_xz = VelocitySet::velocitySet::scale_ij() * pixz;
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].m_yy = VelocitySet::velocitySet::scale_ii() * piyy;
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].m_yz = VelocitySet::velocitySet::scale_ij() * piyz;
        //                 fMom[idx(x, y, z, 0, 0, 0, nBlockx, nBlocky)].m_zz = VelocitySet::velocitySet::scale_ii() * pizz;
        //             }
        //         }
        //     }

        //     return fMom;
        // }

        // __host__ [[nodiscard]] const std::vector<momentArray> moments_aos(const host::latticeMesh &mesh, const scalar_t u_inf)
        // {
        //     // Calculate grid decomposition
        //     const label_t nBlockx = mesh.nx() / block::nx();
        //     const label_t nBlocky = mesh.ny() / block::ny();

        //     // Total grid points
        //     const size_t total_elements = mesh.nx() * mesh.ny() * mesh.nz();

        //     // Calculate number of chunks needed (round up)
        //     const size_t num_chunks = (total_elements + VEC_SIZE() - 1) / VEC_SIZE();

        //     std::cout << "num_chunks = " << num_chunks << std::endl;

        //     // Initialize AoSoA storage with zero values
        //     std::vector<momentArray> fMom(num_chunks);

        //     // Loop over all grid points
        //     for (label_t z = 0; z < mesh.nz(); z++)
        //     {
        //         for (label_t y = 0; y < mesh.ny(); y++)
        //         {
        //             // for (label_t z = 0; z < mesh.nz(); z++)
        //             for (label_t x = 0; x < mesh.nx(); x++)
        //             {
        //                 // Compute global index
        //                 const label_t gid = idx(
        //                     x, y, z,
        //                     x / block::nx(), y / block::ny(), z / block::nz(),
        //                     nBlockx, nBlocky);

        //                 if (gid > (mesh.nx() * mesh.ny() * mesh.nz()) - 1)
        //                 {
        //                     std::cout << "idx = " << gid << std::endl;
        //                 }

        //                 const label_t chunk_idx = gid / VEC_SIZE();
        //                 const label_t lane = gid % VEC_SIZE();

        //                 // Default: no-slip (zero velocity) on all boundaries
        //                 // constexpr scalar_t ux = 0.0;
        //                 // constexpr scalar_t uy = 0.0;
        //                 // constexpr scalar_t uz = 0.0;

        //                 // Zeroth moment (density fluctuation)
        //                 fMom[chunk_idx].rho[lane] = 0; // rho - rho0() = 0

        //                 // Velocity moments
        //                 if (y == mesh.ny() - 1)
        //                 {
        //                     // Top wall: ux = u_inf
        //                     fMom[chunk_idx].u[lane] = VelocitySet::velocitySet::scale_i() * u_inf;
        //                 }
        //                 else
        //                 {
        //                     fMom[chunk_idx].u[lane] = 0;
        //                 }
        //                 fMom[chunk_idx].v[lane] = 0; // v
        //                 fMom[chunk_idx].w[lane] = 0; // w

        //                 // Compute second moments
        //                 const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::F_eq(
        //                     fMom[chunk_idx].u[lane], fMom[chunk_idx].v[lane], fMom[chunk_idx].w[lane]);

        //                 // Reuse intermediate calculations
        //                 const scalar_t pop_7_8 = pop[7] + pop[8];
        //                 const scalar_t pop_9_10 = pop[9] + pop[10];
        //                 const scalar_t pop_13_14 = pop[13] + pop[14];
        //                 const scalar_t pop_15_16 = pop[15] + pop[16];
        //                 const scalar_t pop_11_12 = pop[11] + pop[12];
        //                 const scalar_t pop_17_18 = pop[17] + pop[18];

        //                 const scalar_t pixx = (pop[1] + pop[2] + pop_7_8 + pop_9_10 + pop_13_14 + pop_15_16) - VelocitySet::velocitySet::cs2();
        //                 const scalar_t pixy = pop_7_8 - pop_13_14;
        //                 const scalar_t pixz = pop_9_10 - pop_15_16;
        //                 const scalar_t piyy = (pop[3] + pop[4] + pop_7_8 + pop_11_12 + pop_13_14 + pop_17_18) - VelocitySet::velocitySet::cs2();
        //                 const scalar_t piyz = pop_11_12 - pop_17_18;
        //                 const scalar_t pizz = (pop[5] + pop[6] + pop_9_10 + pop_11_12 + pop_15_16 + pop_17_18) - VelocitySet::velocitySet::cs2();

        //                 // Store second-order moments
        //                 fMom[chunk_idx].m_xx[lane] = VelocitySet::velocitySet::scale_ii() * pixx; // xx
        //                 fMom[chunk_idx].m_xy[lane] = VelocitySet::velocitySet::scale_ij() * pixy; // xy
        //                 fMom[chunk_idx].m_xz[lane] = VelocitySet::velocitySet::scale_ij() * pixz; // xz
        //                 fMom[chunk_idx].m_yy[lane] = VelocitySet::velocitySet::scale_ii() * piyy; // yy
        //                 fMom[chunk_idx].m_yz[lane] = VelocitySet::velocitySet::scale_ij() * piyz; // yz
        //                 fMom[chunk_idx].m_zz[lane] = VelocitySet::velocitySet::scale_ii() * pizz; // zz
        //             }
        //         }
        //     }

        //     return fMom;
        // }

        __host__ [[nodiscard]] const std::vector<momentArray> moments_aos(const host::latticeMesh &mesh, const scalar_t u_inf)
        {
            // Calculate grid decomposition
            const label_t nBlockx = mesh.nx() / block::nx();
            const label_t nBlocky = mesh.ny() / block::ny();
            const label_t nBlockz = mesh.nz() / block::nz(); // Add z-dimension blocks

            // Total grid points
            const size_t total_elements = mesh.nx() * mesh.ny() * mesh.nz();
            const size_t num_chunks = (total_elements + VEC_SIZE() - 1) / VEC_SIZE();

            // Initialize AoSoA storage with zero values
            std::vector<momentArray> fMom(num_chunks);

            // Loop over all grid points
            for (label_t z = 0; z < mesh.nz(); z++)
            {
                for (label_t y = 0; y < mesh.ny(); y++)
                {
                    for (label_t x = 0; x < mesh.nx(); x++)
                    {
                        // Compute block indices
                        const label_t bx = x / block::nx();
                        const label_t by = y / block::ny();
                        const label_t bz = z / block::nz();

                        // Compute thread indices within block
                        const label_t tx = x % block::nx();
                        const label_t ty = y % block::ny();
                        const label_t tz = z % block::nz();

                        // CORRECTED INDEX CALCULATION
                        const size_t gid = tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + nBlockx * (by + nBlocky * bz))));

                        const size_t chunk_idx = gid / VEC_SIZE();
                        const size_t lane = gid % VEC_SIZE();

                        // Safety check
                        if (chunk_idx >= num_chunks || lane >= VEC_SIZE())
                        {
                            continue; // Skip out-of-bounds accesses
                        }

                        /// Zeroth moment (density fluctuation)
                        fMom[chunk_idx].rho[lane] = 0; // rho - rho0() = 0

                        // Velocity moments
                        if (y == mesh.ny() - 1)
                        {
                            // Top wall: ux = u_inf
                            fMom[chunk_idx].u[lane] = VelocitySet::velocitySet::scale_i() * u_inf;
                        }
                        else
                        {
                            fMom[chunk_idx].u[lane] = 0;
                        }
                        fMom[chunk_idx].v[lane] = 0; // v
                        fMom[chunk_idx].w[lane] = 0; // w

                        // Compute second moments
                        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::F_eq(
                            fMom[chunk_idx].u[lane], fMom[chunk_idx].v[lane], fMom[chunk_idx].w[lane]);

                        // Reuse intermediate calculations
                        const scalar_t pop_7_8 = pop[7] + pop[8];
                        const scalar_t pop_9_10 = pop[9] + pop[10];
                        const scalar_t pop_13_14 = pop[13] + pop[14];
                        const scalar_t pop_15_16 = pop[15] + pop[16];
                        const scalar_t pop_11_12 = pop[11] + pop[12];
                        const scalar_t pop_17_18 = pop[17] + pop[18];

                        const scalar_t pixx = (pop[1] + pop[2] + pop_7_8 + pop_9_10 + pop_13_14 + pop_15_16) - VelocitySet::velocitySet::cs2();
                        const scalar_t pixy = pop_7_8 - pop_13_14;
                        const scalar_t pixz = pop_9_10 - pop_15_16;
                        const scalar_t piyy = (pop[3] + pop[4] + pop_7_8 + pop_11_12 + pop_13_14 + pop_17_18) - VelocitySet::velocitySet::cs2();
                        const scalar_t piyz = pop_11_12 - pop_17_18;
                        const scalar_t pizz = (pop[5] + pop[6] + pop_9_10 + pop_11_12 + pop_15_16 + pop_17_18) - VelocitySet::velocitySet::cs2();

                        // Store second-order moments
                        fMom[chunk_idx].m_xx[lane] = VelocitySet::velocitySet::scale_ii() * pixx; // xx
                        fMom[chunk_idx].m_xy[lane] = VelocitySet::velocitySet::scale_ij() * pixy; // xy
                        fMom[chunk_idx].m_xz[lane] = VelocitySet::velocitySet::scale_ij() * pixz; // xz
                        fMom[chunk_idx].m_yy[lane] = VelocitySet::velocitySet::scale_ii() * piyy; // yy
                        fMom[chunk_idx].m_yz[lane] = VelocitySet::velocitySet::scale_ij() * piyz; // yz
                        fMom[chunk_idx].m_zz[lane] = VelocitySet::velocitySet::scale_ii() * pizz; // zz
                    }
                }
            }
            return fMom;
        }

    }

}

#endif