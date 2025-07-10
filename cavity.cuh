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
    }

}

#endif