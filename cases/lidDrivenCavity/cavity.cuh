/**
Filename: cavity.cuh
Contents: Case setup specific to the lid driven cavity case
**/

#ifndef __MBLBM_CAVITY_CUH
#define __MBLBM_CAVITY_CUH

#include "../../LBMIncludes.cuh"
#include "../../LBMTypedefs.cuh"

#include "../../velocitySet/velocitySet.cuh"
#include "../../boundaryConditions/boundaryConditions.cuh"

namespace LBM
{
    namespace host
    {
        __host__ [[nodiscard]] const std::vector<scalar_t> moments(const host::latticeMesh &mesh, const scalar_t u_inf)
        {
            std::vector<scalar_t> fMom(mesh.nx() * mesh.ny() * mesh.nz() * NUMBER_MOMENTS(), 0);

            // Loop over all grid points
            for (label_t bz = 0; bz < mesh.nzBlocks(); bz++)
            {
                for (label_t by = 0; by < mesh.nyBlocks(); by++)
                {
                    for (label_t bx = 0; bx < mesh.nxBlocks(); bx++)
                    {
                        for (label_t tz = 0; tz < block::nz(); tz++)
                        {
                            for (label_t ty = 0; ty < block::ny(); ty++)
                            {
                                for (label_t tx = 0; tx < block::nx(); tx++)
                                {
                                    // Zeroth moment (density fluctuation and velocity)
                                    fMom[idxMom<index::rho()>(tx, ty, tz, bx, by, bz, mesh)] = 0; // rho - rho0() = 0
                                    // Override for the top wall (y = mesh.ny()-1): ux = U_MAX
                                    if ((ty + (by * block::ny())) == (mesh.ny() - 1))
                                    {
                                        // fMom[idxMom<index::u()>(tx, ty, tz, bx, by, bz, nBlockx, nBlocky)] = VelocitySet::velocitySet::scale_i() * u_inf;
                                        fMom[idxMom<index::u()>(tx, ty, tz, bx, by, bz, mesh)] = u_inf;
                                    }
                                    else
                                    {
                                        fMom[idxMom<index::u()>(tx, ty, tz, bx, by, bz, mesh)] = 0;
                                    }
                                    fMom[idxMom<index::v()>(tx, ty, tz, bx, by, bz, mesh)] = 0;
                                    fMom[idxMom<index::w()>(tx, ty, tz, bx, by, bz, mesh)] = 0;

                                    // Second moments: compute equilibrium populations
                                    const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::F_eq(
                                        0,
                                        0,
                                        0);

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
                                    fMom[idxMom<index::xx()>(tx, ty, tz, bx, by, bz, mesh)] = VelocitySet::velocitySet::scale_ii() * pixx;
                                    fMom[idxMom<index::xy()>(tx, ty, tz, bx, by, bz, mesh)] = VelocitySet::velocitySet::scale_ij() * pixy;
                                    fMom[idxMom<index::xz()>(tx, ty, tz, bx, by, bz, mesh)] = VelocitySet::velocitySet::scale_ij() * pixz;
                                    fMom[idxMom<index::yy()>(tx, ty, tz, bx, by, bz, mesh)] = VelocitySet::velocitySet::scale_ii() * piyy;
                                    fMom[idxMom<index::yz()>(tx, ty, tz, bx, by, bz, mesh)] = VelocitySet::velocitySet::scale_ij() * piyz;
                                    fMom[idxMom<index::zz()>(tx, ty, tz, bx, by, bz, mesh)] = VelocitySet::velocitySet::scale_ii() * pizz;
                                }
                            }
                        }
                    }
                }
            }

            return fMom;
        }
    }

}

#endif