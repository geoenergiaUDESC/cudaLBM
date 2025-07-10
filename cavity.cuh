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
        __host__ [[nodiscard]] const std::vector<std::vector<scalar_t>> moments_v2(const host::latticeMesh &mesh, const scalar_t u_inf)
        {
            const label_t nBlockx = mesh.nx() / block::nx();
            const label_t nBlocky = mesh.ny() / block::ny();

            std::vector<std::vector<scalar_t>> fMom(NUMBER_MOMENTS(), std::vector<scalar_t>(mesh.nx() * mesh.ny() * mesh.nz(), 0));

            // Loop over all grid points
            for (label_t x = 0; x < mesh.nx(); x++)
            {
                for (label_t y = 0; y < mesh.ny(); y++)
                {
                    for (label_t z = 0; z < mesh.nz(); z++)
                    {
                        // Default: no-slip (zero velocity) on all boundaries
                        // constexpr const scalar_t ux = 0.0;
                        // constexpr const scalar_t uy = 0.0;
                        // constexpr const scalar_t uz = 0.0;

                        // Zeroth moment (density fluctuation and velocity)
                        // fMom[idxMom<index::rho()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0; // rho - rho0() = 0
                        fMom[0][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = 0;
                        // Override for the top wall (y = mesh.ny()-1): ux = U_MAX
                        if (y == mesh.ny() - 1)
                        {
                            fMom[1][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = VelocitySet::velocitySet::scale_i() * u_inf;
                        }
                        else
                        {
                            fMom[1][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = 0;
                        }
                        fMom[2][idxMom<index::v()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;
                        fMom[3][idxMom<index::w()>(x, y, z, 0, 0, 0, nBlockx, nBlocky)] = 0;

                        // Second moments: compute equilibrium populations
                        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = VelocitySet::D3Q19::F_eq(
                            fMom[1][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())],
                            fMom[2][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())],
                            fMom[3][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())]);

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
                        fMom[4][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = VelocitySet::velocitySet::scale_ii() * pixx;
                        fMom[5][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = VelocitySet::velocitySet::scale_ij() * pixy;
                        fMom[6][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = VelocitySet::velocitySet::scale_ij() * pixz;
                        fMom[7][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = VelocitySet::velocitySet::scale_ii() * piyy;
                        fMom[8][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = VelocitySet::velocitySet::scale_ij() * piyz;
                        fMom[9][host::idxScalarGlobal(x, y, z, mesh.nx(), mesh.ny())] = VelocitySet::velocitySet::scale_ii() * pizz;
                    }
                }
            }
            return fMom;
        }

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

        // __host__ [[nodiscard]] const std::vector<nodeType_t> nodeType(const latticeMesh &mesh) noexcept
        // {
        //     std::vector<nodeType_t> nodeTypes(mesh.nx() * mesh.ny() * mesh.nz(), INTERIOR());

        //     for (label_t x = 0; x < mesh.nx(); x++)
        //     {
        //         for (label_t y = 0; y < mesh.ny(); y++)
        //         {
        //             for (label_t z = 0; z < mesh.nz(); z++)
        //             {
        //                 nodeTypes[idxScalarBlock(x % block::nx(), y % block::ny(), z % block::nz(), x / block::nx(), y / block::ny(), z / block::nz(), mesh.nx(), mesh.ny())] = boundaryConditions::initialCondition(x, y, z, mesh.nx(), mesh.ny(), mesh.nz());
        //             }
        //         }
        //     }

        //     return nodeTypes;
        // }
    }

    struct vector
    {
        scalar_t x;
        scalar_t y;
        scalar_t z;
    };

    // This can probably be pre - computed on the CPU before transferring to the GPU
    __device__ vector compute_wall_normal()
    {
        const label_t i = (threadIdx.x + blockDim.x * blockIdx.x);
        const label_t j = (threadIdx.y + blockDim.y * blockIdx.y);
        const label_t k = (threadIdx.z + blockDim.z * blockIdx.z);
        vector normal{0.0, 0.0, 0.0};

        // X-direction boundaries
        if (i == 0)
        {
            normal.x = 1.0;
        } // West face
        if (i == d_nx - 1)
        {
            normal.x = 1.0;
        } // East face

        // Y-direction boundaries
        if (j == 0)
        {
            normal.y = 1.0;
        } // South face
        if (j == d_ny - 1)
        {
            normal.y = 1.0;
        } // North face

        // Z-direction boundaries
        if (k == 0)
        {
            normal.z = 1.0;
        } // Bottom face
        if (k == d_nz - 1)
        {
            normal.z = 1.0;
        } // Top face

        // Normalize vector
        const scalar_t length = sqrt(
            normal.x * normal.x +
            normal.y * normal.y +
            normal.z * normal.z);

        if (length > 1e-10)
        {
            normal.x /= length;
            normal.y /= length;
            normal.z /= length;
        }

        return normal;
    }

    __device__ void fixedVelocity(
        scalar_t *const ptrRestrict moments, const scalar_t (&ptrRestrict pop)[19],
        const vector u_b, const vector normal, const scalar_t omega,
        const vector *const ptrRestrict c)
    {
        // Step 1: Compute known density and stress
        scalar_t rho_known = 0.0f;
        scalar_t sigma_xx_k = 0.0f, sigma_xy_k = 0.0f, sigma_xz_k = 0.0f;
        scalar_t sigma_yy_k = 0.0f, sigma_yz_k = 0.0f, sigma_zz_k = 0.0f;
        const scalar_t cs2 = 1.0f / 3.0f; // D3Q19 value

        for (int i = 0; i < 19; i++)
        {
            const scalar_t ci_dot_n = c[i].x * normal.x + c[i].y * normal.y + c[i].z * normal.z;

            if (ci_dot_n <= 0.0f)
            {
                const scalar_t pop_val = pop[i];
                const scalar_t cx = c[i].x, cy = c[i].y, cz = c[i].z;

                rho_known += pop_val;
                sigma_xx_k += pop_val * (cx * cx - cs2);
                sigma_xy_k += pop_val * cx * cy;
                sigma_xz_k += pop_val * cx * cz;
                sigma_yy_k += pop_val * (cy * cy - cs2);
                sigma_yz_k += pop_val * cy * cz;
                sigma_zz_k += pop_val * (cz * cz - cs2);
            }
        }

        // Normalize by known density
        if (rho_known > 1e-6f)
        {
            const scalar_t inv_rho_k = 1.0f / rho_known;
            sigma_xx_k *= inv_rho_k;
            sigma_xy_k *= inv_rho_k;
            sigma_xz_k *= inv_rho_k;
            sigma_yy_k *= inv_rho_k;
            sigma_yz_k *= inv_rho_k;
            sigma_zz_k *= inv_rho_k;
        }

        // Step 2: Estimate boundary density
        const scalar_t u_dot_n = u_b.x * normal.x + u_b.y * normal.y + u_b.z * normal.z;
        const scalar_t rho_boundary = (1.0f - u_dot_n > 1e-8f) ? rho_known / (1.0f - u_dot_n) : rho_known;

        // Step 3: Set conserved moments
        moments[0] = rho_boundary;
        moments[1] = u_b.x;
        moments[2] = u_b.y;
        moments[3] = u_b.z;

        // Step 4: Compute normal stress component
        const scalar_t nx = normal.x, ny = normal.y, nz = normal.z;
        const scalar_t sigma_nn_k = sigma_xx_k * nx * nx + sigma_yy_k * ny * ny + sigma_zz_k * nz * nz + 2.0f * (sigma_xy_k * nx * ny + sigma_xz_k * nx * nz + sigma_yz_k * ny * nz);

        // Step 5: Reconstruction factors
        scalar_t A = 3.0f * sigma_nn_k - 3.0f * omega * sigma_nn_k + 4.0f;
        const scalar_t denom = omega + 9.0f;
        if (fabsf(A) < 1e-6f)
        {
            A = copysignf(1e-6f, A);
        }

        // Step 6: Set non-conserved moments (with cs2 subtraction)
        moments[4] = (4.0f * denom * (10.0f * sigma_xx_k - sigma_zz_k)) / (99.0f * A);
        moments[5] = (18.0f * sigma_xy_k - 4.0f * u_b.x + 2.0f * omega * sigma_xy_k - 3.0f * u_b.x * sigma_nn_k + 3.0f * omega * u_b.x * sigma_nn_k) / (3.0f * A);
        moments[6] = (sigma_xz_k * denom) / (3.0f * A);
        moments[7] = (15.0f * sigma_nn_k + 2.0f) / (3.0f * A);
        moments[8] = (2.0f * sigma_yz_k * denom) / (3.0f * A);
        moments[9] = -4.0f * (sigma_xx_k - 10.0f * sigma_zz_k) * denom / (99.0f * A);
    }

}

#endif