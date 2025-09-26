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

#include "momentBasedD3Q19.cuh"

using namespace LBM;

__host__ [[nodiscard]] inline consteval label_t NStreams() noexcept { return 1; }

// const bool

// Definition of a test kernel for S_xx
// This kind of register-light kernel may benefit from varying the block dimensions
// Also see if we can maximise the occupancy since we are not using shared mem
launchBounds __global__ void StrainRateTensor(
    const scalar_t *const ptrRestrict u_Alpha,
    const scalar_t *const ptrRestrict u_Beta,
    const scalar_t *const ptrRestrict m_AlphaBeta,
    scalar_t *const ptrRestrict S_AlphaBeta)
{
    // Calculate the index
    const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

    // Read from global memory
    const scalar_t uAlpha = u_Alpha[idx];
    const scalar_t uBeta = u_Beta[idx];
    const scalar_t mAlphaBeta = m_AlphaBeta[idx];

    // Calculate here
    const scalar_t S = velocitySet::as2<scalar_t>() * ((uAlpha * uBeta) - mAlphaBeta) / device::tau;

    // Coalesced write to global memory
    S_AlphaBeta[idx] = S;
}

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const bool calculateStrainRateTensor = true;

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Allocate the arrays on the device
    device::array<scalar_t, VelocitySet, tType::instantaneous> rho("rho", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> u("u", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> v("v", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> w("w", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> mxx("m_xx", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> mxy("m_xy", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> mxz("m_xz", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> myy("m_yy", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> myz("m_yz", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, tType::instantaneous> mzz("m_zz", mesh, programCtrl);

    const device::ptrCollection<10, scalar_t> devPtrs(
        rho.ptr(),
        u.ptr(),
        v.ptr(),
        w.ptr(),
        mxx.ptr(),
        mxy.ptr(),
        mxz.ptr(),
        myy.ptr(),
        myz.ptr(),
        mzz.ptr());

    device::halo<VelocitySet> blockHalo(mesh, programCtrl);

    // Setup Streams
    const streamHandler<NStreams()> streamsLBM;

    // Allocate the strain rate tensor components on the device
    // Fix this to make it initialise from an initial condition or checkpoint file as in other arrays
    device::array<scalar_t, VelocitySet, tType::instantaneous> S_xx("S_xx", mesh, 0);
    device::array<scalar_t, VelocitySet, tType::instantaneous> S_xy("S_xy", mesh, 0);
    device::array<scalar_t, VelocitySet, tType::instantaneous> S_xz("S_xz", mesh, 0);
    device::array<scalar_t, VelocitySet, tType::instantaneous> S_yy("S_yy", mesh, 0);
    device::array<scalar_t, VelocitySet, tType::instantaneous> S_yz("S_yz", mesh, 0);
    device::array<scalar_t, VelocitySet, tType::instantaneous> S_zz("S_zz", mesh, 0);
    const device::ptrCollection<6, scalar_t> SPtrs(
        S_xx.ptr(),
        S_xy.ptr(),
        S_xz.ptr(),
        S_yy.ptr(),
        S_yz.ptr(),
        S_zz.ptr());

    // const label_t z_stream_segment_size = mesh.nz() / NStreams();

    // const dim3 blockDimensions{
    //     static_cast<uint32_t>(mesh.nx() / block::nx()),
    //     static_cast<uint32_t>(mesh.ny() / block::ny()),
    //     static_cast<uint32_t>(mesh.nz() / (NStreams() * block::nz()))};

    checkCudaErrors(cudaFuncSetCacheConfig(momentBasedD3Q19, cudaFuncCachePreferShared));

    const runTimeIO IO(mesh, programCtrl);

    for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        // Do the run-time IO
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << std::endl;
        }

        // Checkpoint
        if (programCtrl.save(timeStep))
        {
            fileIO::writeFile<tType::instantaneous>(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                host::toHost(devPtrs, mesh),
                timeStep);

            if (calculateStrainRateTensor)
            {
                fileIO::writeFile<tType::instantaneous>(
                    "S_" + std::to_string(timeStep) + ".LBMBin",
                    mesh,
                    {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"},
                    host::toHost(SPtrs, mesh),
                    timeStep);
            }
        }

        // Main kernel
        host::constexpr_for<0, NStreams()>(
            [&](const auto stream)
            {
                momentBasedD3Q19<<<
                    mesh.gridBlock(),
                    mesh.threadBlock(),
                    0,
                    streamsLBM.streams()[stream]>>>(
                    devPtrs,
                    blockHalo.fGhost(),
                    blockHalo.gGhost());
            });

        // Calculate S kernel
        if (calculateStrainRateTensor)
        {
            host::constexpr_for<0, NStreams()>(
                [&](const auto stream)
                {
                    StrainRateTensor<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(u.ptr(), u.ptr(), mxx.ptr(), S_xx.ptr());
                    StrainRateTensor<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(u.ptr(), v.ptr(), mxy.ptr(), S_xy.ptr());
                    StrainRateTensor<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(u.ptr(), w.ptr(), mxz.ptr(), S_xz.ptr());
                    StrainRateTensor<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(v.ptr(), v.ptr(), myy.ptr(), S_yy.ptr());
                    StrainRateTensor<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(v.ptr(), w.ptr(), myz.ptr(), S_yz.ptr());
                    StrainRateTensor<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(w.ptr(), w.ptr(), mzz.ptr(), S_zz.ptr());
                });
        }

        // Halo pointer swap
        blockHalo.swap();
    }

    return 0;
}