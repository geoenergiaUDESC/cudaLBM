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
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Implementation of the multiphase moment representation with the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    multiphaseD3Q19.cu

\*---------------------------------------------------------------------------*/

#include "multiphaseD3Q19.cuh"

using namespace LBM;

__host__ [[nodiscard]] inline consteval label_t NStreams() noexcept { return 1; }

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Allocate the arrays on the device
    device::array<scalar_t, VelocitySet, time::instantaneous> rho("rho", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> u("u", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> v("v", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> w("w", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> mxx("m_xx", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> mxy("m_xy", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> mxz("m_xz", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> myy("m_yy", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> myz("m_yz", mesh, programCtrl);
    device::array<scalar_t, VelocitySet, time::instantaneous> mzz("m_zz", mesh, programCtrl);

    // Phase field arrays (solution field)
    device::array<scalar_t, PhaseVelocitySet, time::instantaneous> phi("phi", mesh, programCtrl);

    // Phase geometry arrays (raw device allocations; NOT auto-registered)
    scalar_t *d_normx = device::allocateArray<scalar_t>(mesh.nPoints(), static_cast<scalar_t>(0));
    scalar_t *d_normy = device::allocateArray<scalar_t>(mesh.nPoints(), static_cast<scalar_t>(0));
    scalar_t *d_normz = device::allocateArray<scalar_t>(mesh.nPoints(), static_cast<scalar_t>(0));
    scalar_t *d_ind = device::allocateArray<scalar_t>(mesh.nPoints(), static_cast<scalar_t>(0));
    scalar_t *d_ffx = device::allocateArray<scalar_t>(mesh.nPoints(), static_cast<scalar_t>(0));
    scalar_t *d_ffy = device::allocateArray<scalar_t>(mesh.nPoints(), static_cast<scalar_t>(0));
    scalar_t *d_ffz = device::allocateArray<scalar_t>(mesh.nPoints(), static_cast<scalar_t>(0));

    const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs(
        rho.ptr(),
        u.ptr(),
        v.ptr(),
        w.ptr(),
        mxx.ptr(),
        mxy.ptr(),
        mxz.ptr(),
        myy.ptr(),
        myz.ptr(),
        mzz.ptr(),
        phi.ptr());

    const device::ptrCollection<NUMBER_MOMENTS<false>(), scalar_t> hydroPtrs(
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

    // Setup Streams
    const streamHandler<NStreams()> streamsLBM;

    objectRegistry<VelocitySet, NStreams()> runTimeObjects(mesh, hydroPtrs, streamsLBM);

    device::halo<VelocitySet, config::periodicX, config::periodicY> fBlockHalo(mesh, programCtrl);      // Hydrodynamic halo
    device::halo<PhaseVelocitySet, config::periodicX, config::periodicY> gBlockHalo(mesh, programCtrl); // Phase field halo

    constexpr const label_t sharedMemoryAllocationSize = block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS<true>()>(sizeof(scalar_t));

    checkCudaErrors(cudaFuncSetCacheConfig(multiphaseD3Q19, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncSetAttribute(multiphaseD3Q19, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryAllocationSize));

    const runTimeIO IO(mesh, programCtrl);

    std::cout << std::endl;
    std::cout << "Allocating " << sharedMemoryAllocationSize << " bytes of shared memory to multiphaseD3Q" << VelocitySet::Q() << " kernel" << std::endl;
    std::cout << std::endl;

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
            fileIO::writeFile<time::instantaneous>(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                functionObjects::solutionVariableNames(true),
                host::toHost(devPtrs, mesh),
                timeStep);

            runTimeObjects.save(timeStep);
        }

        // Main kernels
        host::constexpr_for<0, NStreams()>(
            [&](const auto stream)
            {
                multiphaseD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), sharedMemoryAllocationSize, streamsLBM.streams()[stream]>>>(
                    devPtrs, d_ffx, d_ffy, d_ffz, d_normx, d_normy, d_normz,
                    fBlockHalo.fGhost(), fBlockHalo.gGhost(), gBlockHalo.fGhost(), gBlockHalo.gGhost());

                computeNormals<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(
                    phi.ptr(), d_normx, d_normy, d_normz, d_ind);

                computeForces<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(
                    d_normx, d_normy, d_normz, d_ind, d_ffx, d_ffy, d_ffz);
            });

        // Calculate S kernel
        runTimeObjects.calculate(timeStep);

        // Halo pointer swap
        fBlockHalo.swap();
        gBlockHalo.swap();
    }

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_normx));
    checkCudaErrors(cudaFree(d_normy));
    checkCudaErrors(cudaFree(d_normz));
    checkCudaErrors(cudaFree(d_ind));

    return 0;
}