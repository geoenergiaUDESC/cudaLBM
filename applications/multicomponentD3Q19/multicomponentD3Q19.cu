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
    Implementation of the multicomponent moment representation with the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    multicomponentD3Q19.cu

\*---------------------------------------------------------------------------*/

#include "multicomponentD3Q19.cuh"

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

    // Allocate the hydrodynamic arrays on the device
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

    // Allocate the phase field on the device
    device::array<scalar_t, VelocitySet, time::instantaneous> phi("phi", mesh, programCtrl);

    const device::ptrCollection<10, scalar_t> hydroPtrs(
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

    const device::ptrCollection<1, scalar_t> phasePtrs(
        phi.ptr());

    // Setup Streams
    const streamHandler<NStreams()> streamsLBM;
  
    // Object registry
    objectRegistry<VelocitySet, NStreams()> runTimeObjects(mesh, hydroPtrs, streamsLBM);
    objectRegistry<VelocitySet, NStreams()> runTimeObjects(mesh, phasePtrs, streamsLBM);

    // Halos
    device::halo<VelocitySet> fBlockHalo(mesh, programCtrl);
    device::halo<VelocitySet> gBlockHalo(mesh, programCtrl);

    constexpr const label_t sharedMemoryAllocationSize = block::sharedMemoryBufferSize<VelocitySet, 10>(sizeof(scalar_t));

    checkCudaErrors(cudaFuncSetCacheConfig(multicomponentD3Q19, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncSetAttribute(multicomponentD3Q19, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryAllocationSize));

    const runTimeIO IO(mesh, programCtrl);

    std::cout << std::endl;
    std::cout << "Allocating " << sharedMemoryAllocationSize << " bytes of shared memory to multicomponentD3Q" << VelocitySet::Q() << " kernel" << std::endl;
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
                functionObjects::solutionVariableNames,
                host::toHost(hydroPtrs, mesh),
                timeStep);

            runTimeObjects.save(timeStep);
        }

        // Main kernel
        host::constexpr_for<0, NStreams()>(
            [&](const auto stream)
            {
                multicomponentD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), sharedMemoryAllocationSize, streamsLBM.streams()[stream]>>>(hydroPtrs, phasePtrs, fBlockHalo.fGhost(), fBlockHalo.gGhost(), gBlockHalo.fGhost(), gBlockHalo.gGhost());
            });

        // Calculate S kernel
        runTimeObjects.calculate(timeStep);

        // Halo pointer swap
        fBlockHalo.swap();
        gBlockHalo.swap();
    }

    return 0;
}