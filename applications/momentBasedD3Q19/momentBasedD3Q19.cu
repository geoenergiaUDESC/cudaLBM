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

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Setup Streams
    const std::array<cudaStream_t, 1> streamsLBM = host::createCudaStream();

    // Allocate the arrays on the host first
    const host::array<scalar_t, VelocitySet> h_rho("rho", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_u("u", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_v("v", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_w("w", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_m_xx("m_xx", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_m_xy("m_xy", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_m_xz("m_xz", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_m_yy("m_yy", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_m_yz("m_yz", mesh, programCtrl);
    const host::array<scalar_t, VelocitySet> h_m_zz("m_zz", mesh, programCtrl);

    device::array<scalar_t> rho(h_rho, mesh);
    device::array<scalar_t> u(h_u, mesh);
    device::array<scalar_t> v(h_v, mesh);
    device::array<scalar_t> w(h_w, mesh);
    device::array<scalar_t> mxx(h_m_xx, mesh);
    device::array<scalar_t> mxy(h_m_xy, mesh);
    device::array<scalar_t> mxz(h_m_xz, mesh);
    device::array<scalar_t> myy(h_m_yy, mesh);
    device::array<scalar_t> myz(h_m_yz, mesh);
    device::array<scalar_t> mzz(h_m_zz, mesh);

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

    device::halo<VelocitySet> blockHalo(
        {h_rho.arr(),
         h_u.arr(),
         h_v.arr(),
         h_w.arr(),
         h_m_xx.arr(),
         h_m_xy.arr(),
         h_m_xz.arr(),
         h_m_yy.arr(),
         h_m_yz.arr(),
         h_m_zz.arr()},
        mesh);

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
        // if (programCtrl.save(timeStep))
        // {
        //     fileIO::writeFile(
        //         programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
        //         mesh,
        //         {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
        //         host::toHost(devPtrs, mesh),
        //         timeStep);
        // }

        // Main kernel
        momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM[0]>>>(
            devPtrs,
            blockHalo.fGhost(),
            blockHalo.gGhost());

        // Halo pointer swap
        blockHalo.swap();
    }

    return 0;
}