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

namespace LBM
{
    template <const label_t N>
    class streamHandler
    {
    public:
        __host__ [[nodiscard]] streamHandler() noexcept
            : streams_(createCudaStreams())
        {
            static_assert(N > 0, "Number of CUDA streams must be positive!");
        }

        ~streamHandler() noexcept
        {
            if constexpr (N == 1)
            {
                cudaStreamSynchronize(streams_[0]);
                cudaStreamDestroy(streams_[0]);
            }
            else
            {
                for (label_t stream = 0; stream < N; stream++)
                {
                    cudaStreamSynchronize(streams_[stream]);
                }
                for (label_t stream = 0; stream < N; stream++)
                {
                    cudaStreamDestroy(streams_[stream]);
                }
            }
        }

        inline void synchronizeAll() const noexcept
        {
            host::constexpr_for<0, N>(
                [&](const auto stream)
                {
                    cudaStreamSynchronize(streams_[stream]);
                });
        }

        template <const label_t stream_>
        inline void synchronize(const std::integral_constant<label_t, stream_> stream) const noexcept
        {
            static_assert(stream_ < N, "Attempting to access an invalid stream: stream number must be < N");
            cudaStreamSynchronize(streams_[stream()]);
        }

        template <const label_t stream_>
        __device__ cudaStream_t &operator[](const std::integral_constant<label_t, stream_> stream) const noexcept
        {
            return streams_[stream()];
        }

        __host__ [[nodiscard]] inline const std::array<cudaStream_t, N> &streams() const noexcept
        {
            return streams_;
        }

    private:
        __host__ [[nodiscard]] static const std::array<cudaStream_t, N> createCudaStreams() noexcept
        {
            std::array<cudaStream_t, N> streamsLBM;

            for (label_t stream = 0; stream < N; stream++)
            {
                checkCudaErrors(cudaDeviceSynchronize());
                checkCudaErrors(cudaStreamCreate(&streamsLBM[stream]));
                checkCudaErrors(cudaDeviceSynchronize());
            }

            return streamsLBM;
        }

        const std::array<cudaStream_t, N> streams_;
    };
}

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
    device::array<scalar_t> rho(host::array<scalar_t, VelocitySet>("rho", mesh, programCtrl));
    device::array<scalar_t> u(host::array<scalar_t, VelocitySet>("u", mesh, programCtrl));
    device::array<scalar_t> v(host::array<scalar_t, VelocitySet>("v", mesh, programCtrl));
    device::array<scalar_t> w(host::array<scalar_t, VelocitySet>("w", mesh, programCtrl));
    device::array<scalar_t> mxx(host::array<scalar_t, VelocitySet>("m_xx", mesh, programCtrl));
    device::array<scalar_t> mxy(host::array<scalar_t, VelocitySet>("m_xy", mesh, programCtrl));
    device::array<scalar_t> mxz(host::array<scalar_t, VelocitySet>("m_xz", mesh, programCtrl));
    device::array<scalar_t> myy(host::array<scalar_t, VelocitySet>("m_yy", mesh, programCtrl));
    device::array<scalar_t> myz(host::array<scalar_t, VelocitySet>("m_yz", mesh, programCtrl));
    device::array<scalar_t> mzz(host::array<scalar_t, VelocitySet>("m_zz", mesh, programCtrl));

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
            fileIO::writeFile(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
                host::toHost(devPtrs, mesh),
                timeStep);
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

        // Halo pointer swap
        blockHalo.swap();
    }

    return 0;
}