/**
Filename: ghostPtrs.cuh
Contents: A handling the ghost interface pointers on the device
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_GHOSTPTRS_CUH
#define __MBLBM_GHOSTPTRS_CUH

namespace mbLBM
{
    /**
     * @brief Struct holding the number of lattice points on each face of a CUDA block
     **/
    template <class M>
    struct nGhostFace
    {
    public:
        /**
         * @brief Construct from a latticeMesh object
         * @return An nGhostFace struct with xy, xz and yz defined by the mesh
         **/
        [[nodiscard]] nGhostFace(const M &mesh) noexcept
            : xy(block::nx() * block::ny() * mesh.nBlocks()),
              xz(block::nx() * block::nz() * mesh.nBlocks()),
              yz(block::ny() * block::nz() * mesh.nBlocks()) {};

        const label_t xy;
        const label_t xz;
        const label_t yz;
    };

    namespace host
    {

        [[nodiscard]] static inline consteval label_t X0() noexcept { return 0; }
        [[nodiscard]] static inline consteval label_t X1() noexcept { return 1; }
        [[nodiscard]] static inline consteval label_t Y0() noexcept { return 2; }
        [[nodiscard]] static inline consteval label_t Y1() noexcept { return 3; }
        [[nodiscard]] static inline consteval label_t Z0() noexcept { return 4; }
        [[nodiscard]] static inline consteval label_t Z1() noexcept { return 5; }

        class ghostPtrs
        {
        public:
            [[nodiscard]] ghostPtrs(const host::moments &moms) noexcept
                : x0_(foo<X0()>(moms)),
                  x1_(foo<X1()>(moms)),
                  y0_(foo<Y0()>(moms)),
                  y1_(foo<Y1()>(moms)),
                  z0_(foo<Z0()>(moms)),
                  z1_(foo<Z1()>(moms)) {};

            // [[nodiscard]] ghostPtrs(const host::moments &moms, const scalar_t value) noexcept
            //     : x0_(foo<X0()>(moms, value)),
            //       x1_(foo<X1()>(moms, value)),
            //       y0_(foo<Y0()>(moms, value)),
            //       y1_(foo<Y1()>(moms, value)),
            //       z0_(foo<Z0()>(moms, value)),
            //       z1_(foo<Z1()>(moms, value)) {};

            ~ghostPtrs() {};

            [[nodiscard]] inline constexpr const scalarArray &x0() const noexcept
            {
                return x0_;
            }
            [[nodiscard]] inline constexpr const scalarArray &x1() const noexcept
            {
                return x1_;
            }
            [[nodiscard]] inline constexpr const scalarArray &y0() const noexcept
            {
                return y0_;
            }
            [[nodiscard]] inline constexpr const scalarArray &y1() const noexcept
            {
                return y1_;
            }
            [[nodiscard]] inline constexpr const scalarArray &z0() const noexcept
            {
                return z0_;
            }
            [[nodiscard]] inline constexpr const scalarArray &z1() const noexcept
            {
                return z1_;
            }

        private:
            const scalarArray x0_;
            const scalarArray x1_;
            const scalarArray y0_;
            const scalarArray y1_;
            const scalarArray z0_;
            const scalarArray z1_;

            template <const label_t type>
            [[nodiscard]] inline constexpr label_t getNumberFaces(const host::moments &mesh) const noexcept
            {
                static_assert((type == X0()) || (type == X1()) || (type == Y0()) || (type == Y1()) || (type == Z0()) || (type == Z1()));

                // const nGhostFace nFaces(mesh);

                if constexpr ((type == X0()) || (type == X1()))
                {
                    return block::ny() * block::nz() * mesh.nxBlocks() * mesh.nyBlocks() * mesh.nzBlocks();
                }

                if constexpr ((type == Y0()) || (type == Y1()))
                {
                    return block::nx() * block::nz() * mesh.nxBlocks() * mesh.nyBlocks() * mesh.nzBlocks();
                }

                if constexpr ((type == Z0()) || (type == Z1()))
                {
                    return block::nx() * block::ny() * mesh.nxBlocks() * mesh.nyBlocks() * mesh.nzBlocks();
                }
            }

            template <const label_t type>
            [[nodiscard]] scalarArray foo(const host::moments &moms) noexcept
            {
                static_assert((type == X0()) || (type == X1()) || (type == Y0()) || (type == Y1()) || (type == Z0()) || (type == Z1()));

                constexpr const scalar_t RHO_0 = 1.0;

                scalarArray_t arr(getNumberFaces<type>(moms) * vSet::QF(), 0);

                for (label_t blockIDz = 0; blockIDz < moms.nzBlocks(); blockIDz++)
                {
                    for (label_t blockIDy = 0; blockIDy < moms.nyBlocks(); blockIDy++)
                    {
                        for (label_t blockIDx = 0; blockIDx < moms.nxBlocks(); blockIDx++)
                        {
                            if constexpr ((type == X0()) || (type == X1()))
                            {
                                for (label_t z = 0; z < block::nz(); z++)
                                {
                                    for (label_t y = 0; y < block::ny(); y++)
                                    {
                                        // Hold x = 0 or nx - 1
                                        if constexpr (type == X0())
                                        {
                                            const std::array<scalar_t, 10> moments_0 = {
                                                moms.rho()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())] + RHO_0,
                                                moms.u()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.v()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.w()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xx()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xy()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xz()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yy()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yz()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_zz()[idxMom(0, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())]};
                                            const std::array<scalar_t, 19> pop_0 = vSet::reconstruct(moments_0);
                                            arr[idxPopX<0>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[2];
                                            arr[idxPopX<1>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[8];
                                            arr[idxPopX<2>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[10];
                                            arr[idxPopX<3>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[14];
                                            arr[idxPopX<4>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[16];
                                        }
                                        if constexpr (type == X1())
                                        {
                                            const std::array<scalar_t, 10> moments_1 = {
                                                moms.rho()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())] + RHO_0,
                                                moms.u()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.v()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.w()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xx()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xy()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xz()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yy()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yz()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_zz()[idxMom(block::nx() - 1, y, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())]};
                                            const std::array<scalar_t, 19> pop_1 = vSet::reconstruct(moments_1);
                                            arr[idxPopX<0>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[1];
                                            arr[idxPopX<1>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[7];
                                            arr[idxPopX<2>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[9];
                                            arr[idxPopX<3>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[13];
                                            arr[idxPopX<4>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[15];
                                        }
                                    }
                                }
                            }

                            if constexpr ((type == Y0()) || (type == Y1()))
                            {
                                for (label_t z = 0; z < block::nz(); z++)
                                {
                                    for (label_t x = 0; x < block::nx(); x++)
                                    {
                                        // Hold y = 0 or ny - 1
                                        if constexpr (type == Y0())
                                        {
                                            const std::array<scalar_t, 10> moments_0 = {
                                                moms.rho()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())] + RHO_0,
                                                moms.u()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.v()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.w()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xx()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xy()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xz()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yy()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yz()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_zz()[idxMom(x, 0, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())]};
                                            const std::array<scalar_t, 19> pop_0 = vSet::reconstruct(moments_0);
                                            arr[idxPopY<0>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[4];
                                            arr[idxPopY<1>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[8];
                                            arr[idxPopY<2>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[12];
                                            arr[idxPopY<3>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[13];
                                            arr[idxPopY<4>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[18];
                                        }
                                        if constexpr (type == Y1())
                                        {
                                            const std::array<scalar_t, 10> moments_1 = {
                                                moms.rho()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())] + RHO_0,
                                                moms.u()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.v()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.w()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xx()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xy()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xz()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yy()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yz()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_zz()[idxMom(x, block::ny() - 1, z, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())]};
                                            const std::array<scalar_t, 19> pop_1 = vSet::reconstruct(moments_1);
                                            arr[idxPopY<0>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[3];
                                            arr[idxPopY<1>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[7];
                                            arr[idxPopY<2>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[11];
                                            arr[idxPopY<3>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[14];
                                            arr[idxPopY<4>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[17];
                                        }
                                    }
                                }
                            }

                            if constexpr ((type == Z0()) || (type == Z1()))
                            {
                                for (label_t y = 0; y < block::ny(); y++)
                                {
                                    for (label_t x = 0; x < block::nx(); x++)
                                    {
                                        // Hold z = 0 or nz - 1
                                        if constexpr (type == Z0())
                                        {
                                            const std::array<scalar_t, 10> moments_0 = {
                                                moms.rho()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())] + RHO_0,
                                                moms.u()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.v()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.w()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xx()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xy()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xz()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yy()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yz()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_zz()[idxMom(x, y, 0, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())]};
                                            const std::array<scalar_t, 19> pop_0 = vSet::reconstruct(moments_0);
                                            arr[idxPopZ<0>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[6];
                                            arr[idxPopZ<1>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[10];
                                            arr[idxPopZ<2>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[12];
                                            arr[idxPopZ<3>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[15];
                                            arr[idxPopZ<4>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_0[17];
                                        }
                                        if constexpr (type == Z1())
                                        {
                                            const std::array<scalar_t, 10> moments_1 = {
                                                moms.rho()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())] + RHO_0,
                                                moms.u()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.v()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.w()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xx()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xy()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_xz()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yy()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_yz()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())],
                                                moms.m_zz()[idxMom(x, y, block::nz() - 1, blockIDx, blockIDy, blockIDz, moms.nx(), moms.ny())]};
                                            const std::array<scalar_t, 19> pop_1 = vSet::reconstruct(moments_1);
                                            arr[idxPopZ<0>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[5];
                                            arr[idxPopZ<1>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[9];
                                            arr[idxPopZ<2>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[11];
                                            arr[idxPopZ<3>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[16];
                                            arr[idxPopZ<4>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = pop_1[18];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (type == X0())
                {
                    return scalarArray(moms.mesh(), arr, "x0");
                }
                if (type == X1())
                {
                    return scalarArray(moms.mesh(), arr, "x1");
                }
                if (type == Y0())
                {
                    return scalarArray(moms.mesh(), arr, "y0");
                }
                if (type == Y1())
                {
                    return scalarArray(moms.mesh(), arr, "y1");
                }
                if (type == Z0())
                {
                    return scalarArray(moms.mesh(), arr, "z0");
                }
                if (type == Z1())
                {
                    return scalarArray(moms.mesh(), arr, "z1");
                }
            }

            template <const label_t type>
            [[nodiscard]] scalarArray foo(const host::moments &moms, const scalar_t value) noexcept
            {
                static_assert((type == X0()) || (type == X1()) || (type == Y0()) || (type == Y1()) || (type == Z0()) || (type == Z1()));

                // constexpr const scalar_t RHO_0 = 1.0;

                scalarArray_t arr(getNumberFaces<type>(moms) * vSet::QF(), 0);

                for (label_t blockIDz = 0; blockIDz < moms.nzBlocks(); blockIDz++)
                {
                    for (label_t blockIDy = 0; blockIDy < moms.nyBlocks(); blockIDy++)
                    {
                        for (label_t blockIDx = 0; blockIDx < moms.nxBlocks(); blockIDx++)
                        {
                            if constexpr ((type == X0()) || (type == X1()))
                            {
                                for (label_t z = 0; z < block::nz(); z++)
                                {
                                    for (label_t y = 0; y < block::ny(); y++)
                                    {
                                        arr[idxPopX<0>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopX<1>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopX<2>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopX<3>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopX<4>(y, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                    }
                                }
                            }

                            if constexpr ((type == Y0()) || (type == Y1()))
                            {
                                for (label_t z = 0; z < block::nz(); z++)
                                {
                                    for (label_t x = 0; x < block::nx(); x++)
                                    {
                                        // Hold y = 0 or ny - 1
                                        arr[idxPopY<0>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopY<1>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopY<2>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopY<3>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopY<4>(x, z, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                    }
                                }
                            }

                            if constexpr ((type == Z0()) || (type == Z1()))
                            {
                                for (label_t y = 0; y < block::ny(); y++)
                                {
                                    for (label_t x = 0; x < block::nx(); x++)
                                    {
                                        // Hold z = 0 or nz - 1
                                        arr[idxPopZ<0>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopZ<1>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopZ<2>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopZ<3>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                        arr[idxPopZ<4>(x, y, blockIDx, blockIDy, blockIDz, moms.nxBlocks(), moms.nyBlocks())] = value;
                                    }
                                }
                            }
                        }
                    }
                }

                if (type == X0())
                {
                    return scalarArray(moms.mesh(), arr, "x0");
                }
                if (type == X1())
                {
                    return scalarArray(moms.mesh(), arr, "x1");
                }
                if (type == Y0())
                {
                    return scalarArray(moms.mesh(), arr, "y0");
                }
                if (type == Y1())
                {
                    return scalarArray(moms.mesh(), arr, "y1");
                }
                if (type == Z0())
                {
                    return scalarArray(moms.mesh(), arr, "z0");
                }
                if (type == Z1())
                {
                    return scalarArray(moms.mesh(), arr, "z1");
                }
            }

            template <label_t pop>
            [[nodiscard]] inline constexpr label_t idxPopX(
                const label_t ty,
                const label_t tz,
                const label_t bx,
                const label_t by,
                const label_t bz,
                const label_t NUM_BLOCK_X,
                const label_t NUM_BLOCK_Y)
            {
                return ty + block::ny() * (tz + block::nz() * (pop + vSet::QF() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
            }

            template <label_t pop>
            [[nodiscard]] inline constexpr label_t idxPopY(
                const label_t tx,
                const label_t tz,
                const label_t bx,
                const label_t by,
                const label_t bz,
                const label_t NUM_BLOCK_X,
                const label_t NUM_BLOCK_Y)
            {
                return tx + block::nx() * (tz + block::nz() * (pop + vSet::QF() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
            }

            template <label_t pop>
            [[nodiscard]] inline constexpr label_t idxPopZ(
                const label_t tx,
                const label_t ty,
                const label_t bx,
                const label_t by,
                const label_t bz,
                const label_t NUM_BLOCK_X,
                const label_t NUM_BLOCK_Y)
            {
                return tx + block::nx() * (ty + block::nz() * (pop + vSet::QF() * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
            }
        };
    }

    namespace device
    {
        class ghostPtrs
        {
        public:
            /**
             * @brief Constructs a list of 6 unique pointers from a latticeMesh object
             * @return A ghostPtrs object constructed from the mesh
             * @param mesh The mesh used to define the amount of memory to allocate to the pointers
             **/
            [[nodiscard]] ghostPtrs(const host::ghostPtrs &ghost) noexcept
                : x0_(ghost.x0()),
                  x1_(ghost.x1()),
                  y0_(ghost.y0()),
                  y1_(ghost.y1()),
                  z0_(ghost.z0()),
                  z1_(ghost.z1()) {};

            /**
             * @brief Destructor for the ghostPtrs object
             **/
            ~ghostPtrs() noexcept {
#ifdef VERBOSE
            // std::cout << "Freeing ghostPtrs object" << std::endl;
#endif
            };

            /**
             * @brief Provides access to the underlying pointers
             * @return An immutable reference to a unique pointer
             **/
            __host__ [[nodiscard]] inline const scalar_t *x0() const noexcept
            {
                return x0_.ptr();
            }
            __host__ [[nodiscard]] inline const scalar_t *x1() const noexcept
            {
                return x1_.ptr();
            }
            __host__ [[nodiscard]] inline const scalar_t *y0() const noexcept
            {
                return y0_.ptr();
            }
            __host__ [[nodiscard]] inline const scalar_t *y1() const noexcept
            {
                return y1_.ptr();
            }
            __host__ [[nodiscard]] inline const scalar_t *z0() const noexcept
            {
                return z0_.ptr();
            }
            __host__ [[nodiscard]] inline const scalar_t *z1() const noexcept
            {
                return z1_.ptr();
            }

            /**
             * @brief Provides access to the underlying pointers
             * @return A mutable reference to a unique pointer
             **/
            __host__ [[nodiscard]] inline scalar_t *x0() noexcept
            {
                return x0_.ptr();
            }
            __host__ [[nodiscard]] inline scalar_t *x1() noexcept
            {
                return x1_.ptr();
            }
            __host__ [[nodiscard]] inline scalar_t *y0() noexcept
            {
                return y0_.ptr();
            }
            __host__ [[nodiscard]] inline scalar_t *y1() noexcept
            {
                return y1_.ptr();
            }
            __host__ [[nodiscard]] inline scalar_t *z0() noexcept
            {
                return z0_.ptr();
            }
            __host__ [[nodiscard]] inline scalar_t *z1() noexcept
            {
                return z1_.ptr();
            }

            __host__ [[nodiscard]] inline scalar_t *ptrRestrict &x0Ref() noexcept
            {
                return x0_.ptrRef();
            }
            __host__ [[nodiscard]] inline scalar_t *ptrRestrict &x1Ref() noexcept
            {
                return x1_.ptrRef();
            }
            __host__ [[nodiscard]] inline scalar_t *ptrRestrict &y0Ref() noexcept
            {
                return y0_.ptrRef();
            }
            __host__ [[nodiscard]] inline scalar_t *ptrRestrict &y1Ref() noexcept
            {
                return y1_.ptrRef();
            }
            __host__ [[nodiscard]] inline scalar_t *ptrRestrict &z0Ref() noexcept
            {
                return z0_.ptrRef();
            }
            __host__ [[nodiscard]] inline scalar_t *ptrRestrict &z1Ref() noexcept
            {
                return z1_.ptrRef();
            }

            /**
             * @brief Returns the number of lattice points on each block face
             * @return Number of lattice points on a block face as a label_t
             **/
            // [[nodiscard]] inline constexpr label_t n_xy() const noexcept
            // {
            //     return nGhostFaces_.xy;
            // }
            // [[nodiscard]] inline constexpr label_t n_xz() const noexcept
            // {
            //     return nGhostFaces_.xz;
            // }
            // [[nodiscard]] inline constexpr label_t n_yz() const noexcept
            // {
            //     return nGhostFaces_.yz;
            // }

        private:
            /**
             * @brief Number of lattice points on each face of a CUDA block
             **/
            // const nGhostFace<latticeMesh> nGhostFaces_;

            /**
             * @brief The underlying device scalar arrays
             **/
            scalarArray x0_;
            scalarArray x1_;
            scalarArray y0_;
            scalarArray y1_;
            scalarArray z0_;
            scalarArray z1_;
        };
    }
}

#endif