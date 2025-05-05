#ifdef INITIALCONDITIONSREADY

inline void initialConditionApply(
    scalarArray_t &rho,
    scalarArray_t &u,
    scalarArray_t &v,
    scalarArray_t &w,
    scalarArray_t &mom_0,
    scalarArray_t &mom_1,
    scalarArray_t &mom_2,
    scalarArray_t &mom_3,
    scalarArray_t &mom_4,
    scalarArray_t &mom_5,
    const nodeTypeArray_t nodeTypes,
    const std::size_t i,
    const scalar_t Re,
    const scalar_t u_inf,
    const label_t nx) noexcept
{
    switch (nodeTypes[i])
    {
    case nodeType::UNDEFINED:
    {
        // std::cout << "Undefined node" << std::endl;
        return;
    }
    case nodeType::BULK:
    {
        // std::cout << "Bulk node" << std::endl;
        // Set density to initial value
        rho[i] = rho_init - RHO_0;

        // Set velocity to initial value
        u[i] = VelocitySet::velocitySet::F_M_I_SCALE() * ux_init;
        v[i] = VelocitySet::velocitySet::F_M_I_SCALE() * uy_init;
        w[i] = VelocitySet::velocitySet::F_M_I_SCALE() * uz_init;

        // Second moments
        // Define equilibrium populations
        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t invRho = 1.0 / rho_init;
        const scalar_t pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - VelocitySet::velocitySet::cs2();
        const scalar_t pixy = ((pop[7] + pop[8]) - (pop[13] + pop[14])) * invRho;
        const scalar_t pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
        const scalar_t piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - VelocitySet::velocitySet::cs2();
        const scalar_t piyz = ((pop[11] + pop[12]) - (pop[17] + pop[18])) * invRho;
        const scalar_t pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - VelocitySet::velocitySet::cs2();

        mom_0[i] = VelocitySet::velocitySet::F_M_II_SCALE() * pixx;
        mom_1[i] = VelocitySet::velocitySet::F_M_IJ_SCALE() * pixy;
        mom_2[i] = VelocitySet::velocitySet::F_M_IJ_SCALE() * pixz;
        mom_3[i] = VelocitySet::velocitySet::F_M_II_SCALE() * piyy;
        mom_4[i] = VelocitySet::velocitySet::F_M_IJ_SCALE() * piyz;
        mom_5[i] = VelocitySet::velocitySet::F_M_II_SCALE() * pizz;
        return;
    }
    case nodeType::SOUTHWESTBACK:
    {
        // printf("Doing SOUTHWESTBACK\n");
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];

        const scalar_t rho_loc = (12.0 * rho_I) / 7.0;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTHWESTFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];

        const scalar_t rho_loc = (12 * rho_I) / 7;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHWESTBACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];

        const scalar_t rho_loc = (12 * rho_I) / 7;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHWESTFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];

        const scalar_t rho_loc = (12 * rho_I) / 7;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTHEASTBACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];

        const scalar_t rho_loc = (12 * rho_I) / 7;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTHEASTFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];

        const scalar_t rho_loc = (12 * rho_I) / 7;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHEASTBACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];

        const scalar_t rho_loc = (12 * rho_I) / 7;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHEASTFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];

        const scalar_t rho_loc = (12 * rho_I) / 7;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTHWEST:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (pop[8]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (OMEGA * m_xy_I - m_xy_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = (25 * m_xy_I - 1) / (9 * (OMEGA * m_xy_I - m_xy_I + 1));
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHWEST:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (-pop[14]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (m_xy_I - OMEGA * m_xy_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = (25 * m_xy_I + 1) / (9 * (m_xy_I - OMEGA * m_xy_I + 1));
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTHEAST:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (-pop[13]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (m_xy_I - OMEGA * m_xy_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = (25 * m_xy_I + 1) / (9 * (m_xy_I - OMEGA * m_xy_I + 1));
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHEAST:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (pop[7]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (OMEGA * m_xy_I - m_xy_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = (25 * m_xy_I - 1) / (9 * (OMEGA * m_xy_I - m_xy_I + 1));
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::WESTBACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xz_I = inv_rho_I * (pop[10]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (OMEGA * m_xz_I - m_xz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = (25 * m_xz_I - 1) / (9 * (OMEGA * m_xz_I - m_xz_I + 1));
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::WESTFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xz_I = inv_rho_I * (-pop[16]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (m_xz_I - OMEGA * m_xz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = (25 * m_xz_I + 1) / (9 * (m_xz_I - OMEGA * m_xz_I + 1));
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::EASTBACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xz_I = inv_rho_I * (-pop[15]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (m_xz_I - OMEGA * m_xz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = (25 * m_xz_I + 1) / (9 * (m_xz_I - OMEGA * m_xz_I + 1));
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::EASTFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xz_I = inv_rho_I * (pop[9]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (OMEGA * m_xz_I - m_xz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = (25 * m_xz_I - 1) / (9 * (OMEGA * m_xz_I - m_xz_I + 1));
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTHBACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_yz_I = inv_rho_I * (pop[12]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (OMEGA * m_yz_I - m_yz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = (25 * m_yz_I - 1) / (9 * (OMEGA * m_yz_I - m_yz_I + 1));
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTHFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_yz_I = inv_rho_I * (-pop[18]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (m_yz_I - OMEGA * m_yz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = (25 * m_yz_I + 1) / (9 * (m_yz_I - OMEGA * m_yz_I + 1));
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHBACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_yz_I = inv_rho_I * (-pop[17]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (m_yz_I - OMEGA * m_yz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = (25 * m_yz_I + 1) / (9 * (m_yz_I - OMEGA * m_yz_I + 1));
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTHFRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_yz_I = inv_rho_I * (pop[11]);

        const scalar_t OMEGA = omega(Re, u_inf, nx);
        const scalar_t rho_loc = (36 * rho_I * (OMEGA * m_yz_I - m_yz_I + 1)) / (OMEGA + 24);

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = (25 * m_yz_I - 1) / (9 * (OMEGA * m_yz_I - m_yz_I + 1));
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::WEST:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[14]);
        const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[16]);

        const scalar_t rho_loc = (6 * rho_I) / 5;

        mom_0[i] = 0;
        mom_1[i] = (5 * m_xy_I) / 3;
        mom_2[i] = (5 * m_xz_I) / 3;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::EAST:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[13]);
        const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[15]);

        const scalar_t rho_loc = (6 * rho_I) / 5;

        mom_0[i] = 0;
        mom_1[i] = (5 * m_xy_I) / 3;
        mom_2[i] = (5 * m_xz_I) / 3;
        mom_3[i] = 0;
        mom_4[i] = 0;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::SOUTH:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (pop[8] - pop[13]);
        const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[18]);

        const scalar_t rho_loc = (6 * rho_I) / 5;

        mom_0[i] = 0;
        mom_1[i] = (5 * m_xy_I) / 3;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = (5 * m_yz_I) / 3;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::NORTH:
    {
        u[i] = u_inf;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, u_inf, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xy_I = inv_rho_I * (pop[7] - pop[14]);
        const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[17]);

        const scalar_t rho_loc = (6 * rho_I) / 5;

        mom_0[i] = (6 * u_inf * u_inf * rho_I) / 5;
        mom_1[i] = (5 * m_xy_I) / 3 - u_inf / 3;
        mom_2[i] = 0;
        mom_3[i] = 0;
        mom_4[i] = (5 * m_yz_I) / 3;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::BACK:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xz_I = inv_rho_I * (pop[10] - pop[15]);
        const scalar_t m_yz_I = inv_rho_I * (pop[12] - pop[17]);

        const scalar_t rho_loc = (6 * rho_I) / 5;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = (5 * m_xz_I) / 3;
        mom_3[i] = 0;
        mom_4[i] = (5 * m_yz_I) / 3;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    case nodeType::FRONT:
    {
        u[i] = 0;
        v[i] = 0;
        w[i] = 0;

        const std::array<scalar_t, VelocitySet::D3Q19::Q()> pop = equilibriumDistribution(1, 0, 0, 0);

        const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
        const scalar_t inv_rho_I = 1.0 / rho_I;
        const scalar_t m_xz_I = inv_rho_I * (pop[9] - pop[16]);
        const scalar_t m_yz_I = inv_rho_I * (pop[11] - pop[18]);

        const scalar_t rho_loc = (6 * rho_I) / 5;

        mom_0[i] = 0;
        mom_1[i] = 0;
        mom_2[i] = (5 * m_xz_I) / 3;
        mom_3[i] = 0;
        mom_4[i] = (5 * m_yz_I) / 3;
        mom_5[i] = 0;

        rho[i] = rho_loc;

        return;
    }
    }
}

void initialCondition(
    scalarArray_t &rho,
    scalarArray_t &u,
    scalarArray_t &v,
    scalarArray_t &w,
    scalarArray_t &mom_0,
    scalarArray_t &mom_1,
    scalarArray_t &mom_2,
    scalarArray_t &mom_3,
    scalarArray_t &mom_4,
    scalarArray_t &mom_5,
    const nodeTypeArray_t &nodeTypes,
    const scalar_t Re,
    const scalar_t u_inf,
    const label_t nx) noexcept
{
    std::size_t pctCounter = 0;
    const std::size_t n = nodeTypes.size();
    std::size_t iCounter = 0;

    std::cout << "Entered initialCondition function" << std::endl;
    for (std::size_t i = 0; i < nodeTypes.size(); i++)
    {
        pctCounter++;
        initialConditionApply(rho, u, v, w, mom_0, mom_1, mom_2, mom_3, mom_4, mom_5, nodeTypes, i, Re, u_inf, nx);
        if ((i % (n / 10) == 0))
        {
            if (i > 0)
            {
                std::cout << "Done " << iCounter * 10 << "%" << std::endl;
            }
            iCounter++;
            pctCounter = 0;
        }
    }
}

#endif