__launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void alt_collide(
    scalar_t *const ptrRestrict rho,
    scalar_t *const ptrRestrict u,
    scalar_t *const ptrRestrict v,
    scalar_t *const ptrRestrict w,
    scalar_t *const ptrRestrict m_xx,
    scalar_t *const ptrRestrict m_xy,
    scalar_t *const ptrRestrict m_xz,
    scalar_t *const ptrRestrict m_yy,
    scalar_t *const ptrRestrict m_yz,
    scalar_t *const ptrRestrict m_zz,
    const nodeType::type *const ptrRestrict nodeTypes,
    scalar_t *const ptrRestrict f_x0,
    scalar_t *const ptrRestrict f_x1,
    scalar_t *const ptrRestrict f_y0,
    scalar_t *const ptrRestrict f_y1,
    scalar_t *const ptrRestrict f_z0,
    scalar_t *const ptrRestrict f_z1,
    scalar_t *const ptrRestrict g_x0,
    scalar_t *const ptrRestrict g_x1,
    scalar_t *const ptrRestrict g_y0,
    scalar_t *const ptrRestrict g_y1,
    scalar_t *const ptrRestrict g_z0,
    scalar_t *const ptrRestrict g_z1)
{
    // Definition of xyz is correct
    const label_t x = (threadIdx.x + blockDim.x * blockIdx.x);
    const label_t y = (threadIdx.y + blockDim.y * blockIdx.y);
    const label_t z = (threadIdx.z + blockDim.z * blockIdx.z);

    // Check if we should do an early return
    if ((x >= d_nx) || (y >= d_ny) || (z >= d_nz))
    {
        return;
    }

    constexpr const scalar_t RHO_0 = 1.0;

    // Definition of idxMom is correct
    scalar_t moments[10] = {
        rho[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] + RHO_0,
        u[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        v[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        w[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        m_xx[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        m_xy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        m_xz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        m_yy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        m_yz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)],
        m_zz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)]};

    // Perform the reconstruction
    scalar_t pop[VelocitySet::D3Q19::Q()];
    {
        const scalar_t pics2 = 1.0 - VelocitySet::velocitySet::cs2() * (moments[4] + moments[7] + moments[9]);

        const scalar_t multiplyTerm_0 = moments[0] * VelocitySet::D3Q19::w_0();
        pop[0] = multiplyTerm_0 * pics2;

        const scalar_t multiplyTerm_1 = moments[0] * VelocitySet::D3Q19::w_1();
        pop[1] = multiplyTerm_1 * (pics2 + moments[1] + moments[4]);
        pop[2] = multiplyTerm_1 * (pics2 - moments[1] + moments[4]);
        pop[3] = multiplyTerm_1 * (pics2 + moments[2] + moments[7]);
        pop[4] = multiplyTerm_1 * (pics2 - moments[2] + moments[7]);
        pop[5] = multiplyTerm_1 * (pics2 + moments[3] + moments[9]);
        pop[6] = multiplyTerm_1 * (pics2 - moments[3] + moments[9]);

        const scalar_t multiplyTerm_2 = moments[0] * VelocitySet::D3Q19::w_2();
        pop[7] = multiplyTerm_2 * (pics2 + moments[1] + moments[2] + moments[4] + moments[7] + moments[5]);
        pop[8] = multiplyTerm_2 * (pics2 - moments[1] - moments[2] + moments[4] + moments[7] + moments[5]);
        pop[9] = multiplyTerm_2 * (pics2 + moments[1] + moments[3] + moments[4] + moments[9] + moments[6]);
        pop[10] = multiplyTerm_2 * (pics2 - moments[1] - moments[3] + moments[4] + moments[9] + moments[6]);
        pop[11] = multiplyTerm_2 * (pics2 + moments[2] + moments[3] + moments[7] + moments[9] + moments[8]);
        pop[12] = multiplyTerm_2 * (pics2 - moments[2] - moments[3] + moments[7] + moments[9] + moments[8]);
        pop[13] = multiplyTerm_2 * (pics2 + moments[1] - moments[2] + moments[4] + moments[7] - moments[5]);
        pop[14] = multiplyTerm_2 * (pics2 - moments[1] + moments[2] + moments[4] + moments[7] - moments[5]);
        pop[15] = multiplyTerm_2 * (pics2 + moments[1] - moments[3] + moments[4] + moments[9] - moments[6]);
        pop[16] = multiplyTerm_2 * (pics2 - moments[1] + moments[3] + moments[4] + moments[9] - moments[6]);
        pop[17] = multiplyTerm_2 * (pics2 + moments[2] - moments[3] + moments[7] + moments[9] - moments[8]);
        pop[18] = multiplyTerm_2 * (pics2 - moments[2] + moments[3] + moments[7] + moments[9] - moments[8]);
    }

    __shared__ scalar_t s_pop[block::size() * (VelocitySet::D3Q19::Q() - 1)];
    {
        s_pop[idxPopBlock<0>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[1];
        s_pop[idxPopBlock<1>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[2];
        s_pop[idxPopBlock<2>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[3];
        s_pop[idxPopBlock<3>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[4];
        s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[5];
        s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[6];
        s_pop[idxPopBlock<6>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[7];
        s_pop[idxPopBlock<7>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[8];
        s_pop[idxPopBlock<8>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[9];
        s_pop[idxPopBlock<9>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[10];
        s_pop[idxPopBlock<10>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[11];
        s_pop[idxPopBlock<11>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[12];
        s_pop[idxPopBlock<12>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[13];
        s_pop[idxPopBlock<13>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[14];
        s_pop[idxPopBlock<14>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[15];
        s_pop[idxPopBlock<15>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[16];
        s_pop[idxPopBlock<16>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[17];
        s_pop[idxPopBlock<17>(threadIdx.x, threadIdx.y, threadIdx.z)] = pop[18];
    }
    __syncthreads();

    // Pull from shared mem
    {
        const label_t xp1 = (threadIdx.x + 1 + block::nx()) % block::nx();
        const label_t xm1 = (threadIdx.x - 1 + block::nx()) % block::nx();

        const label_t yp1 = (threadIdx.y + 1 + block::ny()) % block::ny();
        const label_t ym1 = (threadIdx.y - 1 + block::ny()) % block::ny();

        const label_t zp1 = (threadIdx.z + 1 + block::nz()) % block::nz();
        const label_t zm1 = (threadIdx.z - 1 + block::nz()) % block::nz();

        pop[1] = s_pop[idxPopBlock<0>(xm1, threadIdx.y, threadIdx.z)];
        pop[2] = s_pop[idxPopBlock<1>(xp1, threadIdx.y, threadIdx.z)];
        pop[3] = s_pop[idxPopBlock<2>(threadIdx.x, ym1, threadIdx.z)];
        pop[4] = s_pop[idxPopBlock<3>(threadIdx.x, yp1, threadIdx.z)];
        pop[5] = s_pop[idxPopBlock<4>(threadIdx.x, threadIdx.y, zm1)];
        pop[6] = s_pop[idxPopBlock<5>(threadIdx.x, threadIdx.y, zp1)];
        pop[7] = s_pop[idxPopBlock<6>(xm1, ym1, threadIdx.z)];
        pop[8] = s_pop[idxPopBlock<7>(xp1, yp1, threadIdx.z)];
        pop[9] = s_pop[idxPopBlock<8>(xm1, threadIdx.y, zm1)];
        pop[10] = s_pop[idxPopBlock<9>(xp1, threadIdx.y, zp1)];
        pop[11] = s_pop[idxPopBlock<10>(threadIdx.x, ym1, zm1)];
        pop[12] = s_pop[idxPopBlock<11>(threadIdx.x, yp1, zp1)];
        pop[13] = s_pop[idxPopBlock<12>(xm1, yp1, threadIdx.z)];
        pop[14] = s_pop[idxPopBlock<13>(xp1, ym1, threadIdx.z)];
        pop[15] = s_pop[idxPopBlock<14>(xm1, threadIdx.y, zp1)];
        pop[16] = s_pop[idxPopBlock<15>(xp1, threadIdx.y, zm1)];
        pop[17] = s_pop[idxPopBlock<16>(threadIdx.x, ym1, zp1)];
        pop[18] = s_pop[idxPopBlock<17>(threadIdx.x, yp1, zm1)];
    }

    // popLoad
    {
        const label_t tx = threadIdx.x;
        const label_t ty = threadIdx.y;
        const label_t tz = threadIdx.z;

        const label_t bx = blockIdx.x;
        const label_t by = blockIdx.y;
        const label_t bz = blockIdx.z;

        const label_t txp1 = (tx + 1 + block::nx()) % block::nx();
        const label_t txm1 = (tx - 1 + block::nx()) % block::nx();

        const label_t typ1 = (ty + 1 + block::ny()) % block::ny();
        const label_t tym1 = (ty - 1 + block::ny()) % block::ny();

        const label_t tzp1 = (tz + 1 + block::nz()) % block::nz();
        const label_t tzm1 = (tz - 1 + block::nz()) % block::nz();

        const label_t bxm1 = (bx - 1 + NUM_BLOCK_X) % NUM_BLOCK_X;
        const label_t bxp1 = (bx + 1 + NUM_BLOCK_X) % NUM_BLOCK_X;

        const label_t bym1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
        const label_t byp1 = (by + 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;

        const label_t bzm1 = (bz - 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;
        const label_t bzp1 = (bz + 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;

        if (tx == 0)
        { // w
            pop[1] = f_x1[idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bxm1, by, bz)];
            pop[7] = f_x1[idxPopX<1, VelocitySet::D3Q19::QF()>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)];
            pop[9] = f_x1[idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))];
            pop[13] = f_x1[idxPopX<3, VelocitySet::D3Q19::QF()>(typ1, tz, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)];
            pop[15] = f_x1[idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tzp1, bxm1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
        }
        else if (tx == (block::nx() - 1))
        { // e
            pop[2] = f_x0[idxPopX<0, VelocitySet::D3Q19::QF()>(ty, tz, bxp1, by, bz)];
            pop[8] = f_x0[idxPopX<1, VelocitySet::D3Q19::QF()>(typ1, tz, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)];
            pop[10] = f_x0[idxPopX<2, VelocitySet::D3Q19::QF()>(ty, tzp1, bxp1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
            pop[14] = f_x0[idxPopX<3, VelocitySet::D3Q19::QF()>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)];
            pop[16] = f_x0[idxPopX<4, VelocitySet::D3Q19::QF()>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))];
        }

        if (ty == 0)
        { // s
            pop[3] = f_y1[idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, bym1, bz)];
            pop[7] = f_y1[idxPopY<1, VelocitySet::D3Q19::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)];
            pop[11] = f_y1[idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))];
            pop[14] = f_y1[idxPopY<3, VelocitySet::D3Q19::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, bz)];
            pop[17] = f_y1[idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tzp1, bx, bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
        }
        else if (ty == (block::ny() - 1))
        { // n
            pop[4] = f_y0[idxPopY<0, VelocitySet::D3Q19::QF()>(tx, tz, bx, byp1, bz)];
            pop[8] = f_y0[idxPopY<1, VelocitySet::D3Q19::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, bz)];
            pop[12] = f_y0[idxPopY<2, VelocitySet::D3Q19::QF()>(tx, tzp1, bx, byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))];
            pop[13] = f_y0[idxPopY<3, VelocitySet::D3Q19::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)];
            pop[18] = f_y0[idxPopY<4, VelocitySet::D3Q19::QF()>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))];
        }

        if (tz == 0)
        { // b
            pop[5] = f_z1[idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bzm1)];
            pop[9] = f_z1[idxPopZ<1, VelocitySet::D3Q19::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)];
            pop[11] = f_z1[idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)];
            pop[16] = f_z1[idxPopZ<3, VelocitySet::D3Q19::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzm1)];
            pop[18] = f_z1[idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)];
        }
        else if (tz == (block::nz() - 1))
        { // f
            pop[6] = f_z0[idxPopZ<0, VelocitySet::D3Q19::QF()>(tx, ty, bx, by, bzp1)];
            pop[10] = f_z0[idxPopZ<1, VelocitySet::D3Q19::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzp1)];
            pop[12] = f_z0[idxPopZ<2, VelocitySet::D3Q19::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)];
            pop[15] = f_z0[idxPopZ<3, VelocitySet::D3Q19::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)];
            pop[17] = f_z0[idxPopZ<4, VelocitySet::D3Q19::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)];
        }
    }

    // Perform the moment calculation
    const nodeType::type nodeType = nodeTypes[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

    if (nodeType != nodeType::BULK)
    {
        boundaryConditions::calculateMoments(pop, moments, nodeType);
    }
    else
    {
        VelocitySet::D3Q19::calculateMoments(pop, moments);
    }

    // Scale the moments
    {
        moments[0] = moments[0] * VelocitySet::velocitySet::F_M_0_SCALE();

        moments[1] = moments[1] * VelocitySet::velocitySet::F_M_I_SCALE();
        moments[2] = moments[2] * VelocitySet::velocitySet::F_M_I_SCALE();
        moments[3] = moments[3] * VelocitySet::velocitySet::F_M_I_SCALE();

        moments[4] = moments[4] * VelocitySet::velocitySet::F_M_II_SCALE();
        moments[5] = moments[5] * VelocitySet::velocitySet::F_M_IJ_SCALE();
        moments[6] = moments[6] * VelocitySet::velocitySet::F_M_IJ_SCALE();
        moments[7] = moments[7] * VelocitySet::velocitySet::F_M_II_SCALE();
        moments[8] = moments[8] * VelocitySet::velocitySet::F_M_IJ_SCALE();
        moments[9] = moments[9] * VelocitySet::velocitySet::F_M_II_SCALE();
    }

    // Collide
    {
        const scalar_t t_omegaVar = 1.0 - d_omega;
        const scalar_t omegaVar_d2 = d_omega * 0.5;

        moments[4] = (t_omegaVar * moments[4] + omegaVar_d2 * moments[1] * moments[1]);
        moments[7] = (t_omegaVar * moments[7] + omegaVar_d2 * moments[2] * moments[2]);
        moments[9] = (t_omegaVar * moments[9] + omegaVar_d2 * moments[3] * moments[3]);

        moments[5] = (t_omegaVar * moments[5] + d_omega * moments[1] * moments[2]);
        moments[6] = (t_omegaVar * moments[6] + d_omega * moments[1] * moments[3]);
        moments[8] = (t_omegaVar * moments[8] + d_omega * moments[2] * moments[3]);
    }

    // Reconstruct
    {
        const scalar_t pics2 = 1.0 - VelocitySet::velocitySet::cs2() * (moments[4] + moments[7] + moments[9]);

        const scalar_t multiplyTerm_0 = moments[0] * VelocitySet::D3Q19::w_0();
        pop[0] = multiplyTerm_0 * pics2;

        const scalar_t multiplyTerm_1 = moments[0] * VelocitySet::D3Q19::w_1();
        pop[1] = multiplyTerm_1 * (pics2 + moments[1] + moments[4]);
        pop[2] = multiplyTerm_1 * (pics2 - moments[1] + moments[4]);
        pop[3] = multiplyTerm_1 * (pics2 + moments[2] + moments[7]);
        pop[4] = multiplyTerm_1 * (pics2 - moments[2] + moments[7]);
        pop[5] = multiplyTerm_1 * (pics2 + moments[3] + moments[9]);
        pop[6] = multiplyTerm_1 * (pics2 - moments[3] + moments[9]);

        const scalar_t multiplyTerm_2 = moments[0] * VelocitySet::D3Q19::w_2();
        pop[7] = multiplyTerm_2 * (pics2 + moments[1] + moments[2] + moments[4] + moments[7] + moments[5]);
        pop[8] = multiplyTerm_2 * (pics2 - moments[1] - moments[2] + moments[4] + moments[7] + moments[5]);
        pop[9] = multiplyTerm_2 * (pics2 + moments[1] + moments[3] + moments[4] + moments[9] + moments[6]);
        pop[10] = multiplyTerm_2 * (pics2 - moments[1] - moments[3] + moments[4] + moments[9] + moments[6]);
        pop[11] = multiplyTerm_2 * (pics2 + moments[2] + moments[3] + moments[7] + moments[9] + moments[8]);
        pop[12] = multiplyTerm_2 * (pics2 - moments[2] - moments[3] + moments[7] + moments[9] + moments[8]);
        pop[13] = multiplyTerm_2 * (pics2 + moments[1] - moments[2] + moments[4] + moments[7] - moments[5]);
        pop[14] = multiplyTerm_2 * (pics2 - moments[1] + moments[2] + moments[4] + moments[7] - moments[5]);
        pop[15] = multiplyTerm_2 * (pics2 + moments[1] - moments[3] + moments[4] + moments[9] - moments[6]);
        pop[16] = multiplyTerm_2 * (pics2 - moments[1] + moments[3] + moments[4] + moments[9] - moments[6]);
        pop[17] = multiplyTerm_2 * (pics2 + moments[2] - moments[3] + moments[7] + moments[9] - moments[8]);
        pop[18] = multiplyTerm_2 * (pics2 - moments[2] + moments[3] + moments[7] + moments[9] - moments[8]);
    }

    // Write to global memory
    rho[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[0] - RHO_0;
    u[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[1];
    v[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[2];
    w[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[3];
    m_xx[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[4];
    m_xy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[5];
    m_xz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[6];
    m_yy[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[7];
    m_yz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[8];
    m_zz[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = moments[9];

    // Save population to global memory
    {
        {
            const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
            const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
            const label_t z = threadIdx.z + blockDim.z * blockIdx.z;
            /* write to global pop */
            if (INTERFACE_BC_WEST(x))
            { // w
                g_x0[idxPopX<0, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[2];
                g_x0[idxPopX<1, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[8];
                g_x0[idxPopX<2, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[10];
                g_x0[idxPopX<3, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[14];
                g_x0[idxPopX<4, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[16];
            }
            if (INTERFACE_BC_EAST(x))
            { // e
                g_x1[idxPopX<0, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[1];
                g_x1[idxPopX<1, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[7];
                g_x1[idxPopX<2, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[9];
                g_x1[idxPopX<3, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[13];
                g_x1[idxPopX<4, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[15];
            }

            if (INTERFACE_BC_SOUTH(y))
            { // s
                g_y0[idxPopY<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[4];
                g_y0[idxPopY<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[8];
                g_y0[idxPopY<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[12];
                g_y0[idxPopY<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[13];
                g_y0[idxPopY<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[18];
            }
            if (INTERFACE_BC_NORTH(y))
            { // n
                g_y1[idxPopY<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[3];
                g_y1[idxPopY<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[7];
                g_y1[idxPopY<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[11];
                g_y1[idxPopY<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[14];
                g_y1[idxPopY<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[17];
            }

            if (INTERFACE_BC_BACK(z))
            { // b
                g_z0[idxPopZ<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[6];
                g_z0[idxPopZ<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[10];
                g_z0[idxPopZ<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[12];
                g_z0[idxPopZ<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[15];
                g_z0[idxPopZ<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[17];
            }
            if (INTERFACE_BC_FRONT(z))
            {
                g_z1[idxPopZ<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[5];
                g_z1[idxPopZ<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[9];
                g_z1[idxPopZ<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[11];
                g_z1[idxPopZ<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[16];
                g_z1[idxPopZ<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[18];
            }
        }
    }
}