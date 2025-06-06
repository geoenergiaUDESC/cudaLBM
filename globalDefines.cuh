#ifndef GLOBALDEFINES_CUH
#define GLOBALDEFINES_CUH

namespace mbLBM
{
    constexpr const label_t NUMBER_MOMENTS = 10;

    constexpr const label_t BULK = (0b00000000000000000000000000000000);
    constexpr const label_t NORTH = (0b00000000000000000000000011001100);
    constexpr const label_t SOUTH = (0b00000000000000000000000000110011);
    constexpr const label_t WEST = (0b00000000000000000000000001010101);
    constexpr const label_t EAST = (0b00000000000000000000000010101010);
    constexpr const label_t FRONT = (0b00000000000000000000000011110000);
    constexpr const label_t BACK = (0b00000000000000000000000000001111);
    constexpr const label_t NORTH_WEST = (0b00000000000000000000000011011101);
    constexpr const label_t NORTH_EAST = (0b00000000000000000000000011101110);
    constexpr const label_t NORTH_FRONT = (0b00000000000000000000000011111100);
    constexpr const label_t NORTH_BACK = (0b00000000000000000000000011001111);
    constexpr const label_t SOUTH_WEST = (0b00000000000000000000000001110111);
    constexpr const label_t SOUTH_EAST = (0b00000000000000000000000010111011);
    constexpr const label_t SOUTH_FRONT = (0b00000000000000000000000011110011);
    constexpr const label_t SOUTH_BACK = (0b00000000000000000000000000111111);
    constexpr const label_t WEST_FRONT = (0b00000000000000000000000011110101);
    constexpr const label_t WEST_BACK = (0b00000000000000000000000001011111);
    constexpr const label_t EAST_FRONT = (0b00000000000000000000000011111010);
    constexpr const label_t EAST_BACK = (0b00000000000000000000000010101111);
    constexpr const label_t NORTH_WEST_FRONT = (0b00000000000000000000000011111101);
    constexpr const label_t NORTH_WEST_BACK = (0b00000000000000000000000011011111);
    constexpr const label_t NORTH_EAST_FRONT = (0b00000000000000000000000011111110);
    constexpr const label_t NORTH_EAST_BACK = (0b00000000000000000000000011101111);
    constexpr const label_t SOUTH_WEST_FRONT = (0b00000000000000000000000011110111);
    constexpr const label_t SOUTH_WEST_BACK = (0b00000000000000000000000001111111);
    constexpr const label_t SOUTH_EAST_FRONT = (0b00000000000000000000000011111011);
    constexpr const label_t SOUTH_EAST_BACK = (0b00000000000000000000000010111111);

    constexpr const scalar_t RE = 500;

    constexpr const label_t NX = 128;
    constexpr const label_t NY = 128;
    constexpr const label_t NZ = 128;
    constexpr const label_t NZ_TOTAL = NZ;

    constexpr const scalar_t FX = 0.0; // force in x
    constexpr const scalar_t FY = 0.0; // force in y
    constexpr const scalar_t FZ = 0.0; // force in z (flow direction in most cases)

    constexpr const scalar_t U_MAX = static_cast<scalar_t>(0.05);

    constexpr const scalar_t VISC = U_MAX * static_cast<scalar_t>(NX) / RE;
    constexpr const scalar_t TAU = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * VISC; // relaxation time
    constexpr const scalar_t MACH_NUMBER = U_MAX / static_cast<scalar_t>(0.57735026918962);

    constexpr const scalar_t OMEGA = static_cast<scalar_t>(1.0) / TAU;                                   // (tau)^-1
    constexpr const scalar_t OMEGAd2 = OMEGA / static_cast<scalar_t>(2.0);                               // OMEGA/2
    constexpr const scalar_t OMEGAd9 = OMEGA / static_cast<scalar_t>(9.0);                               // OMEGA/9
    constexpr const scalar_t T_OMEGA = static_cast<scalar_t>(1.0) - OMEGA;                               // 1-OMEGA
    constexpr const scalar_t TT_OMEGA = static_cast<scalar_t>(1.0) - static_cast<scalar_t>(0.5) * OMEGA; // 1.0 - OMEGA/2
    constexpr const scalar_t OMEGA_P1 = static_cast<scalar_t>(1.0) + OMEGA;                              // 1+ OMEGA
    constexpr const scalar_t TT_OMEGA_T3 = TT_OMEGA * static_cast<scalar_t>(3.0);                        // 3*(1-0.5*OMEGA)

    constexpr const scalar_t RHO_0 = 1.0;
    constexpr const scalar_t U_0_X = 0.0;
    constexpr const scalar_t U_0_Y = 0.0;
    constexpr const scalar_t U_0_Z = 0.0;

    constexpr const label_t M_RHO_INDEX = 0;
    constexpr const label_t M_UX_INDEX = 1;
    constexpr const label_t M_UY_INDEX = 2;
    constexpr const label_t M_UZ_INDEX = 3;
    constexpr const label_t M_MXX_INDEX = 4;
    constexpr const label_t M_MXY_INDEX = 5;
    constexpr const label_t M_MXZ_INDEX = 6;
    constexpr const label_t M_MYY_INDEX = 7;
    constexpr const label_t M_MYZ_INDEX = 8;
    constexpr const label_t M_MZZ_INDEX = 9;

    constexpr const label_t BLOCK_NX = 8;
    constexpr const label_t BLOCK_NY = 8;
    constexpr const label_t BLOCK_NZ = 8;
    constexpr const label_t BLOCK_LBM_SIZE = BLOCK_NX * BLOCK_NY * BLOCK_NZ;

    constexpr const label_t NUM_BLOCK_X = NX / BLOCK_NX;
    constexpr const label_t NUM_BLOCK_Y = NY / BLOCK_NY;
    constexpr const label_t NUM_BLOCK_Z = NZ / BLOCK_NZ;

    constexpr const label_t NUMBER_GHOST_FACE_XY = BLOCK_NX * BLOCK_NY * NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;
    constexpr const label_t NUMBER_GHOST_FACE_XZ = BLOCK_NX * BLOCK_NZ * NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;
    constexpr const label_t NUMBER_GHOST_FACE_YZ = BLOCK_NY * BLOCK_NZ * NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;

    constexpr const label_t Q = 19;
    constexpr const label_t QF = 5;

    constexpr const scalar_t W0 = static_cast<scalar_t>(1.0) / static_cast<scalar_t>(3.0);  // population 0 weight (0, 0, 0)
    constexpr const scalar_t W1 = static_cast<scalar_t>(1.0) / static_cast<scalar_t>(18.0); // adjacent populations (1, 0, 0)
    constexpr const scalar_t W2 = static_cast<scalar_t>(1.0) / static_cast<scalar_t>(36.0); // diagonal populations (1, 1, 0)

    constexpr const label_t NUM_BLOCK = NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;
    constexpr const label_t NUMBER_LBM_NODES = NUM_BLOCK * BLOCK_LBM_SIZE;

    constexpr const label_t NUMBER_LBM_POP_NODES = NX * NY * NZ;

    constexpr const label_t MEM_SIZE_SCALAR = sizeof(scalar_t) * NUMBER_LBM_NODES;
    constexpr const label_t MEM_SIZE_POP = sizeof(scalar_t) * NUMBER_LBM_POP_NODES * Q;
    constexpr const label_t MEM_SIZE_MOM = sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS;

    // populations velocities vector 0 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
    __device__ constexpr const scalar_t cx[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
    __device__ constexpr const scalar_t cy[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
    __device__ constexpr const scalar_t cz[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};

    __device__ constexpr const scalar_t w[Q] = {W0, W1, W1, W1, W1, W1, W1, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2};

    constexpr const scalar_t as2 = static_cast<scalar_t>(3.0);
    constexpr const scalar_t cs2 = static_cast<scalar_t>(1.0) / as2;

    constexpr const scalar_t F_M_0_SCALE = static_cast<scalar_t>(1.0);
    constexpr const scalar_t F_M_I_SCALE = as2;
    constexpr const scalar_t F_M_II_SCALE = as2 * as2 / static_cast<scalar_t>(2.0);
    constexpr const scalar_t F_M_IJ_SCALE = as2 * as2;

    constexpr const bool console_flush = false;
    constexpr const label_t GPU_INDEX = 1;

    constexpr const dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    constexpr const dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    constexpr const label_t INI_STEP = 0;
    constexpr const label_t N_STEPS = 1001;
}

#endif