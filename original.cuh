#ifndef ORIGINAL_CUH
#define ORIGINAL_CUH

// #include "./../var.h"

#ifndef __INTERFACE_BC_H
#define __INTERFACE_BC_H

#define INTERFACE_BC_WEST_PERIO (threadIdx.x == 0)
#define INTERFACE_BC_WEST_BLOCK (threadIdx.x == 0 && x != 0)
#define INTERFACE_BC_EAST_PERIO (threadIdx.x == (BLOCK_NX - 1))
#define INTERFACE_BC_EAST_BLOCK (threadIdx.x == (BLOCK_NX - 1) && x != (NX - 1))

#define INTERFACE_BC_SOUTH_PERIO (threadIdx.y == 0)
#define INTERFACE_BC_SOUTH_BLOCK (threadIdx.y == 0 && y != 0)
#define INTERFACE_BC_NORTH_PERIO (threadIdx.y == (BLOCK_NY - 1))
#define INTERFACE_BC_NORTH_BLOCK (threadIdx.y == (BLOCK_NY - 1) && y != (NY - 1))

#define INTERFACE_BC_BACK_PERIO (threadIdx.z == 0)
#define INTERFACE_BC_BACK_BLOCK (threadIdx.z == 0 && z != 0)
#define INTERFACE_BC_FRONT_PERIO (threadIdx.z == (BLOCK_NZ - 1))
#define INTERFACE_BC_FRONT_BLOCK (threadIdx.z == (BLOCK_NZ - 1) && z != (NZ - 1))

#define INTERFACE_BC_WEST INTERFACE_BC_WEST_BLOCK
#define INTERFACE_BC_EAST INTERFACE_BC_EAST_BLOCK

#define INTERFACE_BC_SOUTH INTERFACE_BC_SOUTH_BLOCK
#define INTERFACE_BC_NORTH INTERFACE_BC_NORTH_BLOCK

#define INTERFACE_BC_FRONT INTERFACE_BC_FRONT_BLOCK
#define INTERFACE_BC_BACK INTERFACE_BC_BACK_BLOCK

#endif

[[nodiscard]] inline consteval auto MAX_THREADS_PER_BLOCK() noexcept { return 512; }
[[nodiscard]] inline consteval auto MIN_BLOCKS_PER_MP() noexcept { return 8; }

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

template <const label_t pop>
__host__ __device__ label_t __forceinline__ idxPopBlock(const label_t tx, const label_t ty, const label_t tz)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (pop)));
}

template <const label_t pop>
__device__ label_t __forceinline__ idxPopX(
    const label_t ty,
    const label_t tz,
    const label_t bx,
    const label_t by,
    const label_t bz)
{
    return ty + BLOCK_NY * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

template <const label_t pop>
__device__ label_t __forceinline__ idxPopY(
    const label_t tx,
    const label_t tz,
    const label_t bx,
    const label_t by,
    const label_t bz)
{
    return tx + BLOCK_NX * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

template <const label_t pop>
__device__ label_t __forceinline__ idxPopZ(
    const label_t tx,
    const label_t ty,
    const label_t bx,
    const label_t by,
    const label_t bz)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__host__ __device__ scalar_t __forceinline__ gpu_f_eq(const scalar_t rhow, const scalar_t uc3, const scalar_t p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) ->
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (static_cast<scalar_t>(1.0) + uc3 * static_cast<scalar_t>(0.5))));
}

template <const label_t mom>
__device__ __host__ [[nodiscard]] inline label_t idxMom__(
    const label_t tx, const label_t ty, const label_t tz,
    const label_t bx, const label_t by, const label_t bz)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (mom + NUMBER_MOMENTS * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz))))));
}

typedef struct ghostData
{
    scalar_t *X_0;
    scalar_t *X_1;
    scalar_t *Y_0;
    scalar_t *Y_1;
    scalar_t *Z_0;
    scalar_t *Z_1;
} GhostData;

typedef struct ghostInterfaceData
{
    ghostData fGhost;
    ghostData gGhost;
    ghostData h_fGhost;
} GhostInterfaceData;

__host__ void interfaceMalloc(
    ghostInterfaceData &ghostInterface)
{
    // cudaMalloc((void **)&(ghostInterface.fGhost.X_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    // cudaMalloc((void **)&(ghostInterface.fGhost.X_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    // cudaMalloc((void **)&(ghostInterface.fGhost.Y_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    // cudaMalloc((void **)&(ghostInterface.fGhost.Y_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    // cudaMalloc((void **)&(ghostInterface.fGhost.Z_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);
    // cudaMalloc((void **)&(ghostInterface.fGhost.Z_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);

    cudaMalloc(&(ghostInterface.fGhost.X_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc(&(ghostInterface.fGhost.X_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc(&(ghostInterface.fGhost.Y_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc(&(ghostInterface.fGhost.Y_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc(&(ghostInterface.fGhost.Z_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);
    cudaMalloc(&(ghostInterface.fGhost.Z_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);

    // std::cout << sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF << " bytes allocated to fGhost.X_0" << std::endl;

    // cudaMalloc((void **)&(ghostInterface.gGhost.X_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    // cudaMalloc((void **)&(ghostInterface.gGhost.X_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    // cudaMalloc((void **)&(ghostInterface.gGhost.Y_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    // cudaMalloc((void **)&(ghostInterface.gGhost.Y_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    // cudaMalloc((void **)&(ghostInterface.gGhost.Z_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);
    // cudaMalloc((void **)&(ghostInterface.gGhost.Z_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);

    cudaMalloc(&(ghostInterface.gGhost.X_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc(&(ghostInterface.gGhost.X_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc(&(ghostInterface.gGhost.Y_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc(&(ghostInterface.gGhost.Y_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc(&(ghostInterface.gGhost.Z_0), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);
    cudaMalloc(&(ghostInterface.gGhost.Z_1), sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF);
}

__host__ void allocateHostMemory(
    scalar_t **h_fMom,
    scalar_t **rho,
    scalar_t **ux,
    scalar_t **uy,
    scalar_t **uz)
{
    // checkCudaErrors(cudaMallocHost((void **)h_fMom, MEM_SIZE_MOM));
    // checkCudaErrors(cudaMallocHost((void **)rho, MEM_SIZE_SCALAR));
    // checkCudaErrors(cudaMallocHost((void **)ux, MEM_SIZE_SCALAR));
    // checkCudaErrors(cudaMallocHost((void **)uy, MEM_SIZE_SCALAR));
    // checkCudaErrors(cudaMallocHost((void **)uz, MEM_SIZE_SCALAR));

    checkCudaErrors(cudaMallocHost(h_fMom, MEM_SIZE_MOM));
    checkCudaErrors(cudaMallocHost(rho, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost(ux, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost(uy, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost(uz, MEM_SIZE_SCALAR));
}

__host__ void allocateDeviceMemory(
    scalar_t **d_fMom,
    unsigned int **dNodeType,
    GhostInterfaceData *ghostInterface)
{
    // cudaMalloc((void**)d_fMom, MEM_SIZE_MOM);
    cudaMalloc(d_fMom, MEM_SIZE_MOM);
    // std::cout << "Allocated device moments to address " << d_fMom << std::endl;
    // cudaMalloc((void **)dNodeType, sizeof(int) * NUMBER_LBM_NODES);
    cudaMalloc(dNodeType, sizeof(int) * NUMBER_LBM_NODES);
    interfaceMalloc(*ghostInterface);
}

__host__ void interfaceCudaMemcpy(
    [[maybe_unused]] GhostInterfaceData &ghostInterface,
    ghostData &dst,
    const ghostData &src,
    cudaMemcpyKind kind)
{
    struct MemcpyPair
    {
        scalar_t *dst;
        const scalar_t *src;
        label_t size;
    };

    MemcpyPair memcpyPairs[] = {
        {dst.X_0, src.X_0, sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF},
        {dst.X_1, src.X_1, sizeof(scalar_t) * NUMBER_GHOST_FACE_YZ * QF},
        {dst.Y_0, src.Y_0, sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF},
        {dst.Y_1, src.Y_1, sizeof(scalar_t) * NUMBER_GHOST_FACE_XZ * QF},
        {dst.Z_0, src.Z_0, sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF},
        {dst.Z_1, src.Z_1, sizeof(scalar_t) * NUMBER_GHOST_FACE_XY * QF}};

    checkCudaErrors(cudaDeviceSynchronize());
    for (const auto &pair : memcpyPairs)
    {
        checkCudaErrors(cudaMemcpy(pair.dst, pair.src, pair.size, kind));
    }
}

__host__ __device__ label_t __forceinline__ idxScalarGlobal(unsigned int x, unsigned int y, unsigned int z)
{
    // return NX * (NY * z + y) + x;
    return x + NX * (y + NY * (z));
}

__launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuInitialization_mom(
    scalar_t *fMom,
    scalar_t *randomNumbers)
{
    label_t x = threadIdx.x + blockDim.x * blockIdx.x;
    label_t y = threadIdx.y + blockDim.y * blockIdx.y;
    label_t z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    label_t index = idxScalarGlobal(x, y, z);
    // size_t index = 0;

    // first moments
    scalar_t rho, ux, uy, uz;

#include "CASE_FLOW_INITIALIZATION.cuh"

    // zeroth moment
    fMom[idxMom__<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = rho - RHO_0;
    fMom[idxMom__<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE * ux;
    fMom[idxMom__<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE * uy;
    fMom[idxMom__<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE * uz;

    // second moments
    // define equilibrium populations
    scalar_t pop[Q];
    for (label_t i = 0; i < Q; i++)
    {
        pop[i] = gpu_f_eq(w[i] * RHO_0,
                          3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                          1 - 1.5 * (ux * ux + uy * uy + uz * uz));
    }

    scalar_t invRho = 1.0 / rho;
    scalar_t pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
    scalar_t pixy = ((pop[7] + pop[8]) - (pop[13] + pop[14])) * invRho;
    scalar_t pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
    scalar_t piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
    scalar_t piyz = ((pop[11] + pop[12]) - (pop[17] + pop[18])) * invRho;
    scalar_t pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;

    fMom[idxMom__<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * pixx;
    fMom[idxMom__<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * pixy;
    fMom[idxMom__<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * pixz;
    fMom[idxMom__<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * piyy;
    fMom[idxMom__<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE * piyz;
    fMom[idxMom__<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE * pizz;
}

// #include "velocitySet/velocitySet.cuh"

__device__ static inline void reconstruct(
    scalar_t pop[19],
    const scalar_t rhoVar,
    const scalar_t ux_t30,
    const scalar_t uy_t30,
    const scalar_t uz_t30,
    const scalar_t m_xx_t45,
    const scalar_t m_xy_t90,
    const scalar_t m_xz_t90,
    const scalar_t m_yy_t45,
    const scalar_t m_yz_t90,
    const scalar_t m_zz_t45) noexcept
{
    const scalar_t pics2 = 1.0 - cs2 * (m_xx_t45 + m_yy_t45 + m_zz_t45);

    const scalar_t multiplyTerm_0 = rhoVar * W0;
    pop[0] = multiplyTerm_0 * pics2;

    const scalar_t multiplyTerm_1 = rhoVar * W1;
    pop[1] = multiplyTerm_1 * (pics2 + ux_t30 + m_xx_t45);
    pop[2] = multiplyTerm_1 * (pics2 - ux_t30 + m_xx_t45);
    pop[3] = multiplyTerm_1 * (pics2 + uy_t30 + m_yy_t45);
    pop[4] = multiplyTerm_1 * (pics2 - uy_t30 + m_yy_t45);
    pop[5] = multiplyTerm_1 * (pics2 + uz_t30 + m_zz_t45);
    pop[6] = multiplyTerm_1 * (pics2 - uz_t30 + m_zz_t45);

    const scalar_t multiplyTerm_2 = rhoVar * W2;
    pop[7] = multiplyTerm_2 * (pics2 + ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90);
    pop[8] = multiplyTerm_2 * (pics2 - ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90);
    pop[9] = multiplyTerm_2 * (pics2 + ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90);
    pop[10] = multiplyTerm_2 * (pics2 - ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90);
    pop[11] = multiplyTerm_2 * (pics2 + uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90);
    pop[12] = multiplyTerm_2 * (pics2 - uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90);
    pop[13] = multiplyTerm_2 * (pics2 + ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
    pop[14] = multiplyTerm_2 * (pics2 - ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
    pop[15] = multiplyTerm_2 * (pics2 + ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
    pop[16] = multiplyTerm_2 * (pics2 - ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
    pop[17] = multiplyTerm_2 * (pics2 + uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);
    pop[18] = multiplyTerm_2 * (pics2 - uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);
}

__launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuInitialization_pop(
    scalar_t *fMom, ghostInterfaceData ghostInterface)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    const label_t index = idxScalarGlobal(x, y, z);
    // zeroth moment

    scalar_t rhoVar = RHO_0 + fMom[idxMom__<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t ux_t30 = fMom[idxMom__<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t uy_t30 = fMom[idxMom__<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t uz_t30 = fMom[idxMom__<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_xx_t45 = fMom[idxMom__<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_xy_t90 = fMom[idxMom__<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_xz_t90 = fMom[idxMom__<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_yy_t45 = fMom[idxMom__<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_yz_t90 = fMom[idxMom__<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_zz_t45 = fMom[idxMom__<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

    scalar_t pop[Q];
    // scalar_t multiplyTerm;
    // scalar_t pics2;
    // #include "COLREC_RECONSTRUCTIONS.cuh"
    reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

    // thread xyz
    const label_t tx = threadIdx.x;
    const label_t ty = threadIdx.y;
    const label_t tz = threadIdx.z;

    // block xyz
    const label_t bx = blockIdx.x;
    const label_t by = blockIdx.y;
    const label_t bz = blockIdx.z;

    if (threadIdx.x == 0)
    { // w
        ghostInterface.fGhost.X_0[idxPopX<0>(ty, tz, bx, by, bz)] = pop[2];
        ghostInterface.fGhost.X_0[idxPopX<1>(ty, tz, bx, by, bz)] = pop[8];
        ghostInterface.fGhost.X_0[idxPopX<2>(ty, tz, bx, by, bz)] = pop[10];
        ghostInterface.fGhost.X_0[idxPopX<3>(ty, tz, bx, by, bz)] = pop[14];
        ghostInterface.fGhost.X_0[idxPopX<4>(ty, tz, bx, by, bz)] = pop[16];
    }
    else if (threadIdx.x == (BLOCK_NX - 1))
    {
        ghostInterface.fGhost.X_1[idxPopX<0>(ty, tz, bx, by, bz)] = pop[1];
        ghostInterface.fGhost.X_1[idxPopX<1>(ty, tz, bx, by, bz)] = pop[7];
        ghostInterface.fGhost.X_1[idxPopX<2>(ty, tz, bx, by, bz)] = pop[9];
        ghostInterface.fGhost.X_1[idxPopX<3>(ty, tz, bx, by, bz)] = pop[13];
        ghostInterface.fGhost.X_1[idxPopX<4>(ty, tz, bx, by, bz)] = pop[15];
    }

    if (threadIdx.y == 0)
    { // s
        ghostInterface.fGhost.Y_0[idxPopY<0>(tx, tz, bx, by, bz)] = pop[4];
        ghostInterface.fGhost.Y_0[idxPopY<1>(tx, tz, bx, by, bz)] = pop[8];
        ghostInterface.fGhost.Y_0[idxPopY<2>(tx, tz, bx, by, bz)] = pop[12];
        ghostInterface.fGhost.Y_0[idxPopY<3>(tx, tz, bx, by, bz)] = pop[13];
        ghostInterface.fGhost.Y_0[idxPopY<4>(tx, tz, bx, by, bz)] = pop[18];
    }
    else if (threadIdx.y == (BLOCK_NY - 1))
    {
        ghostInterface.fGhost.Y_1[idxPopY<0>(tx, tz, bx, by, bz)] = pop[3];
        ghostInterface.fGhost.Y_1[idxPopY<1>(tx, tz, bx, by, bz)] = pop[7];
        ghostInterface.fGhost.Y_1[idxPopY<2>(tx, tz, bx, by, bz)] = pop[11];
        ghostInterface.fGhost.Y_1[idxPopY<3>(tx, tz, bx, by, bz)] = pop[14];
        ghostInterface.fGhost.Y_1[idxPopY<4>(tx, tz, bx, by, bz)] = pop[17];
    }

    if (threadIdx.z == 0)
    { // b
        ghostInterface.fGhost.Z_0[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[6];
        ghostInterface.fGhost.Z_0[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[10];
        ghostInterface.fGhost.Z_0[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[12];
        ghostInterface.fGhost.Z_0[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[15];
        ghostInterface.fGhost.Z_0[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[17];
    }
    else if (threadIdx.z == (BLOCK_NZ - 1))
    {
        ghostInterface.fGhost.Z_1[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[5];
        ghostInterface.fGhost.Z_1[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[9];
        ghostInterface.fGhost.Z_1[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[11];
        ghostInterface.fGhost.Z_1[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[16];
        ghostInterface.fGhost.Z_1[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[18];
    }
}

__host__ __device__ label_t __forceinline__ idxScalarBlock(
    const label_t tx,
    const label_t ty,
    const label_t tz,
    const label_t bx,
    const label_t by,
    const label_t bz)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz)))));
}

__host__ void hostInitialization_nodeType(unsigned int *hNodeType)
{
    unsigned int nodeType;

    for (label_t x = 0; x < NX; x++)
    {
        for (label_t y = 0; y < NY; y++)
        {
            for (label_t z = 0; z < NZ_TOTAL; z++)
            {
#include "CASE_BC_INIT.cuh"
                if (nodeType != BULK)
                    hNodeType[idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)] = (unsigned int)nodeType;
            }
        }
    }

    printf("boundary condition done\n");
}

__host__ void initializeDomain(
    GhostInterfaceData &ghostInterface,
    scalar_t *&d_fMom, scalar_t *&h_fMom,
    unsigned int *&hNodeType, unsigned int *&dNodeType, scalar_t **&randomNumbers,
    dim3 gridBlock, dim3 threadBlock)
{

    gpuInitialization_mom<<<gridBlock, threadBlock, 0, 0>>>(d_fMom, randomNumbers[0]);

    gpuInitialization_pop<<<gridBlock, threadBlock, 0, 0>>>(d_fMom, ghostInterface);

    // Node type initialization
    // checkCudaErrors(cudaMallocHost((void **)&hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES));

    hostInitialization_nodeType(hNodeType);
    checkCudaErrors(cudaMemcpy(dNodeType, hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    // Interface population initialization
    interfaceCudaMemcpy(
        ghostInterface,
        ghostInterface.gGhost,
        ghostInterface.fGhost,
        cudaMemcpyDeviceToDevice);

    // Synchronize after all initializations
    checkCudaErrors(cudaDeviceSynchronize());

    // Synchronize and transfer data back to host if needed
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(scalar_t) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Mom copy to host \n");
    if (console_flush)
        fflush(stdout);
}

__device__ static inline void calculateBoundaryMoments(
    const scalar_t pop[19],
    scalar_t *const rhoVar,
    scalar_t *const ux_t30,
    scalar_t *const uy_t30,
    scalar_t *const uz_t30,
    scalar_t *const m_xx_t45,
    scalar_t *const m_xy_t90,
    scalar_t *const m_xz_t90,
    scalar_t *const m_yy_t45,
    scalar_t *const m_yz_t90,
    scalar_t *const m_zz_t45,
    const unsigned int nodeType) noexcept
{
    scalar_t rho_I;
    scalar_t inv_rho_I;

    scalar_t m_xx_I;
    scalar_t m_xy_I;
    scalar_t m_xz_I;
    scalar_t m_yy_I;
    scalar_t m_yz_I;
    scalar_t m_zz_I;

    scalar_t rho;
    // scalar_t inv_rho;

    constexpr scalar_t omegaVar = OMEGA;

    switch (nodeType)
    {
    case SOUTH_WEST_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12]);
        m_xy_I = inv_rho_I * (pop[8]);
        m_xz_I = inv_rho_I * (pop[10]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12]);
        m_yz_I = inv_rho_I * (pop[12]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12]);

        rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = -(14 * m_xy_I - 14 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 2 * m_yz_I - 2 * m_zz_I - 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = -(14 * m_xx_I - 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = -(14 * m_xy_I - 2 * m_xx_I + 2 * m_xz_I - 14 * m_yy_I + 14 * m_yz_I - 2 * m_zz_I + 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I - 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = -(2 * m_xx_I - 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = -(2 * m_xy_I - 2 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 14 * m_yz_I - 14 * m_zz_I + 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 21 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case SOUTH_WEST_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[8]);
        m_xz_I = inv_rho_I * (-pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (-pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

        rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = (14 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = (14 * m_xx_I - 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = (2 * m_xx_I - 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = (2 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = (2 * m_xx_I - 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I + 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case NORTH_WEST_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (-pop[14]);
        m_xz_I = inv_rho_I * (pop[10]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (-pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);

        rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = -(14 * m_xx_I + 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I - 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = (2 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case NORTH_WEST_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);
        m_xy_I = inv_rho_I * (-pop[14]);
        m_xz_I = inv_rho_I * (-pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16]);
        m_yz_I = inv_rho_I * (pop[11]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);

        rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = (14 * m_xx_I + 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = -(2 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case SOUTH_EAST_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);
        m_xy_I = inv_rho_I * (-pop[13]);
        m_xz_I = inv_rho_I * (-pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15]);
        m_yz_I = inv_rho_I * (pop[12]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);

        rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = (14 * m_xx_I + 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = -(2 * m_xx_I + 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I + 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case SOUTH_EAST_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (-pop[13]);
        m_xz_I = inv_rho_I * (pop[9]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (-pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);

        rho = (12 * (rho_I + m_xx_I * rho_I + 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I - 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = (14 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = (14 * m_xx_I + 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I + 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = -(14 * m_xx_I + 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = (2 * m_xx_I + 14 * m_xy_I - 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = (2 * m_xx_I + 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I - 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = (2 * m_xx_I + 2 * m_xy_I - 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I - 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I + 2 * m_xy_I - 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I - 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case NORTH_EAST_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (pop[7]);
        m_xz_I = inv_rho_I * (-pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (-pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

        rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I + 2 * m_xz_I * rho_I + m_yy_I * rho_I + 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I - 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I - 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = (14 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 2 * m_yz_I + 2 * m_zz_I + 21 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = (14 * m_xx_I - 14 * m_xy_I + 50 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I - 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = (2 * m_xx_I - 14 * m_xy_I + 2 * m_xz_I + 14 * m_yy_I + 14 * m_yz_I + 2 * m_zz_I - 9 * omegaVar * m_xx_I - 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I + 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 9 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = (2 * m_xx_I - 14 * m_xy_I + 14 * m_xz_I + 14 * m_yy_I + 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = (2 * m_xx_I - 2 * m_xy_I + 14 * m_xz_I + 2 * m_yy_I + 14 * m_yz_I + 14 * m_zz_I - 9 * omegaVar * m_xx_I + 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I - 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 21 * omegaVar * m_zz_I + 4) / (18 * (m_xx_I - 2 * m_xy_I + 2 * m_xz_I + m_yy_I + 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I - 2 * omegaVar * m_xz_I - omegaVar * m_yy_I - 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case NORTH_EAST_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11]);
        m_xy_I = inv_rho_I * (pop[7]);
        m_xz_I = inv_rho_I * (pop[9]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11]);
        m_yz_I = inv_rho_I * (pop[11]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11]);

        rho = (12 * (rho_I + m_xx_I * rho_I - 2 * m_xy_I * rho_I - 2 * m_xz_I * rho_I + m_yy_I * rho_I - 2 * m_yz_I * rho_I + m_zz_I * rho_I - omegaVar * m_xx_I * rho_I + 2 * omegaVar * m_xy_I * rho_I + 2 * omegaVar * m_xz_I * rho_I - omegaVar * m_yy_I * rho_I + 2 * omegaVar * m_yz_I * rho_I - omegaVar * m_zz_I * rho_I)) / (5 * omegaVar + 2);

        *m_xx_t45 = -(14 * m_xy_I - 14 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 2 * m_yz_I - 2 * m_zz_I - 21 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I - 23 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xy_t90 = -(14 * m_xx_I - 50 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 14 * m_yz_I + 2 * m_zz_I + 7 * omegaVar * m_xx_I - 69 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I - 23 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_xz_t90 = -(14 * m_xx_I - 14 * m_xy_I - 50 * m_xz_I + 2 * m_yy_I - 14 * m_yz_I + 14 * m_zz_I + 7 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I - 69 * omegaVar * m_xz_I - 23 * omegaVar * m_yy_I + 21 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yy_t45 = -(14 * m_xy_I - 2 * m_xx_I + 2 * m_xz_I - 14 * m_yy_I + 14 * m_yz_I - 2 * m_zz_I + 9 * omegaVar * m_xx_I + 7 * omegaVar * m_xy_I - 23 * omegaVar * m_xz_I - 21 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I + 9 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_yz_t90 = -(2 * m_xx_I - 14 * m_xy_I - 14 * m_xz_I + 14 * m_yy_I - 50 * m_yz_I + 14 * m_zz_I - 23 * omegaVar * m_xx_I + 21 * omegaVar * m_xy_I + 21 * omegaVar * m_xz_I + 7 * omegaVar * m_yy_I - 69 * omegaVar * m_yz_I + 7 * omegaVar * m_zz_I + 8) / (36 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));
        *m_zz_t45 = -(2 * m_xy_I - 2 * m_xx_I + 14 * m_xz_I - 2 * m_yy_I + 14 * m_yz_I - 14 * m_zz_I + 9 * omegaVar * m_xx_I - 23 * omegaVar * m_xy_I + 7 * omegaVar * m_xz_I + 9 * omegaVar * m_yy_I + 7 * omegaVar * m_yz_I - 21 * omegaVar * m_zz_I - 4) / (18 * (m_xx_I - 2 * m_xy_I - 2 * m_xz_I + m_yy_I - 2 * m_yz_I + m_zz_I - omegaVar * m_xx_I + 2 * omegaVar * m_xy_I + 2 * omegaVar * m_xz_I - omegaVar * m_yy_I + 2 * omegaVar * m_yz_I - omegaVar * m_zz_I + 1));

        *rhoVar = rho;

        break;
    case SOUTH_WEST:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[8]);
        m_xz_I = inv_rho_I * (pop[10] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[12] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I - 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (2412 * m_xy_I - 504 * m_xx_I - 504 * m_yy_I + 216 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I + 79 * omegaVar * m_yy_I + 34 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (216 * m_xx_I - 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = -(72 * m_xx_I - 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case NORTH_WEST:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (-pop[14]);
        m_xz_I = inv_rho_I * (pop[10] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (pop[11] - pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I + 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (504 * m_xx_I + 2412 * m_xy_I + 504 * m_yy_I - 216 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I - 79 * omegaVar * m_yy_I - 34 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (216 * m_xx_I + 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = -(72 * m_xx_I + 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case SOUTH_EAST:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (-pop[13]);
        m_xz_I = inv_rho_I * (pop[9] - pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[12] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[18]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I + 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (504 * m_xx_I + 2412 * m_xy_I + 504 * m_yy_I - 216 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I - 79 * omegaVar * m_yy_I - 34 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (216 * m_xx_I + 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = -(72 * m_xx_I + 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case NORTH_EAST:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (pop[7]);
        m_xz_I = inv_rho_I * (pop[9] - pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (pop[11] - pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xy_I * rho_I + 24 * m_yy_I * rho_I - 6 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xy_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 6 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I - 1008 * m_xy_I + 216 * m_yy_I - 144 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I - 191 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (2412 * m_xy_I - 504 * m_xx_I - 504 * m_yy_I + 216 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xy_I + 79 * omegaVar * m_yy_I + 34 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (216 * m_xx_I - 1008 * m_xy_I + 936 * m_yy_I - 144 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xy_I + 239 * omegaVar * m_yy_I - 6 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = -(72 * m_xx_I - 216 * m_xy_I + 72 * m_yy_I - 288 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xy_I + 3 * omegaVar * m_yy_I - 162 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xy_I + 24 * m_yy_I - 6 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xy_I - 24 * omegaVar * m_yy_I + 6 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case WEST_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (pop[8] - pop[14]);
        m_xz_I = inv_rho_I * (pop[10]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (pop[12] - pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[17]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (2412 * m_xz_I - 504 * m_xx_I + 216 * m_yy_I - 504 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I + 34 * omegaVar * m_yy_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = -(72 * m_xx_I - 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case WEST_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[8] - pop[14]);
        m_xz_I = inv_rho_I * (-pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[11] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (504 * m_xx_I + 2412 * m_xz_I - 216 * m_yy_I + 504 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I - 34 * omegaVar * m_yy_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = -(72 * m_xx_I + 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case EAST_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (pop[7] - pop[13]);
        m_xz_I = inv_rho_I * (-pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (pop[12] - pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I + 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I - 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (504 * m_xx_I + 2412 * m_xz_I - 216 * m_yy_I + 504 * m_zz_I - 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I - 34 * omegaVar * m_yy_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = -(72 * m_xx_I + 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I + 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_xx_I + 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I - 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I + 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I - 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case EAST_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[7] - pop[13]);
        m_xz_I = inv_rho_I * (pop[9]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[11] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[18]);

        rho = (36 * (23 * rho_I + 24 * m_xx_I * rho_I - 57 * m_xz_I * rho_I - 6 * m_yy_I * rho_I + 24 * m_zz_I * rho_I - 24 * omegaVar * m_xx_I * rho_I + 57 * omegaVar * m_xz_I * rho_I + 6 * omegaVar * m_yy_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = (936 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 216 * m_zz_I + 239 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (2412 * m_xz_I - 504 * m_xx_I + 216 * m_yy_I - 504 * m_zz_I + 79 * omegaVar * m_xx_I + 538 * omegaVar * m_xz_I + 34 * omegaVar * m_yy_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = -(72 * m_xx_I - 216 * m_xz_I - 288 * m_yy_I + 72 * m_zz_I + 3 * omegaVar * m_xx_I - 34 * omegaVar * m_xz_I - 162 * omegaVar * m_yy_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (5 * m_yz_I * (43 * omegaVar + 72)) / (18 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_xx_I - 1008 * m_xz_I - 144 * m_yy_I + 936 * m_zz_I - 191 * omegaVar * m_xx_I + 158 * omegaVar * m_xz_I - 6 * omegaVar * m_yy_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_xx_I - 57 * m_xz_I - 6 * m_yy_I + 24 * m_zz_I - 24 * omegaVar * m_xx_I + 57 * omegaVar * m_xz_I + 6 * omegaVar * m_yy_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case SOUTH_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);
        m_xy_I = inv_rho_I * (pop[8] - pop[13]);
        m_xz_I = inv_rho_I * (pop[10] - pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15]);
        m_yz_I = inv_rho_I * (pop[12]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15]);

        rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I - 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I - 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I - 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (216 * m_xx_I - 504 * m_yy_I + 2412 * m_yz_I - 504 * m_zz_I + 34 * omegaVar * m_xx_I + 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case SOUTH_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[8] - pop[13]);
        m_xz_I = inv_rho_I * (pop[9] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (-pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

        rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I + 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I - 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I + 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I + 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (504 * m_yy_I - 216 * m_xx_I + 2412 * m_yz_I + 504 * m_zz_I - 34 * omegaVar * m_xx_I - 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case NORTH_BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (pop[7] - pop[14]);
        m_xz_I = inv_rho_I * (pop[10] - pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (-pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

        rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I + 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I - 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I + 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I + 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (504 * m_yy_I - 216 * m_xx_I + 2412 * m_yz_I + 504 * m_zz_I - 34 * omegaVar * m_xx_I - 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I - 79 * omegaVar * m_zz_I + 228) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I + 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I - 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I + 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I - 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case NORTH_FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);
        m_xy_I = inv_rho_I * (pop[7] - pop[14]);
        m_xz_I = inv_rho_I * (pop[9] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16]);
        m_yz_I = inv_rho_I * (pop[11]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16]);

        rho = (36 * (23 * rho_I - 6 * m_xx_I * rho_I + 24 * m_yy_I * rho_I - 57 * m_yz_I * rho_I + 24 * m_zz_I * rho_I + 6 * omegaVar * m_xx_I * rho_I - 24 * omegaVar * m_yy_I * rho_I + 57 * omegaVar * m_yz_I * rho_I - 24 * omegaVar * m_zz_I * rho_I)) / (5 * (43 * omegaVar + 72));

        *m_xx_t45 = -(72 * m_yy_I - 288 * m_xx_I - 216 * m_yz_I + 72 * m_zz_I - 162 * omegaVar * m_xx_I + 3 * omegaVar * m_yy_I - 34 * omegaVar * m_yz_I + 3 * omegaVar * m_zz_I + 24) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xy_t90 = (5 * m_xy_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_xz_t90 = (5 * m_xz_I * (43 * omegaVar + 72)) / (18 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yy_t45 = (936 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 216 * m_zz_I - 6 * omegaVar * m_xx_I + 239 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I - 191 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_yz_t90 = (216 * m_xx_I - 504 * m_yy_I + 2412 * m_yz_I - 504 * m_zz_I + 34 * omegaVar * m_xx_I + 79 * omegaVar * m_yy_I + 538 * omegaVar * m_yz_I + 79 * omegaVar * m_zz_I - 228) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));
        *m_zz_t45 = (216 * m_yy_I - 144 * m_xx_I - 1008 * m_yz_I + 936 * m_zz_I - 6 * omegaVar * m_xx_I - 191 * omegaVar * m_yy_I + 158 * omegaVar * m_yz_I + 239 * omegaVar * m_zz_I + 192) / (36 * (24 * m_yy_I - 6 * m_xx_I - 57 * m_yz_I + 24 * m_zz_I + 6 * omegaVar * m_xx_I - 24 * omegaVar * m_yy_I + 57 * omegaVar * m_yz_I - 24 * omegaVar * m_zz_I + 23));

        *rhoVar = rho;

        break;
    case WEST:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[8] - pop[14]);
        m_xz_I = inv_rho_I * (pop[10] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);

        rho = (3 * rho_I * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4)) / (omegaVar + 9);

        *m_xx_t45 = (15 * m_xx_I + 2) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_xy_t90 = (2 * m_xy_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_yy_t45 = (4 * (omegaVar + 9) * (10 * m_yy_I - m_zz_I)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_yz_t90 = (m_yz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_zz_t45 = -(4 * (m_yy_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));

        *rhoVar = rho;

        break;
    case EAST:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[7] - pop[13]);
        m_xz_I = inv_rho_I * (pop[9] - pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17] + (2.0 / 3.0) * pop[18]);

        rho = (3 * rho_I * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4)) / (omegaVar + 9);

        *m_xx_t45 = (15 * m_xx_I + 2) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_xy_t90 = (2 * m_xy_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_yy_t45 = (4 * (omegaVar + 9) * (10 * m_yy_I - m_zz_I)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_yz_t90 = (m_yz_I * (omegaVar + 9)) / (3 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));
        *m_zz_t45 = -(4 * (m_yy_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_xx_I - 3 * omegaVar * m_xx_I + 4));

        *rhoVar = rho;

        break;
    case SOUTH:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[8] - pop[13]);
        m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[12] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

        rho = (3 * rho_I * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4)) / (omegaVar + 9);

        *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_zz_I)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_xy_t90 = (2 * m_xy_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_xz_t90 = (m_xz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_yy_t45 = (15 * m_yy_I + 2) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_zz_t45 = -(4 * (m_xx_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));

        *rhoVar = rho;

        break;
    case NORTH:
        *ux_t30 = U_MAX;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (pop[7] - pop[14]);
        m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (pop[11] - pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[17]);

        rho = (3 * rho_I * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4)) / (omegaVar + 9);

        *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_zz_I)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_xy_t90 = (18 * m_xy_I - 4 * U_MAX + 2 * omegaVar * m_xy_I - 3 * U_MAX * m_yy_I + 3 * omegaVar * U_MAX * m_yy_I) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_xz_t90 = (m_xz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_yy_t45 = (15 * m_yy_I + 2) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));
        *m_zz_t45 = -(4 * (m_xx_I - 10 * m_zz_I) * (omegaVar + 9)) / (99 * (3 * m_yy_I - 3 * omegaVar * m_yy_I + 4));

        *rhoVar = rho;

        break;
    case BACK:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] - (1.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] - (1.0 / 3.0) * pop[17]);
        m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
        m_xz_I = inv_rho_I * (pop[10] - pop[15]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[6] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);
        m_yz_I = inv_rho_I * (pop[12] - pop[17]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[6] - (1.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[10] + (2.0 / 3.0) * pop[12] - (1.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[15] + (2.0 / 3.0) * pop[17]);

        rho = (3 * rho_I * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4)) / (omegaVar + 9);

        *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_yy_I)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_xy_t90 = (m_xy_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_yy_t45 = -(4 * (m_xx_I - 10 * m_yy_I) * (omegaVar + 9)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_zz_t45 = (15 * m_zz_I + 2) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));

        *rhoVar = rho;

        break;
    case FRONT:
        *ux_t30 = 0;
        *uy_t30 = 0;
        *uz_t30 = 0;

        rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] + (2.0 / 3.0) * pop[1] + (2.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] - (1.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] - (1.0 / 3.0) * pop[18]);
        m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
        m_xz_I = inv_rho_I * (pop[9] - pop[16]);
        m_yy_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] + (2.0 / 3.0) * pop[3] + (2.0 / 3.0) * pop[4] - (1.0 / 3.0) * pop[5] + (2.0 / 3.0) * pop[7] + (2.0 / 3.0) * pop[8] - (1.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] + (2.0 / 3.0) * pop[13] + (2.0 / 3.0) * pop[14] - (1.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);
        m_yz_I = inv_rho_I * (pop[11] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0 / 3.0) * pop[0] - (1.0 / 3.0) * pop[1] - (1.0 / 3.0) * pop[2] - (1.0 / 3.0) * pop[3] - (1.0 / 3.0) * pop[4] + (2.0 / 3.0) * pop[5] - (1.0 / 3.0) * pop[7] - (1.0 / 3.0) * pop[8] + (2.0 / 3.0) * pop[9] + (2.0 / 3.0) * pop[11] - (1.0 / 3.0) * pop[13] - (1.0 / 3.0) * pop[14] + (2.0 / 3.0) * pop[16] + (2.0 / 3.0) * pop[18]);

        rho = (3 * rho_I * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4)) / (omegaVar + 9);

        *m_xx_t45 = (4 * (omegaVar + 9) * (10 * m_xx_I - m_yy_I)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_xy_t90 = (m_xy_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_xz_t90 = (2 * m_xz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_yy_t45 = -(4 * (m_xx_I - 10 * m_yy_I) * (omegaVar + 9)) / (99 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_yz_t90 = (2 * m_yz_I * (omegaVar + 9)) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));
        *m_zz_t45 = (15 * m_zz_I + 2) / (3 * (3 * m_zz_I - 3 * omegaVar * m_zz_I + 4));

        *rhoVar = rho;

        break;
    }
}

__launch_bounds__(MAX_THREADS_PER_BLOCK(), MIN_BLOCKS_PER_MP()) __global__ void gpuMomCollisionStream(
    scalar_t *fMom,
    unsigned int *dNodeType,
    ghostInterfaceData ghostInterface)
{
    const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
    const label_t z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    scalar_t pop[Q];
    // scalar_t pics2;
    // scalar_t multiplyTerm;
    __shared__ scalar_t s_pop[BLOCK_LBM_SIZE * (Q - 1)];

    // Load moments from global memory

    // rho'
    const unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)
    {
        return;
    }
    scalar_t rhoVar = RHO_0 + fMom[idxMom__<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t ux_t30 = fMom[idxMom__<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t uy_t30 = fMom[idxMom__<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t uz_t30 = fMom[idxMom__<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_xx_t45 = fMom[idxMom__<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_xy_t90 = fMom[idxMom__<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_xz_t90 = fMom[idxMom__<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_yy_t45 = fMom[idxMom__<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_yz_t90 = fMom[idxMom__<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];
    scalar_t m_zz_t45 = fMom[idxMom__<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)];

    const scalar_t omegaVar = OMEGA;
    const scalar_t t_omegaVar = 1 - omegaVar;
    const scalar_t tt_omegaVar = 1 - omegaVar / 2;
    const scalar_t omegaVar_d2 = omegaVar / 2;
    const scalar_t tt_omega_t3 = tt_omegaVar * 3;

    // Local forces
    scalar_t L_Fx = FX;
    scalar_t L_Fy = FY;
    scalar_t L_Fz = FZ;

    // #include "COLREC_RECONSTRUCTIONS.cuh"

    reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

    const label_t xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const label_t xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const label_t yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const label_t ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const label_t zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const label_t zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;

    // save populations in shared memory
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

    // sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */

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

    const label_t tx = threadIdx.x;
    const label_t ty = threadIdx.y;
    const label_t tz = threadIdx.z;

    const label_t bx = blockIdx.x;
    const label_t by = blockIdx.y;
    const label_t bz = blockIdx.z;

    const label_t txp1 = (tx + 1 + BLOCK_NX) % BLOCK_NX;
    const label_t txm1 = (tx - 1 + BLOCK_NX) % BLOCK_NX;

    const label_t typ1 = (ty + 1 + BLOCK_NY) % BLOCK_NY;
    const label_t tym1 = (ty - 1 + BLOCK_NY) % BLOCK_NY;

    const label_t tzp1 = (tz + 1 + BLOCK_NZ) % BLOCK_NZ;
    const label_t tzm1 = (tz - 1 + BLOCK_NZ) % BLOCK_NZ;

    const label_t bxm1 = (bx - 1 + NUM_BLOCK_X) % NUM_BLOCK_X;
    const label_t bxp1 = (bx + 1 + NUM_BLOCK_X) % NUM_BLOCK_X;

    const label_t bym1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
    const label_t byp1 = (by + 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;

    const label_t bzm1 = (bz - 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;
    const label_t bzp1 = (bz + 1 + NUM_BLOCK_Z) % NUM_BLOCK_Z;

    /* load pop from global in cover nodes */

#include "popLoad.cuh"

    scalar_t invRho;
    if (nodeType != BULK)
    {
        calculateBoundaryMoments(
            pop,
            &rhoVar,
            &ux_t30,
            &uy_t30,
            &uz_t30,
            &m_xx_t45,
            &m_xy_t90,
            &m_xz_t90,
            &m_yy_t45,
            &m_yz_t90,
            &m_zz_t45,
            nodeType);

        invRho = 1.0 / rhoVar;
    }
    else
    {

        // calculate streaming moments

        // equation3
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        invRho = 1 / rhoVar;
        // equation4 + force correction
        ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]) + L_Fx / 2) * invRho;
        uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]) + L_Fy / 2) * invRho;
        uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]) + L_Fz / 2) * invRho;

        // equation5
        m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
        m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14]) * invRho;
        m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16]) * invRho;
        m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
        m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18]) * invRho;
        m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;
    }

    // multiply moments by as2 -- as4*0.5 -- as4 - add correction to m_alpha_beta
    ux_t30 = F_M_I_SCALE * ux_t30;
    uy_t30 = F_M_I_SCALE * uy_t30;
    uz_t30 = F_M_I_SCALE * uz_t30;

    m_xx_t45 = F_M_II_SCALE * (m_xx_t45);
    m_xy_t90 = F_M_IJ_SCALE * (m_xy_t90);
    m_xz_t90 = F_M_IJ_SCALE * (m_xz_t90);
    m_yy_t45 = F_M_II_SCALE * (m_yy_t45);
    m_yz_t90 = F_M_IJ_SCALE * (m_yz_t90);
    m_zz_t45 = F_M_II_SCALE * (m_zz_t45);

    // COLLIDE
#include "COLREC_COLLISION.cuh"

    // calculate post collision populations
    // #include "COLREC_RECONSTRUCTIONS.cuh"
    reconstruct(pop, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);

    /* write to global mom */

    fMom[idxMom__<M_RHO_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar - RHO_0;
    fMom[idxMom__<M_UX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30;
    fMom[idxMom__<M_UY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30;
    fMom[idxMom__<M_UZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30;
    fMom[idxMom__<M_MXX_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xx_t45;
    fMom[idxMom__<M_MXY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xy_t90;
    fMom[idxMom__<M_MXZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xz_t90;
    fMom[idxMom__<M_MYY_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yy_t45;
    fMom[idxMom__<M_MYZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yz_t90;
    fMom[idxMom__<M_MZZ_INDEX>(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = m_zz_t45;

#include "PopulationSave.cuh"
}

__host__ __device__ void interfaceSwap(scalar_t *&pt1, scalar_t *&pt2)
{
    scalar_t *temp = pt1;
    pt1 = pt2;
    pt2 = temp;
}

__host__ void swapGhostInterfaces(GhostInterfaceData &ghostInterface)
{
    // Synchronize device before performing swaps
    checkCudaErrors(cudaDeviceSynchronize());

    // Swap interface pointers for fGhost and gGhost
    interfaceSwap(ghostInterface.fGhost.X_0, ghostInterface.gGhost.X_0);
    interfaceSwap(ghostInterface.fGhost.X_1, ghostInterface.gGhost.X_1);
    interfaceSwap(ghostInterface.fGhost.Y_0, ghostInterface.gGhost.Y_0);
    interfaceSwap(ghostInterface.fGhost.Y_1, ghostInterface.gGhost.Y_1);
    interfaceSwap(ghostInterface.fGhost.Z_0, ghostInterface.gGhost.Z_0);
    interfaceSwap(ghostInterface.fGhost.Z_1, ghostInterface.gGhost.Z_1);
}

#endif