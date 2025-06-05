/**
Filename: ghostInterface.cuh
Contents: A handling the ghost interface on the device
This class is used to exchange the microscopic velocity components at the edge of a CUDA block
**/

#ifndef __MBLBM_GHOSTINTERFACE_CUH
#define __MBLBM_GHOSTINTERFACE_CUH

#include "ghostPtrs.cuh"

namespace mbLBM
{
    namespace host
    {
        class ghostInterface
        {
        public:
            /**
             * @brief Constructs a ghost interface from the device mesh and moments
             * @return A ghost interface object constructed from the moments
             * @param f The host copy of the scalar solution variable
             **/
            [[nodiscard]] ghostInterface(const host::moments &moms) noexcept
                : fGhost_(ghostPtrs(moms)),
                  gGhost_(ghostPtrs(moms)) // Changing this changes nothing
            {
#ifdef VERBOSE
                std::cout << "Allocated host ghostInterface object:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    nx = " << fGhost_.x0().nPoints() << " = " << moms.nPoints() << " * " << vSet::QF() << " / " << block::nx() << ";" << std::endl;
                std::cout << "    ny = " << fGhost_.y0().nPoints() << " = " << moms.nPoints() << " * " << vSet::QF() << " / " << block::ny() << ";" << std::endl;
                std::cout << "    nz = " << fGhost_.z0().nPoints() << " = " << moms.nPoints() << " * " << vSet::QF() << " / " << block::nz() << ";" << std::endl;
                std::cout << "}" << std::endl;
                std::cout << std::endl;
#endif
            };

            /**
             * @brief Default destructor for the ghostInterface class
             **/
            ~ghostInterface() {};

            /**
             * @brief Returns read-only access to each of the ghost pointers
             * @return An immutable reference to a ghostPtr object
             **/
            [[nodiscard]] inline constexpr const ghostPtrs &fGhost() const noexcept
            {
                return fGhost_;
            }
            [[nodiscard]] inline constexpr const ghostPtrs &gGhost() const noexcept
            {
                return gGhost_;
            }

        private:
            /**
             * @brief The underlying ghost pointers
             **/
            const ghostPtrs fGhost_;
            const ghostPtrs gGhost_;
        };
    }

    namespace device
    {
        class ghostInterface
        {
        public:
            /**
             * @brief Constructs 3 individual ghostPtr objects
             * @return A ghostInterface object constructed from the mesh
             * @param mesh The mesh used to define the amount of memory to allocate to the pointers
             **/
            [[nodiscard]] ghostInterface(const host::ghostInterface &interface) noexcept
                : fGhost_(ghostPtrs(interface.fGhost())),
                  gGhost_(ghostPtrs(interface.gGhost()))
            {
#ifdef VERBOSE
                std::cout << "Allocated device ghostInterface object" << std::endl;
#endif
                checkCudaErrors(cudaDeviceSynchronize());
            };

            /**
             * @brief Destructor for the ghostInterface class
             **/
            ~ghostInterface() noexcept
            {
#ifdef VERBOSE
                std::cout << "Freeing ghostInterface object" << std::endl;
#endif
            };

            /**
             * @brief Returns access to the ghost pointer objects
             * @return An immutable reference to the underlying ghostPtrs objects
             **/
            __host__ [[nodiscard]] inline constexpr const ghostPtrs &fGhost() const noexcept
            {
                return fGhost_;
            }
            __host__ [[nodiscard]] inline constexpr const ghostPtrs &gGhost() const noexcept
            {
                return gGhost_;
            }

            __host__ [[nodiscard]] inline constexpr ghostPtrs &fGhost() noexcept
            {
                return fGhost_;
            }
            __host__ [[nodiscard]] inline constexpr ghostPtrs &gGhost() noexcept
            {
                return gGhost_;
            }

            __host__ inline void swap() noexcept
            {
                checkCudaErrors(cudaDeviceSynchronize());

                interfaceSwap(fGhost_.x0Ref(), gGhost_.x0Ref());
                interfaceSwap(fGhost_.x1Ref(), gGhost_.x1Ref());
                interfaceSwap(fGhost_.y0Ref(), gGhost_.y0Ref());
                interfaceSwap(fGhost_.y1Ref(), gGhost_.y1Ref());
                interfaceSwap(fGhost_.z0Ref(), gGhost_.z0Ref());
                interfaceSwap(fGhost_.z1Ref(), gGhost_.z1Ref());
            }

            __device__ static inline void save(
                scalar_t ptrRestrict *const x0,
                scalar_t ptrRestrict *const x1,
                scalar_t ptrRestrict *const y0,
                scalar_t ptrRestrict *const y1,
                scalar_t ptrRestrict *const z0,
                scalar_t ptrRestrict *const z1,
                const scalar_t pop[19]) noexcept
            {
                const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
                const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
                const label_t z = threadIdx.z + blockDim.z * blockIdx.z;
                /* write to global pop */
                if (INTERFACE_BC_WEST(x))
                { // w
                    x0[idxPopX<0, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[2];
                    x0[idxPopX<1, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[8];
                    x0[idxPopX<2, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[10];
                    x0[idxPopX<3, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[14];
                    x0[idxPopX<4, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[16];
                }
                if (INTERFACE_BC_EAST(x))
                { // e
                    x1[idxPopX<0, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[1];
                    x1[idxPopX<1, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[7];
                    x1[idxPopX<2, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[9];
                    x1[idxPopX<3, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[13];
                    x1[idxPopX<4, VelocitySet::D3Q19::QF()>(threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[15];
                }

                if (INTERFACE_BC_SOUTH(y))
                { // s
                    y0[idxPopY<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[4];
                    y0[idxPopY<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[8];
                    y0[idxPopY<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[12];
                    y0[idxPopY<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[13];
                    y0[idxPopY<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[18];
                }
                if (INTERFACE_BC_NORTH(y))
                { // n
                    y1[idxPopY<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[3];
                    y1[idxPopY<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[7];
                    y1[idxPopY<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[11];
                    y1[idxPopY<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[14];
                    y1[idxPopY<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[17];
                }

                if (INTERFACE_BC_BACK(z))
                { // b
                    z0[idxPopZ<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[6];
                    z0[idxPopZ<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[10];
                    z0[idxPopZ<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[12];
                    z0[idxPopZ<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[15];
                    z0[idxPopZ<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[17];
                }
                if (INTERFACE_BC_FRONT(z))
                {
                    z1[idxPopZ<0, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[5];
                    z1[idxPopZ<1, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[9];
                    z1[idxPopZ<2, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[11];
                    z1[idxPopZ<3, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[16];
                    z1[idxPopZ<4, VelocitySet::D3Q19::QF()>(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z)] = pop[18];
                }
            }

        private:
            /**
             * @brief Lists of 6 unique pointers on each face of a CUDA block
             **/
            ghostPtrs fGhost_;
            ghostPtrs gGhost_;

            __host__ inline void interfaceSwap(scalar_t *ptrRestrict &pt1, scalar_t *ptrRestrict &pt2) noexcept
            {
                std::swap(pt1, pt2);
            }

            __device__ [[nodiscard]] static inline bool INTERFACE_BC_WEST(const label_t x) noexcept
            {
                return ((threadIdx.x == 0) && (x != 0));
            }
            __device__ [[nodiscard]] static inline bool INTERFACE_BC_EAST(const label_t x) noexcept
            {
                return ((threadIdx.x == (block::nx() - 1)) && (x != (d_nx - 1)));
            }

            __device__ [[nodiscard]] static inline bool INTERFACE_BC_SOUTH(const label_t y) noexcept
            {
                return ((threadIdx.y == 0) && (y != 0));
            }
            __device__ [[nodiscard]] static inline bool INTERFACE_BC_NORTH(const label_t y) noexcept
            {
                return ((threadIdx.y == (block::ny() - 1)) && (y != (d_ny - 1)));
            }

            __device__ [[nodiscard]] static inline bool INTERFACE_BC_BACK(const label_t z) noexcept
            {
                return ((threadIdx.z == 0) && (z != 0));
            }
            __device__ [[nodiscard]] static inline bool INTERFACE_BC_FRONT(const label_t z) noexcept
            {
                return ((threadIdx.z == (block::nz() - 1)) && (z != (d_nz - 1)));
            }
        };
    }
}

#endif