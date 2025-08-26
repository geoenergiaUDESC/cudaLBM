/**
Filename: momentBasedD3Q19.cuh
Contents: Main kernel for the moment representation with the D3Q19 velocity set
**/

#ifndef __MBLBM_MOMENTBASEDD319_CUH
#define __MBLBM_MOMENTBASEDD319_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"
#include "../collision/collision.cuh"
#include "../blockHalo/blockHalo.cuh"
#include "../fileIO/fileIO.cuh"
#include "../runTimeIO/runTimeIO.cuh"

namespace LBM
{

    using VSet = VelocitySet::D3Q19;
    using Collision = secondOrder;

    //     template <class VSet>
    //     struct boundaryValue
    //     {
    //     public:
    //         // Initialise
    //         // Need to check that the region name is valid
    //         __host__ [[nodiscard]] boundaryValue(const std::string &fieldName, const std::string &regionName)
    //             : value(initialiseValue(fieldName, regionName)){};

    //         __host__ [[nodiscard]] inline constexpr scalar_t operator()() const noexcept
    //         {
    //             return value;
    //         }

    //     private:
    //         // Underlying variables
    //         const scalar_t value;

    //         __host__ [[nodiscard]] scalar_t initialiseValue(const std::string &fieldName, const std::string &regionName, const std::string &initialConditionsName = "initialConditions") const
    //         {
    //             const std::vector<std::string> boundaryLines = string::readFile(initialConditionsName);

    //             // Extracts the entire block of text corresponding to currentField
    //             const std::vector<std::string> fieldBlock = string::extractBlock(boundaryLines, fieldName, "field");

    //             // Extracts the block of text corresponding to internalField within the current field block
    //             const std::vector<std::string> internalFieldBlock = string::extractBlock(fieldBlock, regionName);

    //             // Now read the value line
    //             const std::string value_ = string::extractParameterLine(internalFieldBlock, "value");

    //             // Try fixing its value
    //             if (string::is_number(value_))
    //             {
    //                 const std::unordered_set<std::string> allowed = {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"};

    //                 const bool isMember = allowed.find(fieldName) != allowed.end();

    //                 if (isMember)
    //                 { // Check to see if it is a moment or a velocity and scale appropriately
    //                     if (fieldName == "rho")
    //                     {
    //                         return string::extractParameter<scalar_t>(internalFieldBlock, "value");
    //                     }
    //                     if ((fieldName == "u") | (fieldName == "v") | (fieldName == "w"))
    //                     {
    //                         return string::extractParameter<scalar_t>(internalFieldBlock, "value") * VelocitySet::velocitySet::scale_i<scalar_t>();
    //                     }
    //                     if ((fieldName == "m_xx") | (fieldName == "m_yy") | (fieldName == "m_zz"))
    //                     {
    //                         return string::extractParameter<scalar_t>(internalFieldBlock, "value") * VelocitySet::velocitySet::scale_ii<scalar_t>();
    //                     }
    //                     if ((fieldName == "m_xy") | (fieldName == "m_xz") | (fieldName == "m_yz"))
    //                     {
    //                         return string::extractParameter<scalar_t>(internalFieldBlock, "value") * VelocitySet::velocitySet::scale_ij<scalar_t>();
    //                     }
    //                 }

    //                 throw std::runtime_error("Invalid field name \" " + fieldName + "\" for equilibrium distribution");
    //             }
    //             // Otherwise, test to see if it is an equilibrium moment
    //             else if (value_ == "equilibrium")
    //             {
    //                 // Check to see if the variable is one of the moments
    //                 const std::unordered_set<std::string> allowed = {"m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"};
    //                 const bool isMember = allowed.find(fieldName) != allowed.end();

    //                 std::cout << "Constructing equilibrium moment" << std::endl;

    //                 // It is an equilibrium moment
    //                 if (isMember)
    //                 {
    //                     // Construct the velocity values
    //                     const boundaryValue u("u", regionName);
    //                     const boundaryValue v("v", regionName);
    //                     const boundaryValue w("w", regionName);

    //                     // Construct the equilibrium distribution
    //                     const std::array<scalar_t, VSet::Q()> pop = VSet::F_eq(u.value, v.value, w.value);

    //                     // Compute second-order moments
    //                     const scalar_t pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) - VelocitySet::velocitySet::cs2<scalar_t>();
    //                     const scalar_t pixy = (pop[7] + pop[8] - pop[13] - pop[14]);
    //                     const scalar_t pixz = (pop[9] + pop[10] - pop[15] - pop[16]);
    //                     const scalar_t piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) - VelocitySet::velocitySet::cs2<scalar_t>();
    //                     const scalar_t piyz = (pop[11] + pop[12] - pop[17] - pop[18]);
    //                     const scalar_t pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) - VelocitySet::velocitySet::cs2<scalar_t>();

    //                     // Store second-order moments (this can probably be improved)
    //                     if (fieldName == "m_xx")
    //                     {
    //                         return VelocitySet::velocitySet::scale_ii<scalar_t>() * (pixx);
    //                     }
    //                     if (fieldName == "m_xy")
    //                     {
    //                         return VelocitySet::velocitySet::scale_ij<scalar_t>() * (pixy);
    //                     }
    //                     if (fieldName == "m_xz")
    //                     {
    //                         return VelocitySet::velocitySet::scale_ij<scalar_t>() * (pixz);
    //                     }
    //                     if (fieldName == "m_yy")
    //                     {
    //                         return VelocitySet::velocitySet::scale_ii<scalar_t>() * (piyy);
    //                     }
    //                     if (fieldName == "m_yz")
    //                     {
    //                         return VelocitySet::velocitySet::scale_ij<scalar_t>() * (piyz);
    //                     }
    //                     if (fieldName == "m_zz")
    //                     {
    //                         return VelocitySet::velocitySet::scale_ii<scalar_t>() * (pizz);
    //                     }

    //                     throw std::runtime_error("Invalid field name \" " + fieldName + "\" for equilibrium distribution");
    //                 }
    //                 // Otherwise, not valid
    //                 else
    //                 {
    //                     std::cerr << "Entry for " << fieldName << " in region " << regionName << " not a valid numerical value and not an equilibrium moment" << std::endl;

    //                     throw std::runtime_error("Invalid field name for equilibrium distribution");

    //                     return 0;
    //                 }
    //             }
    //             // Not valid
    //             else
    //             {
    //                 std::cerr << "Entry for " << fieldName << " in region " << regionName << " not a valid numerical value and not an equilibrium moment" << std::endl;

    //                 throw std::runtime_error("Invalid field name for equilibrium distribution");

    //                 return 0;
    //             }
    //         }
    //     };

    //     // Defines a group of fields at a particular boundary
    //     template <class VSet>
    //     struct boundaryRegion
    //     {
    //     public:
    //         // Need to check that the length of fieldNames is 10
    //         [[nodiscard]] boundaryRegion(const std::string &regionName)
    //             : values_{
    //                   boundaryValue("rho", regionName),
    //                   boundaryValue("u", regionName),
    //                   boundaryValue("v", regionName),
    //                   boundaryValue("w", regionName),
    //                   boundaryValue("m_xx", regionName),
    //                   boundaryValue("m_xy", regionName),
    //                   boundaryValue("m_xz", regionName),
    //                   boundaryValue("m_yy", regionName),
    //                   boundaryValue("m_yz", regionName),
    //                   boundaryValue("m_zz", regionName)}
    //         {
    // #ifdef VERBOSE
    //             print();
    // #endif
    //         };

    //         [[nodiscard]] inline constexpr scalar_t rho() const noexcept
    //         {
    //             return values_[index::rho()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t u() const noexcept
    //         {
    //             return values_[index::u()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t v() const noexcept
    //         {
    //             return values_[index::v()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t w() const noexcept
    //         {
    //             return values_[index::w()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t m_xx() const noexcept
    //         {
    //             return values_[index::xx()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t m_xy() const noexcept
    //         {
    //             return values_[index::xy()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t m_xz() const noexcept
    //         {
    //             return values_[index::xz()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t m_yy() const noexcept
    //         {
    //             return values_[index::yy()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t m_yz() const noexcept
    //         {
    //             return values_[index::yz()]();
    //         }
    //         [[nodiscard]] inline constexpr scalar_t m_zz() const noexcept
    //         {
    //             return values_[index::zz()]();
    //         }

    //         void print() const noexcept
    //         {
    //             const std::vector<std::string> regionNames({"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"});
    //             for (std::size_t field = 0; field < regionNames.size(); field++)
    //             {
    //                 std::cout << regionNames[field] << ": " << values_[field]() << std::endl;
    //             }
    //         }

    //     private:
    //         const boundaryValue<VSet> values_[10];
    //     };

    //     // Defines the value of a field at a group of boundaries
    //     template <class VSet>
    //     struct boundaryFields
    //     {
    //     public:
    //         [[nodiscard]] boundaryFields(const std::string &fieldName)
    //             : values_{
    //                   boundaryValue<VSet>(fieldName, "North"),
    //                   boundaryValue<VSet>(fieldName, "South"),
    //                   boundaryValue<VSet>(fieldName, "East"),
    //                   boundaryValue<VSet>(fieldName, "West"),
    //                   boundaryValue<VSet>(fieldName, "Front"),
    //                   boundaryValue<VSet>(fieldName, "Back"),
    //                   boundaryValue<VSet>(fieldName, "internalField")},
    //               fieldName_(fieldName)
    //         {
    //             print();
    //         };

    //         [[nodiscard]] inline constexpr scalar_t North() const noexcept { return values_[0](); }
    //         [[nodiscard]] inline constexpr scalar_t South() const noexcept { return values_[1](); }
    //         [[nodiscard]] inline constexpr scalar_t East() const noexcept { return values_[2](); }
    //         [[nodiscard]] inline constexpr scalar_t West() const noexcept { return values_[3](); }
    //         [[nodiscard]] inline constexpr scalar_t Front() const noexcept { return values_[4](); }
    //         [[nodiscard]] inline constexpr scalar_t Back() const noexcept { return values_[5](); }
    //         [[nodiscard]] inline constexpr scalar_t internalField() const noexcept { return values_[6](); }

    //         void print() const noexcept
    //         {
    //             std::cout << fieldName_ << " boundary values:" << std::endl;

    //             const std::vector<std::string> fieldNames({"North", "South", "East", "West", "Front", "Back", "Internal"});
    //             for (std::size_t var = 0; var < fieldNames.size(); var++)
    //             {
    //                 std::cout << fieldNames[var] << ": " << values_[var]() << std::endl;
    //             }
    //         }

    //     private:
    //         const boundaryValue<VSet> values_[7];

    //         const std::string &fieldName_;
    //     };

    /**
     * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
     * @param devPtrs Collection of 10 pointers to device arrays on the GPU
     * @param blockHalo Object containing pointers to the block halo faces used to exchange the population densities
     **/
    launchBounds __global__ void momentBasedD3Q19(
        const device::ptrCollection<10, scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> fGhost,
        const device::ptrCollection<6, scalar_t> gGhost)
    {
        // Always a multiple of 32, so no need to check this(I think)
        // if (device::out_of_bounds())
        // {
        //     return;
        // }

        // Declare shared memory (flattened)
        __shared__ scalar_t shared_buffer[block::sharedMemoryBufferSize<VSet, NUMBER_MOMENTS()>()];

        const label_t tid = threadIdx.x + (threadIdx.y * block::nx()) + (threadIdx.z * block::nx() * block::ny());
        const label_t idx = device::idx();

        // Coalesced read from global memory
        threadArray<scalar_t, NUMBER_MOMENTS()> moments;
        {
            // Read into shared
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    shared_buffer[moment * block::stride() + tid] = devPtrs.ptr<moment>()[idx];
                    if constexpr (moment == index::rho())
                    {
                        moments.arr[moment] = shared_buffer[moment * block::stride() + tid] + rho0<scalar_t>();
                    }
                    else
                    {
                        moments.arr[moment] = shared_buffer[moment * block::stride() + tid];
                    }
                });
        }

        // Reconstruct the population from the moments
        threadArray<scalar_t, VSet::Q()> pop = VSet::reconstruct(moments);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            sharedMemory::save<VSet>(pop.arr, shared_buffer, tid);

            __syncthreads();

            // Pull from shared memory
            sharedMemory::pull<VSet>(pop.arr, shared_buffer);
        }

        // Load pop from global memory in cover nodes
        device::halo<VSet>::popLoad(
            pop.arr,
            fGhost.ptr<0>(),
            fGhost.ptr<1>(),
            fGhost.ptr<2>(),
            fGhost.ptr<3>(),
            fGhost.ptr<4>(),
            fGhost.ptr<5>());

        // Calculate the moments either at the boundary or interior
        {
            const normalVector b_n;
            if (b_n.isBoundary())
            {
                boundaryConditions::calculateMoments<VSet>(pop.arr, moments.arr, b_n);
            }
            else
            {
                VSet::calculateMoments(pop.arr, moments.arr);
            }
        }

        // Scale the moments correctly
        VelocitySet::velocitySet::scale(moments.arr);

        // Collide
        Collision::collide(moments.arr);

        // Calculate post collision populations
        VSet::reconstruct(pop.arr, moments.arr);

        // Coalesced write to global memory
        moments.arr[0] = moments.arr[0] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments.arr[moment];
            });

        // Save the populations to the block halo
        device::halo<VSet>::popSave(
            pop.arr,
            gGhost.ptr<0>(),
            gGhost.ptr<1>(),
            gGhost.ptr<2>(),
            gGhost.ptr<3>(),
            gGhost.ptr<4>(),
            gGhost.ptr<5>());
    }
}

#endif