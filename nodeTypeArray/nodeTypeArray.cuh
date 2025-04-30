/**
Filename: scalarArray.cuh
Contents: A class representing a scalar variable
The host version of this class is principally used to
partition the mesh prior to distribution to the devices
**/

#ifndef __MBLBM_NODETYPEARRAY_CUH
#define __MBLBM_NODETYPEARRAY_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "latticeMesh.cuh"
#include "labelArray.cuh"

namespace mbLBM
{
    // namespace host
    // {
    //     class nodeTypeArray
    //     {
    //     public:
    //         // Default constructor - return all UNDEFINED
    //         [[nodiscard]] nodeTypeArray()
    //             : nodeTypes_(extractNodeTypes("nodeTypes", ctorType::NO_READ)) {};

    //         // Construct specifying
    //         ~nodeTypeArray() {};

    //     private:
    //         const nodeTypeArray_t nodeTypes_;

    //         [[nodiscard]] const nodeTypeArray_t extractNodeTypes(const std::string &functionName, const ctorType::type readType) const noexcept
    //         {
    //             if (readType == ctorType::MUST_READ)
    //             {
    //                 const std::vector<std::string> fileString = string::readCaseDirectory("mesh");
    //                 const string::functionNameLines_t lines(fileString, functionName);
    //                 nodeTypeArray_t v(lines.nLines, nodeType::UNDEFINED);
    //                 for (std::size_t i = lines.openBracketLine + 1; i < lines.closeBracketLine; i++)
    //                 {
    //                     v[i - lines.openBracketLine - 1] = static_cast<nodeType::type>(stoi(fileString[i]));
    //                 }
    //                 return v;
    //             }
    //             else
    //             {
    //                 return nodeTypeArray_t(nPoints_, nodeType::UNDEFINED);
    //             }
    //         }
    //     };
    // }

    namespace device
    {
        class nodeTypeArray
        {
        public:
            /**
             * @brief Constructs a scalar solution variable on the device from a copy of its host version
             * @return A scalarArray object copied from the host to the device
             * @param f The host copy of the scalar solution variable
             **/
            [[nodiscard]] nodeTypeArray(const nodeTypeArray_t &nodeTypes) noexcept
                : ptr_(device::allocateArray<nodeType::type>(nodeTypes))
            {
#ifdef VERBOSE
                std::cout << "Allocated device node types for " << nodeTypes.size() << " lattice points" << std::endl;
#endif
            };

            /**
             * @brief Destructor
             **/
            ~nodeTypeArray() noexcept
            {
#ifdef VERBOSE
                std::cout << "Freeing device node types" << std::endl;
#endif
                cudaFree((void *)ptr_);
            };

            /**
             * @brief Returns immutable access to the underlying pointer
             * @return A const-qualified pointer
             **/
            __device__ [[nodiscard]] inline constexpr const nodeType::type *ptr() const noexcept
            {
                return ptr_;
            }

        private:
            /**
             * @brief Pointer to the underlying array
             **/
            const nodeType::type *ptr_;
        };
    }
}

#endif
