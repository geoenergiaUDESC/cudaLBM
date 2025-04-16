/**
Filename: scalarArray.cuh
Contents: A class representing a scalar variable
The host version of this class is principally used to
partition the mesh prior to distribution to the devices
**/

#ifndef __MBLBM_SCALARARRAYS_CUH
#define __MBLBM_SCALARARRAYS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace mbLBM
{
    namespace host
    {
        class scalarArray
        {
        public:
            /**
             * @brief Constructs a scalar solution variable from a number of lattice points
             * @return A scalar solution variable
             * @param nPoints Total number of lattice points
             * @note This constructor zero-initialises everything
             **/
            [[nodiscard]] scalarArray(const label_t nPoints) : nPoints_(nPoints), arr_(scalarArray_t(nPoints, 0)) {};

            /**
             * @brief Constructs a scalar solution variable from another scalarArray and a partition list
             * @return A partition scalar solution variable
             * @param originalArray The original array to be partitioned
             * @param partitionIndices A list of unsigned integers corresponding to indices of originalArray
             * @note This constructor copies the elements corresponding to the partition points into the new object
             **/
            [[nodiscard]] scalarArray(const scalarArray &originalArray, const labelArray_t &partitionIndices) : nPoints_(partitionIndices.size()), arr_(partitionOriginal(originalArray, partitionIndices)) {};

            /**
             * @brief Destructor
             **/
            ~scalarArray() {};

            /**
             * @brief Returns immutable access to the underlying array
             * @return An immutable reference to the underlying array
             **/
            [[nodiscard]] inline scalarArray_t const &arrRef() const noexcept
            {
                return arr_;
            }

        private:
            /**
             * @brief Total number of lattice points
             **/
            const label_t nPoints_;

            /**
             * @brief The underlying solution array
             **/
            const scalarArray_t arr_;

            /**
             * @brief Used to partition an original array by an arbitrary list of partition indices
             * @return The elements of originalArray partitioned by partitionIndices
             * @param originalArray The original array to be partitioned
             * @param partitionIndices A list of unsigned integers corresponding to indices of originalArray
             **/
            [[nodiscard]] scalarArray_t partitionOriginal(
                const scalarArray &originalArray,
                const labelArray_t &partitionIndices)
            {
                // Create an appropriately sized array
                scalarArray_t partitionedArray(partitionIndices.size());
                for (std::size_t i = 0; i < partitionIndices.size(); i++)
                {
                    partitionedArray[i] = originalArray.arrRef()[partitionIndices[i]];
                }
                return partitionedArray;
            }
        };
    }
}

#endif