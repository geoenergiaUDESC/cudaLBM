/**
Filename: labelArray.cuh
Contents: A class representing a list of labels
The host version of this class is principally used to
partition the mesh prior to distribution to the devices
**/

#ifndef __MBLBM_LABELARRAYS_CUH
#define __MBLBM_LABELARRAYS_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

namespace LBM
{
    namespace host
    {
        template <const ctorType::type CType>
        using labelArray = array<label_t, CType>;
    }

    namespace device
    {
        typedef array<label_t> labelArray;
    }
}

#endif