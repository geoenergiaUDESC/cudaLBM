/**
Filename: nodeTypeArray.cuh
Contents: A class representing the types of each individual mesh node
This is a temporary fix before the boundary condition function pointers are implemented
**/

#ifndef __MBLBM_NODETYPEARRAY_CUH
#define __MBLBM_NODETYPEARRAY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace host
    {
        template <const ctorType::type CType>
        using nodeArray = array<nodeType_t, CType>;
    }

    namespace device
    {
        typedef array<nodeType_t> nodeTypeArray;
    }
}

#endif
