/**
Filename: scalarArray.cuh
Contents: A class representing a scalar variable
The host version of this class is principally used to
partition the mesh prior to distribution to the devices
**/

#ifndef __MBLBM_SCALARARRAYS_CUH
#define __MBLBM_SCALARARRAYS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"

namespace LBM
{
    namespace host
    {
        // typedef array<scalar_t> scalarArray;
    }

    namespace device
    {
        typedef array<scalar_t> scalarArray;
    }
}

#endif
