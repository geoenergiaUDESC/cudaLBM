/**
Filename: array.cuh
Contents: A templated class for various different types of arrays
**/

#ifndef __MBLBM_ARRAY_CUH
#define __MBLBM_ARRAY_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../latticeMesh/latticeMesh.cuh"
#include "../fileIO/fileIO.cuh"
#include "../velocitySet/velocitySet.cuh"
#include "../boundaryConditions/boundaryConditions.cuh"

namespace LBM
{

}

#include "hostArray.cuh"
#include "deviceArray.cuh"

#endif
