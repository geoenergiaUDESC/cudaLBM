/**
Filename: numericalSchemes.cuh
Contents: Collection of numerical schemes used throughout the code
**/

#ifndef __MBLBM_NUMERICALSCHEMES_CUH
#define __MBLBM_NUMERICALSCHEMES_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{

}

#include "derivativeSchemes/derivativeSchemes.cuh"
#include "integrationSchemes/fieldIntegrate.cuh"

#endif