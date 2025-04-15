#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "programControl.cuh"
#include "inputControl.cuh"
#include "mpiStatus.cuh"
#include "cudaCommunicator.cuh"

using namespace mbLBM;

int main(int argc, char *argv[])
{
    const programControl programCtrl(argc, argv);

    const mpiStatus mpiStat(argc, argv);

    const cudaCommunicator cudaComm;

    return 0;
}