#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "globalFunctions.cuh"
#include "programControl.cuh"
#include "inputControl.cuh"
#include "mpiStatus.cuh"
#include "cudaCommunicator.cuh"
#include "moments.cuh"
#include "latticeMesh.cuh"
#include "collision.cuh"

using namespace mbLBM;

// const mpiStatus mpiStat(argc, argv);
// const cudaCommunicator cudaComm;

int main(int argc, char *argv[])
{

    const programControl programCtrl(argc, argv);
    const latticeMesh mesh;

    VelocitySet::D3Q19::print();

    const device::ghostInterface<VelocitySet::D3Q19> interface(mesh);

    // const host::moments moms(mesh);
    // const device::moments devMom_0(                                              //
    //     programCtrl.deviceList()[1],                                             //
    //     host::moments(                                                           //
    //         latticeMesh(mesh, {{0, mesh.nx()}, {0, mesh.ny()}, {0, mesh.nz()}}), //
    //         moms)                                                                //
    // );                                                                           //

    return 0;
}