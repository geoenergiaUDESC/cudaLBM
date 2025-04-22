#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "globalFunctions.cuh"
#include "programControl.cuh"
#include "inputControl.cuh"
#include "mpiStatus.cuh"
#include "cudaCommunicator.cuh"
#include "moments.cuh"
#include "latticeMesh.cuh"

using namespace mbLBM;

int main(int argc, char *argv[])
{
    const programControl programCtrl(argc, argv);
    // const mpiStatus mpiStat(argc, argv);
    // const cudaCommunicator cudaComm;
    const latticeMesh mesh;
    const host::moments moms(mesh);

    // Ok, now try allocating the partitioned moments to each GPU
    const device::moments devMom_0(                                                                                        //
        programCtrl.deviceList()[0],                                                                                       //
        host::moments(                                                                                                     //
            latticeMesh(mesh, {{0, mesh.nx()}, {0, mesh.ny()}, {0, mesh.nz() / programCtrl.deviceList().size()}}),         //
            moms)                                                                                                          //
    );                                                                                                                     //
    const device::moments devMom_1(                                                                                        //
        programCtrl.deviceList()[1],                                                                                       //
        host::moments(                                                                                                     //
            latticeMesh(mesh, {{0, mesh.nx()}, {0, mesh.ny()}, {mesh.nz() / programCtrl.deviceList().size(), mesh.nz()}}), //
            moms)                                                                                                          //
    );                                                                                                                     //

    return 0;
}