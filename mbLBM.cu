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
#include "nodeTypeArray/nodeTypeArray.cuh"

using namespace mbLBM;

int main(int argc, char *argv[])
{
    // const mpiStatus mpiStat(argc, argv);
    // const cudaCommunicator cudaComm;

    const programControl programCtrl(argc, argv);
    const host::latticeMesh mesh(ctorType::MUST_READ);
    const host::moments moms(mesh, ctorType::MUST_READ);

    // moms.writeFile(0);

    vSet::print();

    const device::ghostInterface interface(mesh);
    const device::moments devMom_0(  //
        programCtrl.deviceList()[0], //
        moms);                       //

    kernel_collide<<<
        dim3{mesh.nx(), mesh.ny(), mesh.nz()},
        dim3{16, 16, 4},
        0,
        0>>>(
        devMom_0,
        mesh,
        interface,
        mesh.nodeTypes(),
        programCtrl);

    return 0;
}