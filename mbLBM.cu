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

    VelocitySet::D3Q19::print();

    const device::ghostInterface<VelocitySet::D3Q19> interface(mesh);

    const host::moments moms(mesh);
    const device::moments devMom_0(                                                    //
        programCtrl.deviceList()[0],                                                   //
        host::moments(                                                                 //
            host::latticeMesh(mesh, {{0, mesh.nx()}, {0, mesh.ny()}, {0, mesh.nz()}}), //
            moms)                                                                      //
    );                                                                                 //

    const device::nodeTypeArray deviceNodeTypes(mesh.nodeTypes());

    kernel_collide<VelocitySet::D3Q19><<<
        dim3{mesh.nx(), mesh.ny(), mesh.nz()},
        dim3{16, 16, 4},
        0,
        0>>>(
        devMom_0,
        mesh,
        interface,
        deviceNodeTypes,
        programCtrl);

    // do_nothing<0><<<dim3mesh_,
    //                 dim3{16, 16, 4},
    //                 128,
    //                 0>>>();

    return 0;
}