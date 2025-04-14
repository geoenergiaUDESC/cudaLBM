#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "programControl.cuh"
#include "inputControl.cuh"

using namespace mbLBM;

int main(const int argc, const char *argv[])
{
    const programControl program(argc, argv);

    std::cout << "nx = " << program.nx() << std::endl;
    std::cout << "ny = " << program.ny() << std::endl;
    std::cout << "nz = " << program.nz() << std::endl;

    return 0;
}