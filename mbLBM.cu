#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "programControl.cuh"
#include "inputControl.cuh"
#include "mpiStatus.cuh"
#include "cudaCommunicator.cuh"
#include "moments.cuh"

using namespace mbLBM;

struct blockLabel_t
{
    const label_t nx;
    const label_t ny;
    const label_t nz;
};

template <typename T>
[[nodiscard]] inline constexpr T blockLabel(const T i, const T j, const T k, const blockLabel_t &blockLabel)
{
    return (k * (blockLabel.nx * blockLabel.ny)) + (j * blockLabel.nx) + (i);
}

int main(int argc, char *argv[])
{
    const programControl programCtrl(argc, argv);

    const mpiStatus mpiStat(argc, argv);

    const cudaCommunicator cudaComm;

    host::moments moms(programCtrl.nPoints());

    std::cout << "Allocated moments with " << programCtrl.nPoints() << " lattices" << std::endl;

    // moms.setRho();

    // Now partition the host arrays in order to prepare for distribution amongst the GPUs
    const label_t nPointsHalf = programCtrl.nPoints() / 2;
    std::cout << "Partitioning moment arrays in blocks of " << nPointsHalf << " lattices" << std::endl;

    // Create the partition labels
    labelArray_t part_0(nPointsHalf);
    labelArray_t part_1(nPointsHalf);
    for (label_t i = 0; i < nPointsHalf; i++)
    {
        part_0[i] = i;
        part_1[i] = i + nPointsHalf;
    }

    // Create the partitioned arrays from the original and the partition labels
    const host::scalarArray f_0 = host::scalarArray(moms.rho(), part_0);
    const host::scalarArray f_1 = host::scalarArray(moms.rho(), part_1);

    // Now try printing the partitioned arrays
    for (label_t k = 0; k < programCtrl.nz() / 2; k++)
    {
        for (label_t j = 0; j < programCtrl.ny(); j++)
        {
            for (label_t i = 0; i < programCtrl.nx(); i++)
            {
                std::cout << f_0.arrRef()[blockLabel<label_t>(i, j, k, {programCtrl.nx(), programCtrl.ny(), programCtrl.nz() / 2})] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    for (label_t k = 0; k < programCtrl.nz() / 2; k++)
    {
        for (label_t j = 0; j < programCtrl.ny(); j++)
        {
            for (label_t i = 0; i < programCtrl.nx(); i++)
            {
                // std::cout << part_1[blockLabel<label_t>(i, j, k, {programCtrl.nx(), programCtrl.ny(), programCtrl.nz() / 2})] << " ";
                std::cout << f_1.arrRef()[blockLabel<label_t>(i, j, k, {programCtrl.nx(), programCtrl.ny(), programCtrl.nz() / 2})] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}