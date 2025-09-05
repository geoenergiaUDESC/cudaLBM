/**
Filename: fieldConvert.cuh
Contents: Function definitions specific to the fieldConvert executable
**/

#ifndef __MBLBM_FIELDCONVERT_CUH
#define __MBLBM_FIELDCONVERT_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../array/array.cuh"
#include "../collision/collision.cuh"
#include "../blockHalo/blockHalo.cuh"
#include "../fileIO/fileIO.cuh"
#include "../runTimeIO/runTimeIO.cuh"
#include "../postProcess/postProcess.cuh"
#include "../inputControl.cuh"

namespace LBM
{
    using WriterFunction = void (*)(
        const std::vector<std::vector<scalar_t>> &,
        const std::string &,
        const host::latticeMesh &,
        const std::vector<std::string> &,
        const std::string &);

    /**
     * @brief Veriefies if the command line has the argument -type
     * @return A string representing the convertion type passed at the command line
     * @param argc First argument passed to main
     * @param argv Second argument passed to main
     **/
    __host__ [[nodiscard]] const std::string getConversionType(const programControl &programCtrl)
    {
        if (programCtrl.input().isArgPresent("-type"))
        {
            for (label_t arg = 0; arg < programCtrl.commandLine().size(); arg++)
            {
                if (programCtrl.commandLine()[arg] == "-type")
                {
                    if (arg + 1 == programCtrl.commandLine().size())
                    {
                        throw std::runtime_error("Conversion type not specified: the correct syntax is -type T");
                        return 0;
                    }
                    else
                    {
                        return programCtrl.commandLine()[arg + 1];
                    }
                }
            }
        }

        throw std::runtime_error("Mandatory parameter -type not specified: the correct syntax is -type T");
    }

    /**
     * @brief Unordered map of the writer types to the appropriate functions
     **/
    const std::unordered_map<std::string, WriterFunction> writers = {
        {"vtu", postProcess::writeVTU},
        {"tecplot", postProcess::writeTecplot}};
}

#endif