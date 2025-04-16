/**
Filename: programControl.cuh
Contents: A class handling the setup of the solver
**/

#ifndef __MBLBM_PROGRAMCONTROL_CUH
#define __MBLBM_PROGRAMCONTROL_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "exceptionHandler.cuh"
#include "strings.cuh"
#include "inputControl.cuh"

namespace mbLBM
{
    class programControl
    {
    public:
        /**
         * @brief Constructor for the programControl class
         **/
        [[nodiscard]] programControl(int argc, char *argv[]) noexcept
            : input_(inputControl(argc, argv)),
              nx_(string::extractParameter<label_t>(readCaseDirectory("caseInfo"), "nx")),
              ny_(string::extractParameter<label_t>(readCaseDirectory("caseInfo"), "ny")),
              nz_(string::extractParameter<label_t>(readCaseDirectory("caseInfo"), "nz")),
              Re_(string::extractParameter<scalar_t>(readCaseDirectory("caseInfo"), "Re")),
              Lx_(string::extractParameter<scalar_t>(readCaseDirectory("caseInfo"), "Lx")),
              Ly_(string::extractParameter<scalar_t>(readCaseDirectory("caseInfo"), "Ly")),
              Lz_(string::extractParameter<scalar_t>(readCaseDirectory("caseInfo"), "Lz")),
              nTimeSteps_(string::extractParameter<label_t>(readCaseDirectory("caseInfo"), "nTimeSteps"))
        {
            std::cout << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //" << std::endl;
            std::cout << "//                                                                         //" << std::endl;
            std::cout << "// UDESC mbLBM                                                             //" << std::endl;
            std::cout << "// Universidade do Esdado de Santa Catarina                                //" << std::endl;
            std::cout << "//                                                                         //" << std::endl;
            std::cout << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //" << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            if (deviceList().size() > 1)
            {
                std::cout << "Executing on devices ";
            }
            else
            {
                std::cout << "Executing on device ";
            }
            for (label_t i = 0; i < deviceList().size() - 1; i++)
            {
                std::cout << deviceList()[i] << ", ";
            }
            std::cout << deviceList()[deviceList().size() - 1] << std::endl;
        };

        /**
         * @brief Destructor for the programControl class
         **/
        ~programControl() noexcept {};

        /**
         * @brief Returns the number of lattices in the x, y and z directions
         **/
        [[nodiscard]] inline constexpr label_t nx() const noexcept
        {
            return nx_;
        }
        [[nodiscard]] inline constexpr label_t ny() const noexcept
        {
            return ny_;
        }
        [[nodiscard]] inline constexpr label_t nz() const noexcept
        {
            return nz_;
        }
        [[nodiscard]] inline constexpr label_t nPoints() const noexcept
        {
            return nx_ * ny_ * nz_;
        }

        /**
         * @brief Returns the array of device indices
         * @return A read-only reference to deviceList_ contained within input_
         **/
        [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return input_.deviceList();
        }

    private:
        /**
         * @brief A reference to the input control object
         **/
        const inputControl &input_;

        /**
         * @brief The number of lattices in the x, y and z directions
         **/
        const label_t nx_;
        const label_t ny_;
        const label_t nz_;

        /**
         * @brief The Reynolds number
         **/
        const scalar_t Re_;

        /**
         * @brief Characteristic domain length in the x, y and z directions
         **/
        const scalar_t Lx_;
        const scalar_t Ly_;
        const scalar_t Lz_;

        /**
         * @brief Total number of simulation time steps
         **/
        const label_t nTimeSteps_;

        /**
         * @brief Reads the caseInfo file in the current directory into a vector of strings
         * @return A std::vector of std::string_view objects contained within the caseInfo file
         * @note This function will cause the program to exit if caseInfo is not found in the launch directory
         **/
        [[nodiscard]] std::vector<std::string> readCaseDirectory(const std::string_view &fileName) const noexcept
        {
            // Does the file even exist?
            if (!std::filesystem::exists(fileName))
            {
                exceptions::program_exit(-1, std::string(fileName) + std::string(" file not opened"));
            }

            // Read the caseInfo file contained within the directory
            std::ifstream caseInfo(std::string(fileName).c_str());
            std::vector<std::string> S;
            std::string s;

            // Count the number of lines
            label_t nLines = 0;

            // Count the number of lines
            while (std::getline(caseInfo, s))
            {
                S.push_back(s);
                nLines = nLines + 1;
            }

            S.resize(nLines);

            return S;
        }
    };
}

#endif