/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    A class handling the setup of the solver

Namespace
    LBM

SourceFiles
    programControl.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PROGRAMCONTROL_CUH
#define __MBLBM_PROGRAMCONTROL_CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "strings.cuh"
#include "inputControl.cuh"
#include "fileIO/fileIO.cuh"

namespace LBM
{
    class programControl
    {
    public:
        /**
         * @brief Constructor for the programControl class
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        [[nodiscard]] programControl(const int argc, const char *const argv[]) noexcept
            : input_(inputControl(argc, argv)),
              caseName_(string::extractParameter<std::string>(string::readFile("caseInfo"), "caseName")),
              Re_(initialiseConst<scalar_t>("Re")),
              u_inf_(initialiseConst<scalar_t>("u_inf")),
              Lx_(string::extractParameter<scalar_t>(string::readFile("caseInfo"), "Lx")),
              Ly_(string::extractParameter<scalar_t>(string::readFile("caseInfo"), "Ly")),
              Lz_(string::extractParameter<scalar_t>(string::readFile("caseInfo"), "Lz")),
              nTimeSteps_(string::extractParameter<label_t>(string::readFile("caseInfo"), "nTimeSteps")),
              saveInterval_(string::extractParameter<label_t>(string::readFile("caseInfo"), "saveInterval")),
              infoInterval_(string::extractParameter<label_t>(string::readFile("caseInfo"), "infoInterval")),
              latestTime_(fileIO::latestTime(caseName_))
        {
            static_assert((std::is_same_v<scalar_t, float>) | (std::is_same_v<scalar_t, double>), "Invalid floating point size: must be either 32 or 64 bit");

            static_assert((std::is_same_v<label_t, uint32_t>) | (std::is_same_v<label_t, uint64_t>), "Invalid label size: must be either 32 bit unsigned or 64 bit unsigned");

            std::cout << "{ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }" << std::endl;
            std::cout << "{                                                                         }" << std::endl;
            std::cout << "{ UDESC LBM                                                               }" << std::endl;
            std::cout << "{ Universidade do Esdado de Santa Catarina                                }" << std::endl;
            std::cout << "{                                                                         }" << std::endl;
            std::cout << "{ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }" << std::endl;
            std::cout << std::endl;
            std::cout << "Executable: " << input_.commandLine()[0] << std::endl;
            std::cout << "programControl:" << std::endl;
            std::cout << "{" << std::endl;
            std::cout << "    deviceList: [";
            if (deviceList().size() > 1)
            {
                for (label_t i = 0; i < deviceList().size() - 1; i++)
                {
                    std::cout << deviceList()[i] << ", ";
                }
            }
            std::cout << deviceList()[deviceList().size() - 1] << "];" << std::endl;
            std::cout << "    Case: " << caseName_ << ";" << std::endl;
            std::cout << "    Re = " << Re_ << ";" << std::endl;
            std::cout << "    Lx = " << Lx_ << ";" << std::endl;
            std::cout << "    Ly = " << Ly_ << ";" << std::endl;
            std::cout << "    Lz = " << Lz_ << ";" << std::endl;
            std::cout << "    nTimeSteps = " << nTimeSteps_ << ";" << std::endl;
            std::cout << "    saveInterval = " << saveInterval_ << ";" << std::endl;
            std::cout << "    infoInterval = " << infoInterval_ << ";" << std::endl;
            std::cout << "    latestTime = " << latestTime_ << ";" << std::endl;
            std::cout << "    Scalar type: " << ((sizeof(scalar_t) == 4) ? "32 bit" : "64 bit") << std::endl;
            std::cout << "    Label type: " << ((sizeof(label_t) == 4) ? "uint32_t" : "uint64_t") << std::endl;
            std::cout << "};" << std::endl;
            std::cout << std::endl;

            cudaDeviceSynchronize();
        };

        /**
         * @brief Destructor for the programControl class
         **/
        ~programControl() noexcept {};

        /**
         * @brief Returns the name of the case
         * @return A const std::string
         **/
        [[nodiscard]] inline constexpr const std::string &caseName() const noexcept
        {
            return caseName_;
        }

        /**
         * @brief Returns the array of device indices
         * @return A read-only reference to deviceList_ contained within input_
         **/
        [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return input_.deviceList();
        }

        /**
         * @brief Returns the Reynolds number
         * @return The Reynolds number
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t Re() const noexcept
        {
            return Re_;
        }

        /**
         * @brief Returns the characteristic velocity
         * @return The characteristic velocity
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t u_inf() const noexcept
        {
            return u_inf_;
        }

        /**
         * @brief Returns the total number of simulation time steps
         * @return The total number of simulation time steps
         **/
        __device__ __host__ [[nodiscard]] inline constexpr label_t nt() const noexcept
        {
            return nTimeSteps_;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool save(const label_t timeStep) const noexcept
        {
            return (timeStep % saveInterval_) == 0;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool print(const label_t timeStep) const noexcept
        {
            return (timeStep % infoInterval_) == 0;
        }

        /**
         * @brief Returns the latest time step of the solution files contained within the current directory
         * @return The latest time step as a label_t
         **/
        __device__ __host__ [[nodiscard]] inline constexpr label_t latestTime() const noexcept
        {
            return latestTime_;
        }

        /**
         * @brief Provides read-only access to the input control
         * @return A const reference to an inputControl object
         **/
        __host__ [[nodiscard]] inline constexpr const inputControl &input() const noexcept
        {
            return input_;
        }

        /**
         * @brief Provides read-only access to the arguments supplied at the command line
         * @return The command line input as a vector of strings
         **/
        __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &commandLine() const noexcept
        {
            return input_.commandLine();
        }

        __host__ [[nodiscard]] inline constexpr const pointVector L() const noexcept
        {
            return {Lx_, Ly_, Lz_};
        }

    private:
        /**
         * @brief A reference to the input control object
         **/
        const inputControl input_;

        /**
         * @brief The name of the simulation case
         **/
        const std::string caseName_;

        /**
         * @brief The Reynolds number
         **/
        const scalar_t Re_;

        /**
         * @brief The characteristic velocity
         **/
        const scalar_t u_inf_;

        /**
         * @brief Characteristic domain length in the x, y and z directions
         **/
        const scalar_t Lx_;
        const scalar_t Ly_;
        const scalar_t Lz_;

        /**
         * @brief Total number of simulation time steps, the save interval, info output interval and the latest time step at program start
         **/
        const label_t nTimeSteps_;
        const label_t saveInterval_;
        const label_t infoInterval_;
        const label_t latestTime_;

        /**
         * @brief Reads a variable from the caseInfo file into a parameter of type T
         * @return The variable as type T
         * @param varName The name of the variable to read
         **/
        template <typename T>
        [[nodiscard]] T initialiseConst(const std::string varName) const noexcept
        {
            return string::extractParameter<T>(string::readFile("caseInfo"), varName);
        }
    };
}

#endif