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
        [[nodiscard]] programControl(int argc, char *argv[]) noexcept
            : input_(inputControl(argc, argv)),
              Re_(initialiseConst<scalar_t>("Re")),
              u_inf_(initialiseConst<scalar_t>("u_inf")),
              Lx_(string::extractParameter<scalar_t>(string::readCaseDirectory("caseInfo"), "Lx")),
              Ly_(string::extractParameter<scalar_t>(string::readCaseDirectory("caseInfo"), "Ly")),
              Lz_(string::extractParameter<scalar_t>(string::readCaseDirectory("caseInfo"), "Lz")),
              nTimeSteps_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "nTimeSteps")),
              saveInterval_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "saveInterval")),
              infoInterval_(string::extractParameter<label_t>(string::readCaseDirectory("caseInfo"), "infoInterval"))
        {
            std::cout << "{ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }" << std::endl;
            std::cout << "{                                                                         }" << std::endl;
            std::cout << "{ UDESC LBM                                                             }" << std::endl;
            std::cout << "{ Universidade do Esdado de Santa Catarina                                }" << std::endl;
            std::cout << "{                                                                         }" << std::endl;
            std::cout << "{ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }" << std::endl;
            std::cout << std::endl;
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
            std::cout << "    Re = " << Re_ << ";" << std::endl;
            std::cout << "    Lx = " << Lx_ << ";" << std::endl;
            std::cout << "    Ly = " << Ly_ << ";" << std::endl;
            std::cout << "    Lz = " << Lz_ << ";" << std::endl;
            std::cout << "    nTimeSteps = " << nTimeSteps_ << ";" << std::endl;
            std::cout << "}" << std::endl;
            std::cout << std::endl;

            // checkCudaErrors(cudaMemcpyToSymbol(d_Re, &Re_, sizeof(d_Re)));
            // checkCudaErrors(cudaMemcpyToSymbol(d_u_inf, &u_inf_, sizeof(u_inf_)));
            // const scalar_t viscosity = u_inf_ * static_cast<scalar_t>(nx) / Re_;
            // const scalar_t tau = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosity;

            // const scalar_t omega = static_cast<scalar_t>(1.0) / tau;

            // checkCudaErrors(cudaMemcpyToSymbol(d_tau, &tau, sizeof(tau)));
            // checkCudaErrors(cudaMemcpyToSymbol(d_omega, &omega, sizeof(omega)));

            cudaDeviceSynchronize();
        };

        void copyDeviceSymbols(const label_t nx) const noexcept
        {
            const scalar_t viscosity = u_inf_ * static_cast<scalar_t>(nx) / Re_;
            const scalar_t tau = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosity;
            const scalar_t omega = static_cast<scalar_t>(1.0) / tau;

            cudaDeviceSynchronize();
            checkCudaErrors(cudaMemcpyToSymbol(d_Re, &Re_, sizeof(d_Re)));
            cudaDeviceSynchronize();
            checkCudaErrors(cudaMemcpyToSymbol(d_u_inf, &u_inf_, sizeof(u_inf_)));
            cudaDeviceSynchronize();
            cudaDeviceSynchronize();
            cudaDeviceSynchronize();
            checkCudaErrors(cudaMemcpyToSymbol(d_tau, &tau, sizeof(tau)));
            cudaDeviceSynchronize();
            checkCudaErrors(cudaMemcpyToSymbol(d_omega, &omega, sizeof(omega)));
            cudaDeviceSynchronize();
        }

        /**
         * @brief Destructor for the programControl class
         **/
        ~programControl() noexcept {};

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

    private:
        /**
         * @brief A reference to the input control object
         **/
        const inputControl input_;

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
         * @brief Total number of simulation time steps, the save interval and the info output interval
         **/
        const label_t nTimeSteps_;
        const label_t saveInterval_;
        const label_t infoInterval_;

        /**
         * @brief Reads a variable from the caseInfo file into a parameter of type T
         * @return The variable as type T
         * @param varName The name of the variable to read
         **/
        template <typename T>
        [[nodiscard]] T initialiseConst(const std::string varName) const noexcept
        {
            return string::extractParameter<T>(string::readCaseDirectory("caseInfo"), varName);
        }
    };
}

#endif