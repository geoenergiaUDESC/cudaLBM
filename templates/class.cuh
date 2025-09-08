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
Authors: [Your Name] (Geoenergia Lab, UDESC)

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
    [Brief description of the purpose and contents of this file]

Namespace
    LBM

SourceFiles
    [YourFileName].cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_[YOUR_FILENAME_UPPERCASE] _CUH
#define __MBLBM_ [YOUR_FILENAME_UPPERCASE] _CUH

#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"

// Add additional includes as needed
// #include "path/to/other/dependencies.cuh"

namespace LBM
{
    /**
     * @class [YourClassName]
     * @brief [Brief description of the class]
     *
     * @details [Detailed description of the class functionality]
     */
    class[YourClassName]
    {
    public:
        /**
         * @brief Default constructor
         */
        [YourClassName]() = default;

        /**
         * @brief Parameterized constructor
         * @param[in] param1 Description of first parameter
         * @param[in] param2 Description of second parameter
         */
        [YourClassName](ParamType1 param1, ParamType2 param2);

        /**
         * @brief Destructor
         */
        ~[YourClassName]() = default;

        /**
         * @brief Copy constructor
         * @param[in] other Object to copy from
         */
        [YourClassName](const[YourClassName] & other);

        /**
         * @brief Move constructor
         * @param[in] other Object to move from
         */
        [YourClassName]([YourClassName] &&other) noexcept;

        /**
         * @brief Copy assignment operator
         * @param[in] other Object to copy from
         * @return Reference to this object
         */
        [YourClassName] &operator=(const[YourClassName] & other);

        /**
         * @brief Move assignment operator
         * @param[in] other Object to move from
         * @return Reference to this object
         */
        [YourClassName] &operator=([YourClassName] && other) noexcept;

        /**
         * @brief Example member function
         * @param[in] input Description of input parameter
         * @return Description of return value
         */
        [[nodiscard]] ReturnType exampleFunction(InputType input) const;

        /**
         * @brief Example CUDA device function
         * @param[in] input Description of input parameter
         * @return Description of return value
         */
        __device__ [[nodiscard]] DeviceReturnType deviceFunction(DeviceInputType input) const;

    private:
        // Member variables with brief descriptions
        MemberType member1_; ///< Description of member1
        MemberType member2_; ///< Description of member2

        /**
         * @brief Example private helper function
         * @param[in] input Description of input parameter
         */
        void helperFunction(HelperInputType input);
    };

    // Optional: Non-member functions related to the class
    /**
     * @brief Example non-member function
     * @param[in] obj Instance of [YourClassName]
     * @param[in] param Additional parameter
     * @return Description of return value
     */
    [[nodiscard]] ReturnType nonMemberFunction(const[YourClassName] & obj, ParamType param);

} // namespace LBM

#endif // __MBLBM_[YOUR_FILENAME_UPPERCASE]_CUH