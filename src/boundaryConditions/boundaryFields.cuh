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
    A class handling the representation of a single field value across all
    boundary regions

Namespace
    LBM

SourceFiles
    boundaryFields.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BOUNDARYFIELDS_CUH
#define __MBLBM_BOUNDARYFIELDS_CUH

namespace LBM
{
    /**
     * @struct boundaryFields
     * @brief Represents a single field value across all boundary regions
     * @tparam VelocitySet Velocity set configuration defining lattice structure
     *
     * This struct provides access to a specific field's value across all
     * boundary regions (North, South, East, West, Front, Back) and the internal field.
     */
    template <class VelocitySet>
    class boundaryFields
    {
    public:
        /**
         * @brief Constructs boundary field values for all regions
         * @param[in] fieldName Name of the field to initialize across all regions
         */
        [[nodiscard]] boundaryFields(const std::string &fieldName)
            : values_{
                  boundaryValue<VelocitySet>(fieldName, "North"),
                  boundaryValue<VelocitySet>(fieldName, "South"),
                  boundaryValue<VelocitySet>(fieldName, "East"),
                  boundaryValue<VelocitySet>(fieldName, "West"),
                  boundaryValue<VelocitySet>(fieldName, "Front"),
                  boundaryValue<VelocitySet>(fieldName, "Back"),
                  boundaryValue<VelocitySet>(fieldName, "internalField")},
              fieldName_(fieldName) {};

        /**
         * @name Region Accessors
         * @brief Provide access to field values for specific boundary regions
         * @return The value of the field in the specified region
         */
        [[nodiscard]] inline constexpr scalar_t North() const noexcept
        {
            return values_[0]();
        }
        [[nodiscard]] inline constexpr scalar_t South() const noexcept
        {
            return values_[1]();
        }
        [[nodiscard]] inline constexpr scalar_t East() const noexcept
        {
            return values_[2]();
        }
        [[nodiscard]] inline constexpr scalar_t West() const noexcept
        {
            return values_[3]();
        }
        [[nodiscard]] inline constexpr scalar_t Front() const noexcept
        {
            return values_[4]();
        }
        [[nodiscard]] inline constexpr scalar_t Back() const noexcept
        {
            return values_[5]();
        }
        [[nodiscard]] inline constexpr scalar_t internalField() const noexcept
        {
            return values_[6]();
        }

        /**
         * @brief Print field values for all regions
         */
        void print() const noexcept
        {
            std::cout << fieldName_ << " boundary values:" << std::endl;

            const std::vector<std::string> fieldNames({"North", "South", "East", "West", "Front", "Back", "Internal"});
            for (std::size_t var = 0; var < fieldNames.size(); var++)
            {
                std::cout << fieldNames[var] << ": " << values_[var]() << std::endl;
            }
        }

    private:
        /**
         * @brief Field values for all regions
         */
        const boundaryValue<VelocitySet> values_[7];

        /**
         * @brief Name of the field
         */
        const std::string &fieldName_;
    };

}

#endif