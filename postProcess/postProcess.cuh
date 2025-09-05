#ifndef __MBLBM_POSTPROCESS_CUH
#define __MBLBM_POSTPROCESS_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Calculates the coordinates of the points of a latticeMesh object
         * @param mesh The mesh
         * @return An std::vector of type T containing the latticeMesh object points
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> meshCoordinates(const host::latticeMesh &mesh)
        {
            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            const pointVector &L = mesh.L();

            const label_t numNodes = nx * ny * nz;
            std::vector<T> coords(numNodes * 3);

            for (label_t k = 0; k < nz; ++k)
            {
                for (label_t j = 0; j < ny; ++j)
                {
                    for (label_t i = 0; i < nx; ++i)
                    {
                        const label_t idx = k * ny * nx + j * nx + i;
                        // Do the conversion in double, then cast to the desired type
                        coords[3 * idx + 0] = static_cast<T>((static_cast<double>(L.x) * static_cast<double>(i)) / static_cast<double>(nx - 1));
                        coords[3 * idx + 1] = static_cast<T>((static_cast<double>(L.y) * static_cast<double>(j)) / static_cast<double>(ny - 1));
                        coords[3 * idx + 2] = static_cast<T>((static_cast<double>(L.z) * static_cast<double>(k)) / static_cast<double>(nz - 1));
                    }
                }
            }

            return coords;
        }

        /**
         * @brief Calculates the connectivity of the points of a latticeMesh object
         * @param mesh The mesh
         * @return An std::vector of type label_t containing the latticeMesh object connectivity
         **/
        template <const bool one_based>
        __host__ [[nodiscard]] const std::vector<label_t> meshConnectivity(const host::latticeMesh &mesh)
        {
            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            const label_t numElements = (nx - 1) * (ny - 1) * (nz - 1);

            std::vector<label_t> connectivity(numElements * 8);
            label_t cell_idx = 0;
            constexpr const label_t offset = one_based ? 1 : 0;

            for (label_t k = 0; k < nz - 1; ++k)
            {
                for (label_t j = 0; j < ny - 1; ++j)
                {
                    for (label_t i = 0; i < nx - 1; ++i)
                    {
                        const label_t base = k * nx * ny + j * nx + i;
                        const label_t stride_y = nx;
                        const label_t stride_z = nx * ny;

                        connectivity[cell_idx * 8 + 0] = base + offset;
                        connectivity[cell_idx * 8 + 1] = base + 1 + offset;
                        connectivity[cell_idx * 8 + 2] = base + stride_y + 1 + offset;
                        connectivity[cell_idx * 8 + 3] = base + stride_y + offset;
                        connectivity[cell_idx * 8 + 4] = base + stride_z + offset;
                        connectivity[cell_idx * 8 + 5] = base + stride_z + 1 + offset;
                        connectivity[cell_idx * 8 + 6] = base + stride_z + stride_y + 1 + offset;
                        connectivity[cell_idx * 8 + 7] = base + stride_z + stride_y + offset;
                        ++cell_idx;
                    }
                }
            }

            return connectivity;
        }

        /**
         * @brief Calculates the point offsets of the points of a latticeMesh object
         * @param mesh The mesh
         * @return An std::vector of type label_t containing the latticeMesh object point offsets
         **/
        __host__ [[nodiscard]] const std::vector<label_t> meshOffsets(const host::latticeMesh &mesh)
        {
            const label_t nx = mesh.nx();
            const label_t ny = mesh.ny();
            const label_t nz = mesh.nz();
            const label_t numElements = (nx - 1) * (ny - 1) * (nz - 1);

            std::vector<label_t> offsets(numElements);

            for (label_t i = 0; i < numElements; ++i)
            {
                offsets[i] = (i + 1) * 8;
            }

            return offsets;
        }
    }
}

#include "Tecplot.cuh"
#include "VTU.cuh"

#endif