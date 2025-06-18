#include "LBMIncludes.cuh"
#include "LBMTypedefs.cuh"
#include "globalFunctions.cuh"
#include "latticeMesh/latticeMesh.cuh"
#include "strings.cuh"

using namespace LBM;

[[nodiscard]] const nodeTypeArray_t defineMesh(const host::latticeMesh &mesh) noexcept
{
    nodeTypeArray_t nodeTypes(mesh.nPoints(), nodeType::UNDEFINED);

    // Try setting the node types from the mesh
    for (label_t z = 1; z < mesh.nz() - 1; z++)
    {
        for (label_t y = 1; y < mesh.ny() - 1; y++)
        {
            for (label_t x = 1; x < mesh.nx() - 1; x++)
            {
                nodeTypes[blockLabel(x, y, z, mesh)] = nodeType::BULK;
            }
        }
    }

    // Hold x constant along the West and East sides
    for (label_t z = 1; z < mesh.nz() - 1; z++)
    {
        for (label_t y = 1; y < mesh.ny() - 1; y++)
        {
            nodeTypes[blockLabel(mesh.West(), y, z, mesh)] = nodeType::WEST;
            nodeTypes[blockLabel(mesh.East(), y, z, mesh)] = nodeType::EAST;
        }
    }

    // Hold y constant along the North and South sides
    for (label_t z = 1; z < mesh.nz() - 1; z++)
    {
        for (label_t x = 1; x < mesh.nx() - 1; x++)
        {
            nodeTypes[blockLabel(x, mesh.South(), z, mesh)] = nodeType::SOUTH;
            nodeTypes[blockLabel(x, mesh.North(), z, mesh)] = nodeType::NORTH;
        }
    }

    // Hold z constant along the Back and Front sides
    for (label_t y = 1; y < mesh.ny() - 1; y++)
    {
        for (label_t x = 1; x < mesh.nx() - 1; x++)
        {
            nodeTypes[blockLabel(x, y, mesh.Back(), mesh)] = nodeType::BACK;
            nodeTypes[blockLabel(x, y, mesh.Front(), mesh)] = nodeType::FRONT;
        }
    }

    // Hold x and y constant along edges
    for (label_t z = 1; z < mesh.nz() - 1; z++)
    {
        nodeTypes[blockLabel(mesh.West(), mesh.North(), z, mesh)] = nodeType::NORTHWEST;
        nodeTypes[blockLabel(mesh.East(), mesh.North(), z, mesh)] = nodeType::NORTHEAST;
        nodeTypes[blockLabel(mesh.West(), mesh.South(), z, mesh)] = nodeType::SOUTHWEST;
        nodeTypes[blockLabel(mesh.East(), mesh.South(), z, mesh)] = nodeType::SOUTHEAST;
    }

    // Hold x and z constant along edges
    for (label_t y = 1; y < mesh.ny() - 1; y++)
    {
        nodeTypes[blockLabel(mesh.West(), y, mesh.Back(), mesh)] = nodeType::WESTBACK;
        nodeTypes[blockLabel(mesh.West(), y, mesh.Front(), mesh)] = nodeType::WESTFRONT;
        nodeTypes[blockLabel(mesh.East(), y, mesh.Back(), mesh)] = nodeType::EASTBACK;
        nodeTypes[blockLabel(mesh.East(), y, mesh.Front(), mesh)] = nodeType::EASTFRONT;
    }

    // Hold y and z constant along edges
    for (label_t x = 1; x < mesh.nx() - 1; x++)
    {
        nodeTypes[blockLabel(x, mesh.South(), mesh.Back(), mesh)] = nodeType::SOUTHBACK;
        nodeTypes[blockLabel(x, mesh.South(), mesh.Front(), mesh)] = nodeType::SOUTHFRONT;
        nodeTypes[blockLabel(x, mesh.North(), mesh.Back(), mesh)] = nodeType::NORTHBACK;
        nodeTypes[blockLabel(x, mesh.North(), mesh.Front(), mesh)] = nodeType::NORTHFRONT;
    }

    // Handle the corners
    nodeTypes[blockLabel(mesh.West(), mesh.South(), mesh.Back(), mesh)] = nodeType::SOUTHWESTBACK;
    nodeTypes[blockLabel(mesh.West(), mesh.South(), mesh.Front(), mesh)] = nodeType::SOUTHWESTFRONT;
    nodeTypes[blockLabel(mesh.East(), mesh.South(), mesh.Back(), mesh)] = nodeType::SOUTHEASTBACK;
    nodeTypes[blockLabel(mesh.East(), mesh.South(), mesh.Front(), mesh)] = nodeType::SOUTHEASTFRONT;
    nodeTypes[blockLabel(mesh.West(), mesh.North(), mesh.Back(), mesh)] = nodeType::NORTHWESTBACK;
    nodeTypes[blockLabel(mesh.West(), mesh.North(), mesh.Front(), mesh)] = nodeType::NORTHEASTFRONT;
    nodeTypes[blockLabel(mesh.East(), mesh.North(), mesh.Back(), mesh)] = nodeType::NORTHEASTBACK;
    nodeTypes[blockLabel(mesh.East(), mesh.North(), mesh.Front(), mesh)] = nodeType::NORTHEASTFRONT;

    return nodeTypes;
}

[[nodiscard]] inline bool nodeTypeCheck(const nodeTypeArray_t &nodeTypes) noexcept
{
    for (label_t n = 0; n < nodeTypes.size(); n++)
    {
        if (nodeTypes[n] == nodeType::UNDEFINED)
        {
            return false;
        }
    }
    return true;
}

void writeMesh(const nodeTypeArray_t &nodeTypes, const std::string &fileName) noexcept
{
    if (nodeTypeCheck(nodeTypes))
    {
        std::ofstream myFile;
        myFile.open(fileName);

        myFile << "nodeTypes[" << nodeTypes.size() << "]:" << std::endl;
        myFile << "{" << std::endl;
        for (label_t n = 0; n < nodeTypes.size(); n++)
        {
            myFile << "    " << nodeTypes[n] << "\n";
        }
        myFile << "}" << std::endl;
        myFile.close();
    }
    else
    {
        std::cout << "Failed node type check: not writing mesh" << std::endl;
    }
}

int main(void)
{
    const host::latticeMesh mesh(ctorType::NO_READ);

    const nodeTypeArray_t nodeTypes = defineMesh(mesh);

    writeMesh(nodeTypes, "nodeTypes");

    return 0;
}
