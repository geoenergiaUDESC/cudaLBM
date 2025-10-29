case normalVector::EAST(): 
case normalVector::EAST_FRONT(): 
case normalVector::WEST(): 
case normalVector::WEST_FRONT(): 
case normalVector::NORTH(): 
case normalVector::NORTH_EAST(): 
case normalVector::NORTH_WEST(): 
case normalVector::NORTH_FRONT():
//case normalVector::NORTH_EAST_FRONT():
//case normalVector::NORTH_WEST_FRONT():
case normalVector::SOUTH(): 
case normalVector::SOUTH_EAST(): 
case normalVector::SOUTH_WEST(): 
case normalVector::SOUTH_FRONT(): 
//case normalVector::SOUTH_EAST_FRONT():
//case normalVector::SOUTH_WEST_FRONT():
case normalVector::FRONT(): 
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    device::constexpr_for<0, NUMBER_MOMENTS()>(
        [&](const auto moment)
        {
            const label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
            moments[moment] = shared_buffer[ID];
        });

    already_handled = true;

    return;
}