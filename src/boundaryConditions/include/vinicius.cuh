// case normalVector::EAST(): // CORRECT!
// {
//     const scalar_t mxy_I = EAST_mxy_I(pop, inv_rho_I);
//     const scalar_t mxz_I = EAST_mxz_I(pop, inv_rho_I);
//     const scalar_t myy_I = ((pop[label_constant<3>()]) + (pop[label_constant<4>()]) + (pop[label_constant<7>()]) + (pop[label_constant<11>()]) + (pop[label_constant<12>()]) + (pop[label_constant<13>()]) + (pop[label_constant<17>()]) + (pop[label_constant<18>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();
//     const scalar_t myz_I = ((pop[label_constant<11>()]) + (pop[label_constant<12>()]) - (pop[label_constant<17>()]) - (pop[label_constant<18>()])) * inv_rho_I;
//     const scalar_t mzz_I = ((pop[label_constant<5>()]) + (pop[label_constant<6>()]) + (pop[label_constant<9>()]) + (pop[label_constant<11>()]) + (pop[label_constant<12>()]) + (pop[label_constant<15>()]) + (pop[label_constant<17>()]) + (pop[label_constant<18>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();
    
//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho;
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

//     // IRBC-Neumann
//     moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
//     moments[label_constant<5>()] = -((-static_cast<scalar_t>(6) * mxy_I * rho_I + moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxy
//     moments[label_constant<6>()] = -((-static_cast<scalar_t>(6) * mxz_I * rho_I + moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxz
//     moments[label_constant<7>()] = -((-static_cast<scalar_t>(4) * myy_I * rho_I + static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // myy
//     moments[label_constant<8>()] = (myz_I * rho_I) / rho; // myz
//     moments[label_constant<9>()] = -((static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mzz

//     already_handled = true;

//     return;
// }
case normalVector::EAST_FRONT(): // CORRECT! but could mxy,myz be multiplying erroneously?
{
    const scalar_t mxy_I = ((pop[label_constant<7>()]) - (pop[label_constant<13>()])) * inv_rho_I;
    const scalar_t myz_I = ((pop[label_constant<11>()]) - (pop[label_constant<18>()])) * inv_rho_I;

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I - moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz  
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
// case normalVector::WEST(): // CORRECT!
// {
//     const scalar_t mxy_I = WEST_mxy_I(pop, inv_rho_I);
//     const scalar_t mxz_I = WEST_mxz_I(pop, inv_rho_I);
//     const scalar_t myy_I = ((pop[label_constant<3>()]) + (pop[label_constant<4>()]) + (pop[label_constant<8>()]) + (pop[label_constant<11>()]) + (pop[label_constant<12>()]) + (pop[label_constant<14>()]) + (pop[label_constant<17>()]) + (pop[label_constant<18>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();
//     const scalar_t myz_I = ((pop[label_constant<11>()]) + (pop[label_constant<12>()]) - (pop[label_constant<17>()]) - (pop[label_constant<18>()])) * inv_rho_I;
//     const scalar_t mzz_I = ((pop[label_constant<5>()]) + (pop[label_constant<6>()]) + (pop[label_constant<10>()]) + (pop[label_constant<11>()]) + (pop[label_constant<12>()]) + (pop[label_constant<16>()]) + (pop[label_constant<17>()]) + (pop[label_constant<18>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();

//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho;
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];
        
//     // IRBC-Neumann
//     moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
//     moments[label_constant<5>()] = -((-static_cast<scalar_t>(6) * mxy_I * rho_I - moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxy
//     moments[label_constant<6>()] = -((-static_cast<scalar_t>(6) * mxz_I * rho_I - moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxz
//     moments[label_constant<7>()] = -((-static_cast<scalar_t>(4) * myy_I * rho_I + static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // myy
//     moments[label_constant<8>()] = (myz_I * rho_I) / rho; // myz
//     moments[label_constant<9>()] = -((static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mzz

//     already_handled = true;

//     return;
// }
case normalVector::WEST_FRONT(): // CORRECT! but could mxy,myz be multiplying erroneously?
{
    const scalar_t mxy_I = ((pop[label_constant<8>()]) - (pop[label_constant<14>()])) * inv_rho_I;
    const scalar_t myz_I = ((pop[label_constant<11>()]) - (pop[label_constant<18>()])) * inv_rho_I;

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I + moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
// case normalVector::NORTH(): // CORRECT!
// {
//     const scalar_t mxy_I = NORTH_mxy_I(pop, inv_rho_I);
//     const scalar_t myz_I = NORTH_myz_I(pop, inv_rho_I);
//     const scalar_t mxx_I = ((pop[label_constant<1>()]) + (pop[label_constant<2>()]) + (pop[label_constant<7>()]) + (pop[label_constant<9>()]) + (pop[label_constant<10>()]) + (pop[label_constant<14>()]) + (pop[label_constant<15>()]) + (pop[label_constant<16>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();
//     const scalar_t mxz_I = ((pop[label_constant<9>()]) + (pop[label_constant<10>()]) - (pop[label_constant<15>()]) - (pop[label_constant<16>()])) * inv_rho_I;
//     const scalar_t mzz_I = ((pop[label_constant<5>()]) + (pop[label_constant<6>()]) + (pop[label_constant<9>()]) + (pop[label_constant<10>()]) + (pop[label_constant<11>()]) + (pop[label_constant<15>()]) + (pop[label_constant<16>()]) + (pop[label_constant<17>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();

//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho; 
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

//     // IRBC-Neumann
//     moments[label_constant<4>()] = -((-static_cast<scalar_t>(4) * mxx_I * rho_I + static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mxx
//     moments[label_constant<5>()] = -((-static_cast<scalar_t>(6) * mxy_I * rho_I + moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxy
//     moments[label_constant<6>()] = (mxz_I * rho_I) / rho; // mxz
//     moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
//     moments[label_constant<8>()] = -((-static_cast<scalar_t>(6) * myz_I * rho_I + moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho)); // myz
//     moments[label_constant<9>()] = -((static_cast<scalar_t>(4) * mxx_I * rho_I - static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mzz

//     already_handled = true;

//     return;
// }
// case normalVector::NORTH_EAST(): // CORRECT!
// {
//     const scalar_t mxz_I = ((pop[label_constant<9>()]) - (pop[label_constant<15>()])) * inv_rho_I;
//     const scalar_t myz_I = ((pop[label_constant<11>()]) - (pop[label_constant<17>()])) * inv_rho_I;

//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho;
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

//     // IRBC-Neumann
//     moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
//     moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
//     moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
//     moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
//     moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz
//     moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

//     already_handled = true;

//     return;
// }
// case normalVector::NORTH_WEST(): // CORRECT!
// {
//     const scalar_t mxz_I = ((pop[label_constant<10>()]) - (pop[label_constant<16>()])) * inv_rho_I;
//     const scalar_t myz_I = ((pop[label_constant<11>()]) - (pop[label_constant<17>()])) * inv_rho_I;
    
//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho;
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

//     // IRBC-Neumann
//     moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
//     moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
//     moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I + moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
//     moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
//     moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I - moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz
//     moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

//     already_handled = true;

//     return;
// }
case normalVector::NORTH_FRONT(): // CORRECT!
{
    const scalar_t mxy_I = ((pop[label_constant<7>()]) - (pop[label_constant<14>()])) * inv_rho_I;
    const scalar_t mxz_I = ((pop[label_constant<9>()]) - (pop[label_constant<16>()])) * inv_rho_I;

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I - moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy
    moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::NORTH_EAST_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::NORTH_WEST_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);
    
    // Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann 
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
// case normalVector::SOUTH(): // CORRECT!
// {
//     const scalar_t mxy_I = SOUTH_mxy_I(pop, inv_rho_I);
//     const scalar_t myz_I = SOUTH_myz_I(pop, inv_rho_I);
//     const scalar_t mxx_I = ((pop[label_constant<1>()]) + (pop[label_constant<2>()]) + (pop[label_constant<8>()]) + (pop[label_constant<9>()]) + (pop[label_constant<10>()]) + (pop[label_constant<13>()]) + (pop[label_constant<15>()]) + (pop[label_constant<16>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();
//     const scalar_t mxz_I = ((pop[label_constant<9>()]) + (pop[label_constant<10>()]) - (pop[label_constant<15>()]) - (pop[label_constant<16>()])) * inv_rho_I;
//     const scalar_t mzz_I = ((pop[label_constant<5>()]) + (pop[label_constant<6>()]) + (pop[label_constant<9>()]) + (pop[label_constant<10>()]) + (pop[label_constant<12>()]) + (pop[label_constant<15>()]) + (pop[label_constant<16>()]) + (pop[label_constant<18>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();

//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho;
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

//     // IRBC-Neumann
//     moments[label_constant<4>()] = -((-static_cast<scalar_t>(4) * mxx_I * rho_I + static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mxx
//     moments[label_constant<5>()] = -((-static_cast<scalar_t>(6) * mxy_I * rho_I - moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxy
//     moments[label_constant<6>()] = (mxz_I * rho_I) / rho; // mxz
//     moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
//     moments[label_constant<8>()] = -((-static_cast<scalar_t>(6) * myz_I * rho_I - moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho)); // myz
//     moments[label_constant<9>()] = -((static_cast<scalar_t>(4) * mxx_I * rho_I - static_cast<scalar_t>(4) * mzz_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<3>()] * moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mzz

//     already_handled = true;

//     return;
// }
// case normalVector::SOUTH_EAST(): // CORRECT!
// {
//     const scalar_t mxz_I = ((pop[label_constant<9>()]) - (pop[label_constant<15>()])) * inv_rho_I;
//     const scalar_t myz_I = ((pop[label_constant<12>()]) - (pop[label_constant<18>()])) * inv_rho_I;

//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho;
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

//     // IRBC-Neumann
//     moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
//     moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
//     moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
//     moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
//     moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I + moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz
//     moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz                                                                                         

//     already_handled = true;

//     return;
// }
// case normalVector::SOUTH_WEST(): // CORRECT!
// {
//     const scalar_t mxz_I = ((pop[label_constant<10>()]) - (pop[label_constant<16>()])) * inv_rho_I;
//     const scalar_t myz_I = ((pop[label_constant<12>()]) - (pop[label_constant<18>()])) * inv_rho_I;

//     const int3 offset = boundaryNormal.interiorOffset();
//     const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

//     // Neumann
//     const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
//     moments[label_constant<0>()] = rho;
//     moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
//     moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
//     moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

//     // IRBC-Neumann
//     moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
//     moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
//     moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I + moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
//     moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
//     moments[label_constant<8>()] = (static_cast<scalar_t>(6) * myz_I * rho_I + moments[label_constant<3>()] * rho) / (static_cast<scalar_t>(3) * rho); // myz
//     moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz     

//     already_handled = true;

//     return;
// }
case normalVector::SOUTH_FRONT(): // CORRECT!   
{
    const scalar_t mxy_I = ((pop[label_constant<8>()]) - (pop[label_constant<13>()])) * inv_rho_I;
    const scalar_t mxz_I = ((pop[label_constant<9>()]) - (pop[label_constant<16>()])) * inv_rho_I;

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = (static_cast<scalar_t>(6) * mxy_I * rho_I + moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxy  
    moments[label_constant<6>()] = (static_cast<scalar_t>(6) * mxz_I * rho_I - moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho); // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::SOUTH_EAST_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::SOUTH_WEST_FRONT():
{
    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    moments[label_constant<0>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = moments[label_constant<1>()] * moments[label_constant<1>()]; // mxx
    moments[label_constant<5>()] = moments[label_constant<1>()] * moments[label_constant<2>()]; // mxy
    moments[label_constant<6>()] = moments[label_constant<1>()] * moments[label_constant<3>()]; // mxz
    moments[label_constant<7>()] = moments[label_constant<2>()] * moments[label_constant<2>()]; // myy
    moments[label_constant<8>()] = moments[label_constant<2>()] * moments[label_constant<3>()]; // myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}
case normalVector::FRONT(): // CORRECT!
{
    const scalar_t mxz_I = FRONT_mxz_I(pop, inv_rho_I);
    const scalar_t myz_I = FRONT_myz_I(pop, inv_rho_I);
    const scalar_t mxx_I = ((pop[label_constant<1>()]) + (pop[label_constant<2>()]) + (pop[label_constant<7>()]) + (pop[label_constant<8>()]) + (pop[label_constant<9>()]) + (pop[label_constant<13>()]) + (pop[label_constant<14>()]) + (pop[label_constant<16>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();
    const scalar_t mxy_I = ((pop[label_constant<7>()]) + (pop[label_constant<8>()]) - (pop[label_constant<13>()]) - (pop[label_constant<14>()])) * inv_rho_I;
    const scalar_t myy_I = ((pop[label_constant<3>()]) + (pop[label_constant<4>()]) + (pop[label_constant<7>()]) + (pop[label_constant<8>()]) + (pop[label_constant<11>()]) + (pop[label_constant<13>()]) + (pop[label_constant<14>()]) + (pop[label_constant<18>()])) * inv_rho_I - velocitySet::cs2<scalar_t>();

    const int3 offset = boundaryNormal.interiorOffset();
    const label_t tid = device::idxBlock(threadIdx.x + offset.x, threadIdx.y + offset.y, threadIdx.z + offset.z);

    // Neumann
    const scalar_t rho = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<0>()];
    moments[label_constant<0>()] = rho;
    moments[label_constant<1>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<1>()];
    moments[label_constant<2>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<2>()];
    moments[label_constant<3>()] = shared_buffer[tid * (NUMBER_MOMENTS() + 1) + label_constant<3>()];

    // IRBC-Neumann
    moments[label_constant<4>()] = -((-static_cast<scalar_t>(4) * mxx_I * rho_I + static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(6) * rho)); // mxx
    moments[label_constant<5>()] = (mxy_I * rho_I) / rho; // mxy
    moments[label_constant<6>()] = -((-static_cast<scalar_t>(6) * mxz_I * rho_I + moments[label_constant<1>()] * rho) / (static_cast<scalar_t>(3) * rho)); // mxz
    moments[label_constant<7>()] = -((static_cast<scalar_t>(4) * mxx_I * rho_I - static_cast<scalar_t>(4) * myy_I * rho_I - static_cast<scalar_t>(3) * moments[label_constant<1>()] * moments[label_constant<1>()] * rho - static_cast<scalar_t>(3) * moments[label_constant<2>()] * moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(6) * rho)); // myy
    moments[label_constant<8>()] = -((-static_cast<scalar_t>(6) * myz_I * rho_I + moments[label_constant<2>()] * rho) / (static_cast<scalar_t>(3) * rho));// myz
    moments[label_constant<9>()] = moments[label_constant<3>()] * moments[label_constant<3>()]; // mzz

    already_handled = true;

    return;
}