// Static corners
case normalVector::SOUTH_WEST_BACK():
{
    printOnce(normalVector::SOUTH_WEST_BACK(), "SOUTH_WEST_BACK");

    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_WEST_FRONT():
{
    printOnce(normalVector::SOUTH_WEST_FRONT(), "SOUTH_WEST_FRONT");

    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_EAST_BACK():
{
    printOnce(normalVector::SOUTH_EAST_BACK(), "SOUTH_EAST_BACK");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_EAST_FRONT():
{
    printOnce(normalVector::SOUTH_EAST_FRONT(), "SOUTH_EAST_FRONT");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH_WEST_BACK():
{
    printOnce(normalVector::NORTH_WEST_BACK(), "NORTH_WEST_BACK");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH_WEST_FRONT():
{
    printOnce(normalVector::NORTH_WEST_FRONT(), "NORTH_WEST_FRONT");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH_EAST_BACK():
{
    printOnce(normalVector::NORTH_EAST_BACK(), "NORTH_EAST_BACK");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH_EAST_FRONT():
{
    printOnce(normalVector::NORTH_EAST_FRONT(), "NORTH_EAST_FRONT");
    
    if constexpr (VelocitySet::Q() == 19)
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
    }
    else
    {
        moments(label_constant<0>()) = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
    }
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}

// Static edges
case normalVector::SOUTH_WEST():
{
    printOnce(normalVector::SOUTH_WEST(), "SOUTH_WEST");
    
    const scalar_t mxy_I = SOUTH_WEST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_EAST():
{
    printOnce(normalVector::SOUTH_EAST(), "SOUTH_EAST");
    
    const scalar_t mxy_I = SOUTH_EAST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH_WEST():
{
    printOnce(normalVector::NORTH_WEST(), "NORTH_WEST");
    
    const scalar_t mxy_I = NORTH_WEST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}
case normalVector::NORTH_EAST():
{
    printOnce(normalVector::NORTH_EAST(), "NORTH_EAST");
    
    const scalar_t mxy_I = NORTH_EAST_mxy_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}
case normalVector::WEST_BACK():
{
    printOnce(normalVector::WEST_BACK(), "WEST_BACK");
    
    const scalar_t mxz_I = WEST_BACK_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::WEST_FRONT():
{
    printOnce(normalVector::WEST_FRONT(), "WEST_FRONT");
    
    const scalar_t mxz_I = WEST_FRONT_mxz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_BACK():
{
    printOnce(normalVector::EAST_BACK(), "EAST_BACK");
    
    const scalar_t mxz_I = EAST_BACK_mxz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST_FRONT():
{
    printOnce(normalVector::EAST_FRONT(), "EAST_FRONT");
    
    const scalar_t mxz_I = EAST_FRONT_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_BACK():
{
    printOnce(normalVector::SOUTH_BACK(), "SOUTH_BACK");
    
    const scalar_t myz_I = SOUTH_BACK_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH_FRONT():
{
    printOnce(normalVector::SOUTH_FRONT(), "SOUTH_FRONT");
    
    const scalar_t myz_I = SOUTH_FRONT_myz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH_BACK():
{
    printOnce(normalVector::NORTH_BACK(), "NORTH_BACK");
    
    const scalar_t myz_I = NORTH_BACK_myz_I(pop, inv_rho_I);

    const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}
case normalVector::NORTH_FRONT():
{
    printOnce(normalVector::NORTH_FRONT(), "NORTH_FRONT");
    
    const scalar_t myz_I = NORTH_FRONT_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
    const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
    return;
}

// Static faces
case normalVector::WEST():
{
    printOnce(normalVector::WEST(), "WEST");
    
    const scalar_t mxy_I = WEST_mxy_I(pop, inv_rho_I);
    const scalar_t mxz_I = WEST_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::EAST():
{
    printOnce(normalVector::EAST(), "EAST");
    
    const scalar_t mxy_I = EAST_mxy_I(pop, inv_rho_I);
    const scalar_t mxz_I = EAST_mxz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = mxz;                      // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = static_cast<scalar_t>(0); // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::SOUTH():
{
    printOnce(normalVector::SOUTH(), "SOUTH");
    
    const scalar_t mxy_I = SOUTH_mxy_I(pop, inv_rho_I);
    const scalar_t myz_I = SOUTH_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
case normalVector::NORTH():
{
    printOnce(normalVector::NORTH(), "NORTH");
    
    const scalar_t mxy_I = NORTH_mxy_I(pop, inv_rho_I);
    const scalar_t myz_I = NORTH_myz_I(pop, inv_rho_I);

    const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
    const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
    const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

    moments(label_constant<0>()) = rho;
    moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
    moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
    moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
    moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
    moments(label_constant<5>()) = mxy;                      // mxy
    moments(label_constant<6>()) = static_cast<scalar_t>(0); // mxz
    moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
    moments(label_constant<8>()) = myz;                      // myz
    moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

    return;
}
