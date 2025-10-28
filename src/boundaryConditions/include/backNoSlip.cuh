case normalVector::WEST_BACK():
{
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
case normalVector::EAST_BACK():
{
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
case normalVector::SOUTH_BACK():
{
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
case normalVector::NORTH_BACK():
{
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
case normalVector::SOUTH_EAST_BACK():
case normalVector::SOUTH_WEST_BACK():
case normalVector::NORTH_EAST_BACK():
case normalVector::NORTH_WEST_BACK():
{
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