switch (boundaryNormal.nodeType())
{
    // Static corners
    case normalVector::SOUTH_WEST_BACK():
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
    case normalVector::SOUTH_WEST_FRONT():
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
    case normalVector::SOUTH_EAST_BACK():
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
    case normalVector::SOUTH_EAST_FRONT():
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
    case normalVector::NORTH_WEST_FRONT():
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
    case normalVector::NORTH_EAST_BACK():
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
    case normalVector::NORTH_EAST_FRONT():
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

    // Static edges
    case normalVector::SOUTH_WEST():
    {
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
    case normalVector::FRONT():
    {
        const scalar_t mxz_I = FRONT_mxz_I(pop, inv_rho_I);
        const scalar_t myz_I = FRONT_myz_I(pop, inv_rho_I);

        const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
        const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
        const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

        moments(label_constant<0>()) = rho;
        moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
        moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
        moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
        moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
        moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
        moments(label_constant<6>()) = mxz;                      // mxz
        moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
        moments(label_constant<8>()) = myz;                      // myz
        moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz

        return;
    }
    case normalVector::NORTH():
    {
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

    // Inflow boundary
    case normalVector::BACK():
    {
        const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
        const label_t y = threadIdx.y + blockDim.y * blockIdx.y;

        const scalar_t dx = static_cast<scalar_t>(x) - static_cast<scalar_t>(64);
        const scalar_t dy = static_cast<scalar_t>(y) - static_cast<scalar_t>(64);
        const scalar_t R  = static_cast<scalar_t>(10);

        const bool in_jet = (dx*dx + dy*dy) <= (R*R);

        if (in_jet)
        {
            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);

            moments(label_constant<0>()) = rho;
            moments(label_constant<1>()) = static_cast<scalar_t>(0);      // ux
            moments(label_constant<2>()) = static_cast<scalar_t>(0);      // uy
            moments(label_constant<3>()) = device::u_inf;                 // uz
            moments(label_constant<4>()) = static_cast<scalar_t>(0);      // mxx
            moments(label_constant<5>()) = static_cast<scalar_t>(0);      // mxy
            moments(label_constant<6>()) = static_cast<scalar_t>(0);      // mxz
            moments(label_constant<7>()) = static_cast<scalar_t>(0);      // myy
            moments(label_constant<8>()) = static_cast<scalar_t>(0);      // myz
            moments(label_constant<9>()) = device::u_inf * device::u_inf; // mzz
        }
        else
        {
            const scalar_t mxz_I = BACK_mxz_I(pop, inv_rho_I);
            const scalar_t myz_I = BACK_myz_I(pop, inv_rho_I);

            const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
            const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
            const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

            moments(label_constant<0>()) = rho;
            moments(label_constant<1>()) = static_cast<scalar_t>(0); // ux
            moments(label_constant<2>()) = static_cast<scalar_t>(0); // uy
            moments(label_constant<3>()) = static_cast<scalar_t>(0); // uz
            moments(label_constant<4>()) = static_cast<scalar_t>(0); // mxx
            moments(label_constant<5>()) = static_cast<scalar_t>(0); // mxy
            moments(label_constant<6>()) = mxz;                      // mxz
            moments(label_constant<7>()) = static_cast<scalar_t>(0); // myy
            moments(label_constant<8>()) = myz;                      // myz
            moments(label_constant<9>()) = static_cast<scalar_t>(0); // mzz
        }

        return;
    }
}