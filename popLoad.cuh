
if (tx == 0)
{ // w
    pop[1] = ghostInterface.fGhost.X_1[idxPopX<0>(ty, tz, bxm1, by, bz)];
    pop[7] = ghostInterface.fGhost.X_1[idxPopX<1>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)];
    pop[9] = ghostInterface.fGhost.X_1[idxPopX<2>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))];
    pop[13] = ghostInterface.fGhost.X_1[idxPopX<3>(typ1, tz, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bz)];
    pop[15] = ghostInterface.fGhost.X_1[idxPopX<4>(ty, tzp1, bxm1, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#ifdef D3Q27
    pop[19] = ghostInterface.fGhost.X_1[idxPopX(tym1, tzm1, 5, bxm1, ((ty == 0) ? bym1 : by), ((tz == 0) ? bzm1 : bz))];
    pop[21] = ghostInterface.fGhost.X_1[idxPopX(tym1, tzp1, 6, bxm1, ((ty == 0) ? bym1 : by), ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[23] = ghostInterface.fGhost.X_1[idxPopX(typ1, tzm1, 7, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), ((tz == 0) ? bzm1 : bz))];
    pop[26] = ghostInterface.fGhost.X_1[idxPopX(typ1, tzp1, 8, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#endif // D3Q27
}
else if (tx == (BLOCK_NX - 1))
{ // e
    pop[2] = ghostInterface.fGhost.X_0[idxPopX<0>(ty, tz, bxp1, by, bz)];
    pop[8] = ghostInterface.fGhost.X_0[idxPopX<1>(typ1, tz, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bz)];
    pop[10] = ghostInterface.fGhost.X_0[idxPopX<2>(ty, tzp1, bxp1, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[14] = ghostInterface.fGhost.X_0[idxPopX<3>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)];
    pop[16] = ghostInterface.fGhost.X_0[idxPopX<4>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))];
#ifdef D3Q27
    pop[20] = ghostInterface.fGhost.X_0[idxPopX(typ1, tzp1, 5, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[22] = ghostInterface.fGhost.X_0[idxPopX(typ1, tzm1, 6, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by), ((tz == 0) ? bzm1 : bz))];
    pop[24] = ghostInterface.fGhost.X_0[idxPopX(tym1, tzp1, 7, bxp1, ((ty == 0) ? bym1 : by), ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[25] = ghostInterface.fGhost.X_0[idxPopX(tym1, tzm1, 8, bxp1, ((ty == 0) ? bym1 : by), ((tz == 0) ? bzm1 : bz))];
#endif // D3Q27
}

if (ty == 0)
{ // s
    pop[3] = ghostInterface.fGhost.Y_1[idxPopY<0>(tx, tz, bx, bym1, bz)];
    pop[7] = ghostInterface.fGhost.Y_1[idxPopY<1>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)];
    pop[11] = ghostInterface.fGhost.Y_1[idxPopY<2>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))];
    pop[14] = ghostInterface.fGhost.Y_1[idxPopY<3>(txp1, tz, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), bym1, bz)];
    pop[17] = ghostInterface.fGhost.Y_1[idxPopY<4>(tx, tzp1, bx, bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#ifdef D3Q27
    pop[19] = ghostInterface.fGhost.Y_1[idxPopY(txm1, tzm1, 5, ((tx == 0) ? bxm1 : bx), bym1, ((tz == 0) ? bzm1 : bz))];
    pop[21] = ghostInterface.fGhost.Y_1[idxPopY(txm1, tzp1, 6, ((tx == 0) ? bxm1 : bx), bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[24] = ghostInterface.fGhost.Y_1[idxPopY(txp1, tzp1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[25] = ghostInterface.fGhost.Y_1[idxPopY(txp1, tzm1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), bym1, ((tz == 0) ? bzm1 : bz))];
#endif // D3Q27
}
else if (ty == (BLOCK_NY - 1))
{ // n
    pop[4] = ghostInterface.fGhost.Y_0[idxPopY<0>(tx, tz, bx, byp1, bz)];
    pop[8] = ghostInterface.fGhost.Y_0[idxPopY<1>(txp1, tz, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), byp1, bz)];
    pop[12] = ghostInterface.fGhost.Y_0[idxPopY<2>(tx, tzp1, bx, byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[13] = ghostInterface.fGhost.Y_0[idxPopY<3>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)];
    pop[18] = ghostInterface.fGhost.Y_0[idxPopY<4>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))];
#ifdef D3Q27
    pop[20] = ghostInterface.fGhost.Y_0[idxPopY(txp1, tzp1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    pop[22] = ghostInterface.fGhost.Y_0[idxPopY(txp1, tzm1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), byp1, ((tz == 0) ? bzm1 : bz))];
    pop[23] = ghostInterface.fGhost.Y_0[idxPopY(txm1, tzm1, 7, ((tx == 0) ? bxm1 : bx), byp1, ((tz == 0) ? bzm1 : bz))];
    pop[26] = ghostInterface.fGhost.Y_0[idxPopY(txm1, tzp1, 8, ((tx == 0) ? bxm1 : bx), byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#endif // D3Q27
}

if (tz == 0)
{ // b
    pop[5] = ghostInterface.fGhost.Z_1[idxPopZ<0>(tx, ty, bx, by, bzm1)];
    pop[9] = ghostInterface.fGhost.Z_1[idxPopZ<1>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)];
    pop[11] = ghostInterface.fGhost.Z_1[idxPopZ<2>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)];
    pop[16] = ghostInterface.fGhost.Z_1[idxPopZ<3>(txp1, ty, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bzm1)];
    pop[18] = ghostInterface.fGhost.Z_1[idxPopZ<4>(tx, typ1, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzm1)];
#ifdef D3Q27
    pop[19] = ghostInterface.fGhost.Z_1[idxPopZ(txm1, tym1, 5, ((tx == 0) ? bxm1 : bx), ((ty == 0) ? bym1 : by), bzm1)];
    pop[22] = ghostInterface.fGhost.Z_1[idxPopZ(txp1, typ1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzm1)];
    pop[23] = ghostInterface.fGhost.Z_1[idxPopZ(txm1, typ1, 7, ((tx == 0) ? bxm1 : bx), ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzm1)];
    pop[25] = ghostInterface.fGhost.Z_1[idxPopZ(txp1, tym1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), ((ty == 0) ? bym1 : by), bzm1)];
#endif // D3Q27
}
else if (tz == (BLOCK_NZ - 1))
{ // f
    pop[6] = ghostInterface.fGhost.Z_0[idxPopZ<0>(tx, ty, bx, by, bzp1)];
    pop[10] = ghostInterface.fGhost.Z_0[idxPopZ<1>(txp1, ty, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bzp1)];
    pop[12] = ghostInterface.fGhost.Z_0[idxPopZ<2>(tx, typ1, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzp1)];
    pop[15] = ghostInterface.fGhost.Z_0[idxPopZ<3>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)];
    pop[17] = ghostInterface.fGhost.Z_0[idxPopZ<4>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)];
#ifdef D3Q27
    pop[20] = ghostInterface.fGhost.Z_0[idxPopZ(txp1, typ1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzp1)];
    pop[21] = ghostInterface.fGhost.Z_0[idxPopZ(txm1, tym1, 6, ((tx == 0) ? bxm1 : bx), ((ty == 0) ? bym1 : by), bzp1)];
    pop[24] = ghostInterface.fGhost.Z_0[idxPopZ(txp1, tym1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), ((ty == 0) ? bym1 : by), bzp1)];
    pop[26] = ghostInterface.fGhost.Z_0[idxPopZ(txm1, typ1, 8, ((tx == 0) ? bxm1 : bx), ((ty == (BLOCK_NY - 1)) ? byp1 : by), bzp1)];
#endif // D3Q27
}