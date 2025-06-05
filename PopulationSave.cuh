/* write to global pop */
if (INTERFACE_BC_WEST)
{ // w
    ghostInterface.gGhost.X_0[idxPopX<0>(ty, tz, bx, by, bz)] = pop[2];
    ghostInterface.gGhost.X_0[idxPopX<1>(ty, tz, bx, by, bz)] = pop[8];
    ghostInterface.gGhost.X_0[idxPopX<2>(ty, tz, bx, by, bz)] = pop[10];
    ghostInterface.gGhost.X_0[idxPopX<3>(ty, tz, bx, by, bz)] = pop[14];
    ghostInterface.gGhost.X_0[idxPopX<4>(ty, tz, bx, by, bz)] = pop[16];
#ifdef D3Q27
    ghostInterface.gGhost.X_0[idxPopX(ty, tz, 5, bx, by, bz)] = pop[20];
    ghostInterface.gGhost.X_0[idxPopX(ty, tz, 6, bx, by, bz)] = pop[22];
    ghostInterface.gGhost.X_0[idxPopX(ty, tz, 7, bx, by, bz)] = pop[24];
    ghostInterface.gGhost.X_0[idxPopX(ty, tz, 8, bx, by, bz)] = pop[25];
#endif // D3Q27
}
if (INTERFACE_BC_EAST)
{ // e
    ghostInterface.gGhost.X_1[idxPopX<0>(ty, tz, bx, by, bz)] = pop[1];
    ghostInterface.gGhost.X_1[idxPopX<1>(ty, tz, bx, by, bz)] = pop[7];
    ghostInterface.gGhost.X_1[idxPopX<2>(ty, tz, bx, by, bz)] = pop[9];
    ghostInterface.gGhost.X_1[idxPopX<3>(ty, tz, bx, by, bz)] = pop[13];
    ghostInterface.gGhost.X_1[idxPopX<4>(ty, tz, bx, by, bz)] = pop[15];
#ifdef D3Q27
    ghostInterface.gGhost.X_1[idxPopX(ty, tz, 5, bx, by, bz)] = pop[19];
    ghostInterface.gGhost.X_1[idxPopX(ty, tz, 6, bx, by, bz)] = pop[21];
    ghostInterface.gGhost.X_1[idxPopX(ty, tz, 7, bx, by, bz)] = pop[23];
    ghostInterface.gGhost.X_1[idxPopX(ty, tz, 8, bx, by, bz)] = pop[26];
#endif // D3Q27
}

if (INTERFACE_BC_SOUTH)
{ // s
    ghostInterface.gGhost.Y_0[idxPopY<0>(tx, tz, bx, by, bz)] = pop[4];
    ghostInterface.gGhost.Y_0[idxPopY<1>(tx, tz, bx, by, bz)] = pop[8];
    ghostInterface.gGhost.Y_0[idxPopY<2>(tx, tz, bx, by, bz)] = pop[12];
    ghostInterface.gGhost.Y_0[idxPopY<3>(tx, tz, bx, by, bz)] = pop[13];
    ghostInterface.gGhost.Y_0[idxPopY<4>(tx, tz, bx, by, bz)] = pop[18];
#ifdef D3Q27
    ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 5, bx, by, bz)] = pop[20];
    ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 6, bx, by, bz)] = pop[22];
    ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 7, bx, by, bz)] = pop[23];
    ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 8, bx, by, bz)] = pop[26];
#endif // D3Q27
}
if (INTERFACE_BC_NORTH)
{ // n
    ghostInterface.gGhost.Y_1[idxPopY<0>(tx, tz, bx, by, bz)] = pop[3];
    ghostInterface.gGhost.Y_1[idxPopY<1>(tx, tz, bx, by, bz)] = pop[7];
    ghostInterface.gGhost.Y_1[idxPopY<2>(tx, tz, bx, by, bz)] = pop[11];
    ghostInterface.gGhost.Y_1[idxPopY<3>(tx, tz, bx, by, bz)] = pop[14];
    ghostInterface.gGhost.Y_1[idxPopY<4>(tx, tz, bx, by, bz)] = pop[17];
#ifdef D3Q27
    ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 5, bx, by, bz)] = pop[19];
    ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 6, bx, by, bz)] = pop[21];
    ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 7, bx, by, bz)] = pop[24];
    ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 8, bx, by, bz)] = pop[25];
#endif // D3Q27
}

if (INTERFACE_BC_BACK)
{ // b
    ghostInterface.gGhost.Z_0[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[6];
    ghostInterface.gGhost.Z_0[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[10];
    ghostInterface.gGhost.Z_0[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[12];
    ghostInterface.gGhost.Z_0[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[15];
    ghostInterface.gGhost.Z_0[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[17];
#ifdef D3Q27
    ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[20];
    ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[21];
    ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[24];
    ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[26];
#endif // D3Q27
}
if (INTERFACE_BC_FRONT)
{
    ghostInterface.gGhost.Z_1[idxPopZ<0>(tx, ty, bx, by, bz)] = pop[5];
    ghostInterface.gGhost.Z_1[idxPopZ<1>(tx, ty, bx, by, bz)] = pop[9];
    ghostInterface.gGhost.Z_1[idxPopZ<2>(tx, ty, bx, by, bz)] = pop[11];
    ghostInterface.gGhost.Z_1[idxPopZ<3>(tx, ty, bx, by, bz)] = pop[16];
    ghostInterface.gGhost.Z_1[idxPopZ<4>(tx, ty, bx, by, bz)] = pop[18];
#ifdef D3Q27
    ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[19];
    ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[22];
    ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[23];
    ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[25];
#endif // D3Q27
}