#ifndef RECONSTRUCT_CUH
#define RECONSTRUCT_CUH

{
    const scalar_t multiplyTerm_0 = moments[0] * vSet::w_0();
    const scalar_t pics2 = 1.0 - VelocitySet::velocitySet::cs2() * (moments[4] + moments[7] + moments[9]);
    pop[0] = multiplyTerm_0 * (pics2);
    const scalar_t multiplyTerm_1 = moments[0] * vSet::w_1();
    pop[1] = multiplyTerm_1 * (pics2 + moments[1] + moments[4]);
    pop[2] = multiplyTerm_1 * (pics2 - moments[1] + moments[4]);
    pop[3] = multiplyTerm_1 * (pics2 + moments[2] + moments[7]);
    pop[4] = multiplyTerm_1 * (pics2 - moments[2] + moments[7]);
    pop[5] = multiplyTerm_1 * (pics2 + moments[3] + moments[9]);
    pop[6] = multiplyTerm_1 * (pics2 - moments[3] + moments[9]);
    const scalar_t multiplyTerm_2 = moments[0] * vSet::w_2();
    pop[7] = multiplyTerm_2 * (pics2 + moments[1] + moments[2] + moments[4] + moments[7] + moments[5]);
    pop[8] = multiplyTerm_2 * (pics2 - moments[1] - moments[2] + moments[4] + moments[7] + moments[5]);
    pop[9] = multiplyTerm_2 * (pics2 + moments[1] + moments[3] + moments[4] + moments[9] + moments[6]);
    pop[10] = multiplyTerm_2 * (pics2 - moments[1] - moments[3] + moments[4] + moments[9] + moments[6]);
    pop[11] = multiplyTerm_2 * (pics2 + moments[2] + moments[3] + moments[7] + moments[9] + moments[8]);
    pop[12] = multiplyTerm_2 * (pics2 - moments[2] - moments[3] + moments[7] + moments[9] + moments[8]);
    pop[13] = multiplyTerm_2 * (pics2 + moments[1] - moments[2] + moments[4] + moments[7] - moments[5]);
    pop[14] = multiplyTerm_2 * (pics2 - moments[1] + moments[2] + moments[4] + moments[7] - moments[5]);
    pop[15] = multiplyTerm_2 * (pics2 + moments[1] - moments[3] + moments[4] + moments[9] - moments[6]);
    pop[16] = multiplyTerm_2 * (pics2 - moments[1] + moments[3] + moments[4] + moments[9] - moments[6]);
    pop[17] = multiplyTerm_2 * (pics2 + moments[2] - moments[3] + moments[7] + moments[9] - moments[8]);
    pop[18] = multiplyTerm_2 * (pics2 - moments[2] + moments[3] + moments[7] + moments[9] - moments[8]);
}

#endif