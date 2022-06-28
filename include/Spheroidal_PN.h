#ifndef SPHEROIDAL_PN_H_ 
#define SPHEROIDAL_PN_H_ 

/*

  Header of the PN Spheroidal harmonics

  8th Nov. 2020; Sis (updated for 4.5PN = O(w^4) data)

*/

//! \file Spheroidal.h

#include "global.h"

// Define type
typedef struct Slm_aw_PNvar { // see datails for BHPC's Maple: outputC_Slm_Zlmkn.mw (by Sis)

    /* xtmp = X = cos(Theta);
         1. - X = Xm; 1. + X = Xp; Xfactor =sqrt( Xm * Xp )  */

    double Xfactor;

    double Sq[5]; /*PNq[n]... = 1, q, q ^2, q^3, q^4*/
    double Sw[5]; /*Sw[n]... = 1, w, w^2, w^3, w^4*/
    
    double Xp[6]; /*Xp[n]... = 1, Xp, Xp^ 2, ..., Xp^5*/
    double X[17]; /*X[n]... = 1, X, X ^ 2, ..., X ^ 15, X^ 16*/
    
} Slm_aw_PNvar; /* Shorthad variables for PN spheroidal harmonics `Slm_aw` (up to ell = 12, w^4) */



// Declare prototype 
CUDA_CALLABLE_MEMBER
void init_PNSlm(const double q, const double Theta, Slm_aw_PNvar* PN_Slm, const int flag);
CUDA_CALLABLE_MEMBER
double Slm_aw (const int l, const int m, Slm_aw_PNvar* PN_Slm);

#endif // SPHEROIDAL_PN_H_ 