#ifndef ZLMKN8_5PNE10_H_ 
#define ZLMKN8_5PNE10_H_ 

/*
  Header of the PN Teukolsky amplitude Zlmkn8_5PNe10 (5PNe10, 2 <= ell <= 12)

  8th Nov. 2020; Sis

*/

//! \file Zlmkn8.h

#include "cuda_complex.hpp"
#include "global.h"


// BHPC headers
//#include "inspiral_waveform.h"

// Define type
typedef struct inspiral_orb_PNvar {

    /* PNK = sqrt( 1. - q^2 ); PNYm = Y - 1.0 , PNYp = Y + 1.0 */
    double PNlnv;
    double PNK;
    double PNYm;

    double PNq[11]; /*PNq[n]... = 1, q, q ^2, ..., q ^ 9, q ^ 10*/
    double PNe[11]; /*PNe[n]... = 1, e, e ^ 2, ..., e ^ 9, e ^ 10*/
    double PNv[11]; /*PNv[n]... = v ^ 8, v ^ 9, ..., v ^ 17, v ^ 18*/

    double PNY[13]; /*PNY[n]... = 1, Y, Y ^ 2, ..., Y ^ 9, Y ^ 12*/
    double PNYp[13]; /*PNYp[n]... = 1, Yp, Yp^2 ,..., Yp^11  Yp^12*/
    double PNy[25]; /*PNy[n]... = 1, y, y ^ 2, ..., y ^ 23, y ^ 24 where y = sqrt(1. - Y^2)*/

} inspiral_orb_PNvar; /* Shorthad variables for PN amplitudes `hZ_lmkn8_5PNe10` (up to ell = 12) */

// Define type
typedef struct inspiral_orb_data {
    
    double tt; /* Slow time tilde t */

    double tp;
    double te; 
    double tY; /* orbital parameters [tilde p(tt), tilde e(tt), tilde Y(tt) ] */

    double tWr;
    double tWth; 
    double tWph; /* orbital frequencies tilde W(tt) */

    double Pr; 
    double Pth; 
    double Pph; /* orbital phase Phi(t)*/

} inspiral_orb_data; /* store the splined orbital data */

// Declare prototype 
//void init_orb_PNvar(const double q, inspiral_orb_data* orb, inspiral_orb_PNvar* PN_orb);
void Zlmkn8_5PNe10_wrap(cmplx *Zlmkn_out, int *l_all, int *m_all, int *k_all, int *n_all, double *q_all, double *tp_all, double *te_all, double *tY_all, double *tWr_all, double *tWth_all, double *tWph_all, int num_modes, int num_points);

#endif // ZLMKN8_5PNE10_H_ 