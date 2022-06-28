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
void Zlmkn8_5PNe10_wrap(cmplx *Almkn_out, int *l_all, int *m_all, int *k_all, int *n_all, double *q_all, double *Theta_all, double *tp_all, double *te_all, double *tY_all, double *tWr_all, double *tWth_all, double *tWph_all, int num_modes, int num_points);

#endif // ZLMKN8_5PNE10_H_ 