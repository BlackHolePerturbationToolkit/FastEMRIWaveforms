#ifndef __GPU_AAK_HH__
#define __GPU_AAK_HH__

#include "global.h"

void get_waveform(cmplx *waveform, double* interp_array,
              double M_phys, double S_phys, double mu, double qS, double phiS, double qK, double phiK, double dist,
              int nmodes, bool mich,
              int init_len, int out_len,
              double delta_t, double *h_t);

#endif // __GPU_AAK_HH__
