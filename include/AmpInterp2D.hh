#ifndef __AMP_INTERP_2D_HH__
#define __AMP_INTERP_2D_HH__

#include "global.h"

#endif // __AMP_INTERP_2D_HH__

void interp2D_wrap(double* z, const double* tx, int nx, const double* ty, int ny, double* c,
             int kx, int ky, const double* x, int mx,
             const double* y, int my, int num_indiv_c, int len_indiv_c);