#ifndef __SWSH__
#define __SWSH__

#include <complex>


void get_spin_weighted_spher_harm(std::complex<double>*harms_out, int *l_arr, int *m_arr, double theta, double phi, int num);

//std::complex<double> SpinWeightedSphericalHarmonic (int l, int m, double theta, double phi);

#endif  // __SWSH__
