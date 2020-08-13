#ifndef __AMPLITUDE_H__
#define __AMPLITUDE_H__

// Code to compute an eccentric flux driven insipral
// into a Schwarzschild black hole
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>
#include <algorithm>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <complex>
#include <cmath>

#include "Interpolant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision

#include <omp.h>
#include <stdio.h>

#include "omp.h"
using namespace std;
using namespace std::chrono;

// The 11 below means the lmax = 10
struct waveform_amps{
	Interpolant ***re[11];
	Interpolant ***im[11];
};

class AmplitudeCarrier{
public:
    struct waveform_amps *amps;
    int lmax, nmax;

    AmplitudeCarrier(int lmax_, int nmax_, std::string few_dir);
    void Interp2DAmplitude(std::complex<double> *amplitude_out, double *p_arr, double *e_arr, int *l_arr, int *m_arr, int *n_arr, int num, int num_modes);

    void dealloc();
};

#endif //__AMPLITUDE_H__
