import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from few.utils.utility import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/gpuAAK.hh":
    ctypedef void* cmplx 'cmplx'
    void get_waveform(cmplx* waveform, double* interp_array,
                  double M_phys, double S_phys, double mu, double qS, double phiS, double qK, double phiK, double dist,
                  int nmodes, bool mich,
                  int init_len, int out_len,
                  double delta_t, double *h_t) except+

def pyWaveform(*args, **kwargs):

    targs, tkwargs = wrapper(*args, **kwargs)

    (waveform, interp_array,
              M_phys, S_phys, mu, qS, phiS, qK, phiK, dist,
              nmodes, mich,
              init_len, out_len,
              delta_t, h_t), tmp = targs, tkwargs

    cdef size_t waveform_in = waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t h_t_in = h_t

    get_waveform(<cmplx*> waveform_in, <double*> interp_array_in,
                  M_phys, S_phys, mu, qS, phiS, qK, phiK, dist,
                  nmodes, mich,
                  init_len, out_len,
                  delta_t, <double *>h_t_in)
