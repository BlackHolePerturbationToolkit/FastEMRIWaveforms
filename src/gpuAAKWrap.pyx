import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/gpuAAK.hh":
    ctypedef void* cmplx 'cmplx'
    void get_waveform(cmplx* waveform, double* interp_array,
                  double M_phys, double S_phys, double mu, double qS, double phiS, double qK, double phiK, double dist,
                  int nmodes, bool mich,
                  int init_len, int out_len,
                  double delta_t, double *h_t)


@pointer_adjust
def pyWaveform(waveform, interp_array,
              M_phys, S_phys, mu, qS, phiS, qK, phiK, dist,
              nmodes, mich,
              init_len, out_len,
              delta_t, h_t):

    cdef size_t waveform_in = waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t h_t_in = h_t

    get_waveform(<cmplx*> waveform_in, <double*> interp_array_in,
                  M_phys, S_phys, mu, qS, phiS, qK, phiK, dist,
                  nmodes, mich,
                  init_len, out_len,
                  delta_t, <double *>h_t_in)
