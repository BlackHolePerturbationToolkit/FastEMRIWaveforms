import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/gpuAAK.hh":
    void get_waveform(double *hI, double *hII, double* interp_array,
                  double M_phys, double mu, double lam, double qS, double phiS, double qK, double phiK, double dist,
                  int nmodes, bool mich,
                  int init_len, int out_len,
                  double delta_t, double *h_t)


@pointer_adjust
def pyWaveform(hI, hII, interp_array,
              M_phys, mu, lam, qS, phiS, qK, phiK, dist,
              nmodes, mich,
              init_len, out_len,
              delta_t, h_t):

    cdef size_t hI_in = hI
    cdef size_t hII_in = hII
    cdef size_t interp_array_in = interp_array
    cdef size_t h_t_in = h_t

    get_waveform(<double*> hI_in, <double*> hII_in, <double*> interp_array_in,
                  M_phys, mu, lam, qS, phiS, qK, phiK, dist,
                  nmodes, mich,
                  init_len, out_len,
                  delta_t, <double *>h_t_in)
