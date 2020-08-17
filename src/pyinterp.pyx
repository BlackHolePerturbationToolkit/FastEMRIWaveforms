import numpy as np
cimport numpy as np

from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "interpolate.hh":
    ctypedef void* cmplx 'cmplx'
    void interpolate_arrays(double *t_arr, double *interp_array, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag)

    void get_waveform(cmplx *d_waveform, double *interp_array,
                  int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t)


@pointer_adjust
def interpolate_arrays_wrap(t_arr, interp_array, ninterps, length, B, upper_diag, diag, lower_diag):

    cdef size_t t_arr_in = t_arr
    cdef size_t interp_array_in = interp_array
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag

    interpolate_arrays(<double *>t_arr_in, <double *>interp_array_in, ninterps, length, <double *>B_in, <double *>upper_diag_in, <double *>diag_in, <double *>lower_diag_in)

@pointer_adjust
def get_waveform_wrap(d_waveform, interp_array,
              d_m, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t):

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t d_m_in = d_m
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform(<cmplx *>d_waveform_in, <double *>interp_array_in,
                <int *>d_m_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <cmplx *>d_Ylms_in,
                delta_t, <double *>h_t_in)
