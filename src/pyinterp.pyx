import numpy as np
cimport numpy as np

from few.utils.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "interpolate.hh":
    ctypedef void* cmplx 'cmplx'
    void interpolate_arrays(double *t_arr, double *y_all, double *c1, double *c2, double *c3, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag)

    void get_waveform(cmplx *d_waveform, double *y_vals, double *c1, double *c2, double *c3,
                  int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t)


@pointer_adjust
def interpolate_arrays_wrap(t_arr, y_all, c1, c2, c3, ninterps, length, B, upper_diag, diag, lower_diag):

    cdef size_t t_arr_in = t_arr
    cdef size_t y_all_in = y_all
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag

    interpolate_arrays(<double *>t_arr_in, <double *>y_all_in, <double *>c1_in, <double *>c2_in, <double *>c3_in, ninterps, length, <double *>B_in, <double *>upper_diag_in, <double *>diag_in, <double *>lower_diag_in)

@pointer_adjust
def get_waveform_wrap(d_waveform, y_vals, c1, c2, c3,
              d_m, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t):

    cdef size_t d_waveform_in = d_waveform
    cdef size_t y_vals_in = y_vals
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t d_m_in = d_m
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t


    get_waveform(<cmplx *>d_waveform_in, <double *>y_vals_in, <double *>c1_in, <double *>c2_in, <double *>c3_in,
                <int *>d_m_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <cmplx *>d_Ylms_in,
                delta_t, <double *>h_t_in)
