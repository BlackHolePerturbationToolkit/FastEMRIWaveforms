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

    void get_waveform_fd(cmplx *waveform,
               double *interp_array,
               double *special_f_interp_array,
               double* special_f_seg_in,
                int *m_arr_in, int *n_arr_in, int num_teuk_modes, cmplx *Ylms_in,
                double* t_arr, int* start_ind_all, int* end_ind_all, int init_length,
                double start_freq, int* turnover_ind_all,
                double* turnover_freqs, int max_points, double df);

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

@pointer_adjust
def get_waveform_fd_wrap(waveform,
           interp_array,
           special_f_interp_array,
           special_f_seg_in,
           m_arr_in, n_arr_in, num_teuk_modes, Ylms_in,
           t_arr, start_ind_all, end_ind_all, init_length,
           start_freq, turnover_ind_all,
           turnover_freqs, max_points, df):

    cdef size_t waveform_in = waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t special_f_interp_array_in = special_f_interp_array
    cdef size_t special_f_seg_in_in = special_f_seg_in
    cdef size_t m_arr_in_in = m_arr_in
    cdef size_t n_arr_in_in = n_arr_in
    cdef size_t Ylms_in_in = Ylms_in
    cdef size_t t_arr_in = t_arr
    cdef size_t start_ind_all_in = start_ind_all
    cdef size_t end_ind_all_in = end_ind_all
    cdef size_t turnover_ind_all_in = turnover_ind_all
    cdef size_t turnover_freqs_in = turnover_freqs

    get_waveform_fd(<cmplx *>waveform_in,
               <double *>interp_array_in,
               <double *>special_f_interp_array_in,
               <double*> special_f_seg_in_in,
                <int *>m_arr_in_in, <int *>n_arr_in_in, num_teuk_modes, <cmplx *>Ylms_in_in,
                <double*> t_arr_in, <int*> start_ind_all_in, <int*> end_ind_all_in, init_length,
                start_freq, <int*> turnover_ind_all_in,
                <double*> turnover_freqs_in, max_points, df)
