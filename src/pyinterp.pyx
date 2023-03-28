import numpy as np
cimport numpy as np
from libcpp cimport bool

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
              double* turnover_freqs, int max_points, double df, double* f_data, int zero_index);

    void get_waveform_fd(cmplx *waveform,
               double *interp_array,
               double *special_f_interp_array,
               double* special_f_seg_in,
                int *m_arr_in, int *n_arr_in, int num_teuk_modes, cmplx *Ylms_in,
                double* t_arr, int* start_ind_all, int* end_ind_all, int init_length,
                double start_freq, int* turnover_ind_all,
                double* turnover_freqs, int max_points, double df, double* f_data, int zero_index, double* shift_freq, double* slope0_all, double* initial_freqs);

    void interp_time_for_fd_wrap(double* output, double *t_arr, double *tstar, int* ind_tstar, double *interp_array, int ninterps, int length, bool* run)

    void find_segments_fd_wrap(int *segment_out, int *start_inds_seg, int *end_inds_seg, int *mode_start_inds, int num_segments, int num_modes, int max_length);

    void get_waveform_generic_fd(cmplx *waveform,
             double *interp_array,
              int *m_arr_in, int *k_arr_in, int *n_arr_in, int num_teuk_modes,
              double delta_t, double *old_time_arr, int init_length, int data_length,
              double *frequencies, int *mode_start_inds, int *mode_end_inds, int num_segments);

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
           turnover_freqs, max_points, df, f_data, zero_index, shift_freq, slope0_all, initial_freqs):

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
    cdef size_t f_data_in = f_data
    cdef size_t shift_freq_in = shift_freq
    cdef size_t slope0_all_in = slope0_all
    cdef size_t initial_freqs_in = initial_freqs

    get_waveform_fd(<cmplx *>waveform_in,
               <double *>interp_array_in,
               <double *>special_f_interp_array_in,
               <double*> special_f_seg_in_in,
                <int *>m_arr_in_in, <int *>n_arr_in_in, num_teuk_modes, <cmplx *>Ylms_in_in,
                <double*> t_arr_in, <int*> start_ind_all_in, <int*> end_ind_all_in, init_length,
                start_freq, <int*> turnover_ind_all_in,
                <double*> turnover_freqs_in, max_points, df, <double*> f_data_in, zero_index, <double*> shift_freq_in, <double*> slope0_all_in, <double*> initial_freqs_in)

@pointer_adjust
def interp_time_for_fd(output, t_arr, tstar, ind_tstar, interp_array, ninterps, length, run):
    cdef size_t output_in = output
    cdef size_t t_arr_in = t_arr
    cdef size_t tstar_in = tstar
    cdef size_t ind_tstar_in = ind_tstar
    cdef size_t interp_array_in = interp_array
    cdef size_t run_in = run

    interp_time_for_fd_wrap(<double*> output_in, <double*> t_arr_in, <double*> tstar_in, <int*> ind_tstar_in, <double*> interp_array_in, ninterps, length, <bool*> run_in)


@pointer_adjust
def find_segments_fd(segment_out, start_inds_seg, end_inds_seg, mode_start_inds, num_segments, num_modes, max_length):

    cdef size_t segment_out_in = segment_out
    cdef size_t start_inds_seg_in = start_inds_seg
    cdef size_t end_inds_seg_in = end_inds_seg
    cdef size_t mode_start_inds_in = mode_start_inds

    find_segments_fd_wrap(<int *>segment_out_in, <int *>start_inds_seg_in, <int *>end_inds_seg_in, <int *>mode_start_inds_in, num_segments, num_modes, max_length)


@pointer_adjust
def get_waveform_generic_fd_wrap(waveform,
             interp_array,
              m_arr_in, k_arr_in, n_arr_in, num_teuk_modes,
              delta_t, old_time_arr, init_length, data_length,
              frequencies, mode_start_inds, mode_end_inds, num_segments):

    cdef size_t waveform_in = waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t m_arr_in_in = m_arr_in
    cdef size_t k_arr_in_in = k_arr_in
    cdef size_t n_arr_in_in = n_arr_in
    cdef size_t old_time_arr_in = old_time_arr
    cdef size_t frequencies_in = frequencies
    cdef size_t mode_start_inds_in = mode_start_inds
    cdef size_t mode_end_inds_in = mode_end_inds

    get_waveform_generic_fd(<cmplx *>waveform_in,
              <double *>interp_array_in,
              <int *>m_arr_in_in, <int *>k_arr_in_in, <int *>n_arr_in_in, num_teuk_modes,
              delta_t, <double *>old_time_arr_in, init_length, data_length,
              <double *>frequencies_in, <int *>mode_start_inds_in, <int *>mode_end_inds_in, num_segments)