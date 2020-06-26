#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <math.h>
#include "global.h"
#include "cublas_v2.h"
#include "elliptic.hh"
#include "kernel.hh"
#include "stdio.h"



void test_layer(long ptr_mat_out, long ptr_mat_in, long ptr_layer_weight, long ptr_bias, int m, int n, int k, int run_bias, int run_activation);

void run_layer(fod *C, fod *C_temp, fod *layer_weight, fod *layer_bias, int dim1, int dim2, int input_len);

void transform_output(cuDoubleComplex *d_teuk_modes, cuDoubleComplex *d_transform_matrix, cuDoubleComplex *d_nn_output_mat, fod *d_C,
                      int input_len, int break_index, cuDoubleComplex d_transform_factor_inv,
                      int num_teuk_modes);

void filter_modes(FilterContainer *filter, cuDoubleComplex *d_teuk_modes, cuDoubleComplex *d_Ylms,
                  int *m_arr, int num_teuk_modes, int length, int num_n, int num_l_m);

void get_waveform(cuDoubleComplex *d_waveform,
              InterpContainer *d_interp_Phi_phi, InterpContainer *d_interp_Phi_r, InterpContainer *d_modes,
              int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cuDoubleComplex *d_Ylms, int num_n,
              fod delta_t, fod *h_t, int num_l_m, FilterContainer *filter, ModeReImContainer * mode_holder);

void ellpe_test();

#endif // __KERNEL_H__
