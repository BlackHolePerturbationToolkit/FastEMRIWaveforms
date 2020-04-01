#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "global.h"
#include "cuComplex.h"
#include <complex>
#include "interpolate.hh"
#include "kernel.hh"

class FastEMRIWaveforms {
  // pointer to the GPU memory where the array is stored
  int time_batch_size;
  int num_layers;
  fod ** d_layers_matrix;
  fod ** d_layers_bias;
  cuDoubleComplex * d_transform_matrix;
  cuDoubleComplex d_transform_factor_inv;
  int * dim1;
  int * dim2;
  int trans_dim1;
  int trans_dim2;
  fod transform_factor;
  int dim_max;
  int break_index;
  int num_teuk_modes;
  int *d_m, *d_n, *d_l;
  cuDoubleComplex *d_Ylms;
  cmplx *Ylms;

  FilterContainer *filter;

  int *d_mode_keep_inds, *d_filter_modes_buffer;

  int max_init_len;
  int num_n, num_l_m;
  int *m_arr, *l_arr;
  int max_input_len;
  double delta_t, int_err;
  fod *temp_t, *temp_p, *temp_e, *temp_Phi_phi, *temp_Phi_r;
  fod *d_init_t, *d_init_p, *d_init_e, *d_init_Phi_phi, *d_init_Phi_r, *d_input_mat;
  InterpContainer *d_interp_p, *d_interp_e, *d_interp_Phi_phi, *d_interp_Phi_r;
  InterpContainer *h_interp_p, *h_interp_e, *h_interp_Phi_phi, *h_interp_Phi_r;
  InterpContainer *h_interp_modes, *d_interp_modes;

  fod *d_C, *d_Phi_phi, *d_Phi_r, *d_p, *d_e;
  cuDoubleComplex *d_nn_output_mat, *d_teuk_modes, *d_waveform;

  InterpClass *interp;

public:

  FastEMRIWaveforms(int time_batch_size_, int num_layers_, int *dim1_, int *dim2_,
      fod *flatten_weight_matrix, fod *flattened_bias_matrix,
    cmplx*transform_matrix, int trans_dim1_, int trans_dim2_, fod transform_factor_,
    int break_index_,
    int *d_l, int *m_, int *n_,
    int max_input_len, int num_l_m_, int num_n_, fod delta_t_,
    int max_init_len_, double int_err_, fod tol_); // constructor (copies to GPU)

  void run_nn(cmplx *waveform, double M, double mu, double p0, double e0, fod theta, fod phi, int* out_len);

  ~FastEMRIWaveforms(); // destructor
};

#endif //__MANAGER_H__
