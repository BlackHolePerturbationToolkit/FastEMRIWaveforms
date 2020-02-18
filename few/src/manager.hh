#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "global.h"
#include "cuComplex.h"
#include <complex>

class FastEMRIWaveforms {
  // pointer to the GPU memory where the array is stored
  int time_batch_size;
  int num_layers;
  fod ** d_layers_matrix;
  fod ** d_layers_bias;
  cuComplex * d_transform_matrix;
  cuComplex d_transform_factor_inv;
  int * dim1;
  int * dim2;
  int trans_dim1;
  int trans_dim2;
  fod transform_factor;
  int dim_max;
  int break_index;
  int num_teuk_modes;
  int *d_m, *d_n, *d_l;
  cuComplex *d_Ylms;
  std::complex<float> *Ylms;

  int num_n, num_l_m;
  int *m_arr, *l_arr;

  fod *d_C, *d_Phi_phi, *d_Phi_r;
  cuComplex *d_nn_output_mat, *d_teuk_modes, *d_waveform;

public:

  FastEMRIWaveforms(int time_batch_size_, int num_layers_, int *dim1_, int *dim2_,
      fod *flatten_weight_matrix, fod *flattened_bias_matrix,
    std::complex<float>*transform_matrix, int trans_dim1_, int trans_dim2_, fod transform_factor_,
    int break_index_,
    int *d_l, int *m_, int *n_,
    int max_input_len, int num_l_m_, int num_n_); // constructor (copies to GPU)

  void run_nn(std::complex<float> *waveform, fod *input_mat, int input_len, fod *Phi_phi, fod *Phi_r, fod theta, fod phi);

  ~FastEMRIWaveforms(); // destructor
};

#endif //__MANAGER_H__
