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
  cuDoubleComplex * d_transform_matrix;
  int * dim1;
  int * dim2;
  int trans_dim1;
  int trans_dim2;
  fod transform_factor;

public:

  FastEMRIWaveforms(int time_batch_size_, int num_layers_, int *dim1_, int *dim2_,
      fod *flatten_weight_matrix, fod *flattened_bias_matrix,
    std::complex<float>*transform_matrix, int trans_dim1_, int trans_dim2_, fod transform_factor_); // constructor (copies to GPU)

  ~FastEMRIWaveforms(); // destructor
};

#endif //__MANAGER_H__
