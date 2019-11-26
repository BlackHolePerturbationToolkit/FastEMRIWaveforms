#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "global.h"

class FastEMRIWaveforms {
  // pointer to the GPU memory where the array is stored
  int time_batch_size;
  int num_layers;
  fod ** layers_matrix;
  fod ** layers_bias;
  int * layers_matrix_dim1;
  int * layers_matrix_dim2;

public:

  FastEMRIWaveforms(int time_batch_size_, int num_layers_, int *dim1_, int *dim2_,
      fod *flatten_weight_matrix, fod *flattened_bias_matrix); // constructor (copies to GPU)

  ~FastEMRIWaveforms(); // destructor
};

#endif //__MANAGER_H__
