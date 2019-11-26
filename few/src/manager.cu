/*
This is the central piece of code. This file implements a class
that takes data in on the cpu side, copies
it to the gpu, and exposes functions that let
you perform actions with the GPU

This class will get translated into python via cython
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "global.h"

using namespace std;

#define NUM_THREADS 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

FastEMRIWaveforms::FastEMRIWaveforms (int time_batch_size_, int num_layers_, int *dim1_, int *dim2_,
    fod *flatten_weight_matrix, fod *flattened_bias_matrix)
{
    time_batch_size = time_batch_size_;
    num_layers = num_layers_;
    dim1 = dim1_;
    dim2 = dim2_;

    layers_matrix = new (fod *)[num_layers];
    layers_bias = new (fod *)[num_layers];
    layers_matrix_dim1 = new int[num_layers];
    layers_matrix_dim2 = new int[num_layers];

    int start_int_weights = 0;
    int start_int_bias = 0;
    for (int i=0; i<num_layers; i++){
      gpuErrchk(cudaMalloc(&(layers_matrix[i]), dim1*dim2*sizeof(fod)));
      gpuErrchk(cudaMemcpy(layers_matrix[i], &flattened_matrix[start_int_weights], dim1*dim2*sizeof(fod), cudaMemcpyHostToDevice));

      gpuErrchk(cudaMalloc(&(layers_bias[i]), dim2*sizeof(fod)));
      gpuErrchk(cudaMemcpy(layers_bias[i], &flattened_bias_matrix[start_int_bias], dim2*sizeof(fod), cudaMemcpyHostToDevice));

      layers_matrix_dim1[i] = dim1;
      layers_matrix_dim2[i] = dim2;

      start_int_weights += dim1*dim2;
      start_int_bias += dim2;
    }

}


FastEMRIWaveforms::~FastEMRIWaveforms()
{
    for (int i=0; i<num_layers; i++){
      gpuErrchk(cudaFree(layers_matrix[i]));
      gpuErrchk(cudaFree(layers_bias[i]));
    }
    delete[] layers_matrix;
    delete[] layers_bias;
    delete[] layers_matrix_dim1;
    delete[] layers_matrix_dim2;
}
