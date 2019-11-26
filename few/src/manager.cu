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
#include <complex>
#include "cuComplex.h"

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
    fod *flatten_weight_matrix, fod *flattened_bias_matrix,
    std::complex<float>*transform_matrix, int trans_dim1_, int trans_dim2_, fod transform_factor_)
{
    time_batch_size = time_batch_size_;
    num_layers = num_layers_;
    dim1 = dim1_;
    dim2 = dim2_;

    trans_dim1 = trans_dim1_;
    trans_dim2 = trans_dim2_;
    transform_factor = transform_factor_;

    d_layers_matrix = new fod*[num_layers];
    d_layers_bias = new fod*[num_layers];


    int start_int_weights = 0;
    int start_int_bias = 0;
    for (int i=0; i<num_layers; i++){
      gpuErrchk(cudaMalloc(&(d_layers_matrix[i]), dim1[i]*dim2[i]*sizeof(fod)));
      gpuErrchk(cudaMemcpy(d_layers_matrix[i], &flatten_weight_matrix[start_int_weights], dim1[i]*dim2[i]*sizeof(fod), cudaMemcpyHostToDevice));

      gpuErrchk(cudaMalloc(&d_layers_bias[i], dim2[i]*sizeof(fod)));
      gpuErrchk(cudaMemcpy(d_layers_bias[i], &flattened_bias_matrix[start_int_bias], dim2[i]*sizeof(fod), cudaMemcpyHostToDevice));

      start_int_weights += dim1[i]*dim2[i];
      start_int_bias += dim2[i];
    }

    gpuErrchk(cudaMalloc(&d_transform_matrix, trans_dim1*trans_dim2*sizeof(cuDoubleComplex)));
    gpuErrchk(cudaMemcpy(d_transform_matrix, transform_matrix, trans_dim1*trans_dim2*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

}


FastEMRIWaveforms::~FastEMRIWaveforms()
{
    for (int i=0; i<num_layers; i++){
      gpuErrchk(cudaFree(d_layers_matrix[i]));
      gpuErrchk(cudaFree(d_layers_bias[i]));
    }
    gpuErrchk(cudaFree(d_transform_matrix));
    delete[] d_layers_matrix;
    delete[] d_layers_bias;
}
