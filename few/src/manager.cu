/*
This is the central piece of code. This file implements a class
that takes data in on the cpu side, copies
it to the gpu, and exposes functions that let
you perform actions with the GPU

This class will get translated into python via cython
*/

#include <kernel.hh>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "global.h"
#include <complex>
#include "cuComplex.h"
#include "elliptic.hh"
#include <boost/math/special_functions/spherical_harmonic.hpp>

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
    std::complex<float>*transform_matrix, int trans_dim1_, int trans_dim2_, fod transform_factor_,
    int break_index_,
    int *l_, int *m_, int *n_,
    int max_input_len, int num_l_m_, int num_n_)
{
    time_batch_size = time_batch_size_;
    num_layers = num_layers_;
    dim1 = dim1_;
    dim2 = dim2_;
    break_index = break_index_;

    trans_dim1 = trans_dim1_;
    trans_dim2 = trans_dim2_;
    transform_factor = transform_factor_;

    num_n = num_n_;
    num_l_m = num_l_m_;

    l_arr = l_;
    m_arr = m_;

    num_teuk_modes = trans_dim2;

    d_transform_factor_inv = make_cuComplex(1./transform_factor, 0.0);

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

    gpuErrchk(cudaMalloc(&d_transform_matrix, trans_dim1*trans_dim2*sizeof(cuComplex)));
    gpuErrchk(cudaMemcpy(d_transform_matrix, transform_matrix, trans_dim1*trans_dim2*sizeof(cuComplex), cudaMemcpyHostToDevice));

    // allocate buffer matrix
    dim_max = 0;
    for (int i=0; i<num_layers; i++){
        if (dim2[i] > dim_max) dim_max = dim2[i];
    }
    if (dim1[0] > dim_max) dim_max = dim1[0];

    gpuErrchk(cudaMalloc(&d_l, num_teuk_modes*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_m, num_teuk_modes*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_n, num_teuk_modes*sizeof(int)));

    gpuErrchk(cudaMemcpy(d_l, l_, num_teuk_modes*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_m, m_, num_teuk_modes*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_n, n_, num_teuk_modes*sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_Ylms, num_l_m*sizeof(cuComplex)));
    Ylms = new std::complex<float>[num_l_m];

    gpuErrchk(cudaMalloc(&d_C, max_input_len*dim_max*sizeof(fod)));

    gpuErrchk(cudaMalloc(&d_Phi_phi, max_input_len*sizeof(fod)));
    gpuErrchk(cudaMalloc(&d_Phi_r, max_input_len*sizeof(fod)));

    int complex_dim = (int)((float) dim2[num_layers - 1]/ 2.0);
    gpuErrchk(cudaMalloc(&d_nn_output_mat, complex_dim*max_input_len*sizeof(cuComplex)));
    gpuErrchk(cudaMalloc(&d_teuk_modes, trans_dim2*max_input_len*sizeof(cuComplex)));
    gpuErrchk(cudaMalloc(&d_waveform, max_input_len*sizeof(cuComplex)));
}

void FastEMRIWaveforms::run_nn(std::complex<float> *waveform, fod *input_mat, int input_len, fod *Phi_phi, fod *Phi_r, fod theta, fod phi){

    gpuErrchk(cudaMemcpy(d_C, input_mat, input_len*dim1[0]*sizeof(fod), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_Phi_phi, Phi_phi, input_len*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Phi_r, Phi_r, input_len*sizeof(fod), cudaMemcpyHostToDevice));

  int l,m;
  for (int i=0; i<num_l_m; i+=1){
        l = l_arr[i*num_n];
        m = m_arr[i*num_n];

        Ylms[i] = boost::math::spherical_harmonic(l, m, theta, phi);
        printf("%d %d, %lf, %lf\n", l , m, Ylms[i].real(), Ylms[i].imag());
  }

  gpuErrchk(cudaMemcpy(d_Ylms, Ylms, num_l_m*sizeof(cuComplex), cudaMemcpyHostToDevice));


    //gpuErrchk(cudaMemcpy(d_waveform, waveform, input_len*sizeof(cuComplex), cudaMemcpyHostToDevice));

    for (int layer_i=0; layer_i<num_layers; layer_i++){
      run_layer(d_C, d_layers_matrix[layer_i], d_layers_bias[layer_i], dim1[layer_i], dim2[layer_i], input_len);
    }

    transform_output(d_teuk_modes, d_transform_matrix, d_nn_output_mat, d_C, input_len, break_index, d_transform_factor_inv, trans_dim2);

    get_waveform(d_waveform, d_teuk_modes, d_Phi_phi, d_Phi_r, d_m, d_n, input_len, num_teuk_modes);

    gpuErrchk(cudaMemcpy(waveform, d_waveform, input_len*sizeof(cuComplex), cudaMemcpyDeviceToHost));

    ellpe_test();
}


FastEMRIWaveforms::~FastEMRIWaveforms()
{
    for (int i=0; i<num_layers; i++){
      gpuErrchk(cudaFree(d_layers_matrix[i]));
      gpuErrchk(cudaFree(d_layers_bias[i]));
    }
    gpuErrchk(cudaFree(d_transform_matrix));
    gpuErrchk(cudaFree(d_l));
    gpuErrchk(cudaFree(d_m));
    gpuErrchk(cudaFree(d_n));
    gpuErrchk(cudaFree(d_Ylms));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_nn_output_mat));
    gpuErrchk(cudaFree(d_teuk_modes));
    gpuErrchk(cudaFree(d_Phi_phi));
    gpuErrchk(cudaFree(d_Phi_r));
    gpuErrchk(cudaFree(d_waveform));
    delete[] d_layers_matrix;
    delete[] d_layers_bias;
    delete[] Ylms;
}
