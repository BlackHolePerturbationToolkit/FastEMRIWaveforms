// Code for matrix operations for roman neural network in Fast EMRI Waveforms

// Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "stdio.h"
#include "matmul.hh"
#include "cuda_complex.hpp"
#include <chrono>
#include "global.h"

using namespace std;

using namespace std::chrono;

// adjust imports for CUDA
#ifdef __CUDACC__
#include "cublas_v2.h"
#else
#include <gsl/gsl_cblas.h>
#endif

#define NUM_THREADS 256

// activation function
// fixed 0.2 in leaky end
CUDA_CALLABLE_MEMBER double LeakyReLU(double x){
     double out = (x >= 0.0) ? x : 0.2*x;
     return out;
}

// funciton for adding bias and then passing through activation
CUDA_KERNEL
void add_bias_relu(double *C, double *bias, int input_len, int dim2)
{

    // adjust loop boundaries in CUDA
    #ifdef __CUDACC__
    int start1 = blockIdx.x * blockDim.x + threadIdx.x;
    int end1 = input_len;
    int diff1 = blockDim.x * gridDim.x;

    int start2 = blockIdx.y * blockDim.y + threadIdx.y;
    int end2 = dim2;
    int diff2 = blockDim.y * gridDim.y;

    #else

    int start1 = 0;
    int end1 = input_len;
    int diff1 = 1;

    int start2 = 0;
    int end2 = dim2;
    int diff2 = 1;


    #endif
    for (int i = start1;
         i < end1;
         i += diff1)
    {

        for (int j = start2;
          j < end2;
          j += diff2)
        {

            C[input_len*j + i] = LeakyReLU(C[input_len*j + i] + bias[j]);

        }
    }
}

// funciton for adding bias and WITHOUT passing through activation
CUDA_KERNEL
void add_bias(double *C, double *bias, int input_len, int dim2){


    #ifdef __CUDACC__
    int start1 = blockIdx.x * blockDim.x + threadIdx.x;
    int end1 = input_len;
    int diff1 = blockDim.x * gridDim.x;

    int start2 = blockIdx.y * blockDim.y + threadIdx.y;
    int end2 = dim2;
    int diff2 = blockDim.y * gridDim.y;

    #else

    int start1 = 0;
    int end1 = input_len;
    int diff1 = 1;

    int start2 = 0;
    int end2 = dim2;
    int diff2 = 1;


    #endif
    for (int i = start1;
         i < end1;
         i += diff1)
    {

        for (int j = start2;
          j < end2;
          j += diff2)
        {

            C[input_len*j + i] = C[input_len*j + i] + bias[j];
        }
    }
}

// perform matrix calculations in blas for a neural network layer
void neural_layer(double *mat_out, double *mat_in, double *weight, double *bias, int m, int k, int n, int run_relu)
{
    #ifdef __CUDACC__
       cublasHandle_t handle;

       char * status;
       cublasStatus_t stat;
       double alpha = 1.0;
       double beta = 0.0;
       stat = cublasCreate(&handle);
          if (stat != CUBLAS_STATUS_SUCCESS) {
                  printf ("CUBLAS initialization failed\n");
                  exit(0);
              }

        // matrix multiplication
       stat = cublasDgemm(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha,
                              mat_in, m,
                              weight, k,
                              &beta,
                              mat_out, m);

       if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }

      stat = cublasDestroy(handle);
         if (stat != CUBLAS_STATUS_SUCCESS) {
                 printf ("CUBLAS initialization failed\n");
                 exit(0);
             }

    // Add the bias and activate, except in last layer do not activate
     int num_threads = 256;
     int num_blocks = std::ceil((m + num_threads -1)/num_threads);
     dim3 gridDim(num_blocks, n);

     if (run_relu){
         add_bias_relu<<<gridDim, num_threads>>>(mat_out, bias, m, n);
     } else {
         add_bias<<<gridDim, num_threads>>>(mat_out, bias, m, n);
     }
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

    #else

    // perform calculations in cblas
     cblas_dgemm (CblasColMajor,
               CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, mat_in, m, weight, k, 0.0, mat_out, m);

    if (run_relu){
        add_bias_relu(mat_out, bias, m, n);
    } else {
        add_bias(mat_out, bias, m, n);
    }

    #endif
}

// take the output of the neural net and conver it from (re_1,..,re_n, im_1, ..., im_n)
// to imaginary
CUDA_KERNEL
void form_complex_output(cmplx *complex_output, double *nn_output, int input_len, int break_index,
                          double transform_factor_inv){

  cmplx temp(0.0, 0.0);

  #ifdef __CUDACC__
  int start1 = blockIdx.x * blockDim.x + threadIdx.x;
  int end1 = input_len;
  int diff1 = blockDim.x * gridDim.x;

  int start2 = blockIdx.y * blockDim.y + threadIdx.y;
  int end2 = break_index;
  int diff2 = blockDim.y * gridDim.y;

  #else

  int start1 = 0;
  int end1 = input_len;
  int diff1 = 1;

  int start2 = 0;
  int end2 = break_index;
  int diff2 = 1;


  #endif
  for (int i = start1;
       i < end1;
       i += diff1){

   for (int ind = start2;
        ind < end2;
        ind += diff2){

            // break index tells how many real entries or imaginary entries
            temp = cmplx(nn_output[ind*input_len + i], nn_output[(break_index+ind)*input_len + i]);
            complex_output[ind*input_len + i] = temp*transform_factor_inv;
         }
  }
}

// post neural net transform from reduced basis back to full amplitude basis
void transform_output(cmplx *teuk_modes, cmplx *transform_matrix, cmplx *nn_output_mat, double *C,
                      int input_len, int break_index, double transform_factor_inv,
                      int num_teuk_modes){

  int m=input_len, k=break_index, n=num_teuk_modes;
  #ifdef __CUDACC__
  int num_blocks = std::ceil((input_len + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks, break_index);

  // form the complex array of neural net outputs
  form_complex_output<<<gridDim, NUM_THREADS>>>(nn_output_mat, C, input_len, break_index, transform_factor_inv);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());


  char * status;
  cublasHandle_t handle;
  cublasStatus_t stat;
  cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
  stat = cublasCreate(&handle);
     if (stat != CUBLAS_STATUS_SUCCESS) {
             printf ("CUBLAS initialization failed\n");
             exit(0);
         }

  // project back onto amplitude basis
  stat = cublasZgemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n, k,
                         &alpha,
                         (cuDoubleComplex*)nn_output_mat, m,
                         (cuDoubleComplex*)transform_matrix, k,
                         &beta,
                         (cuDoubleComplex*)teuk_modes, m);

   status = _cudaGetErrorEnum(stat);
    cudaDeviceSynchronize();

    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
            exit(0);
        }

   #else

   const cmplx alpha(1.0, 0.0);
   const cmplx beta(0.0, 0.0);

    // form the complex array of neural net outputs
   form_complex_output(nn_output_mat, C, input_len, break_index, transform_factor_inv);

   // transform to amplitude basis
   cblas_zgemm (CblasColMajor,
                  CblasNoTrans, CblasNoTrans, m, n, k,
                  (void*)&alpha, (void*)nn_output_mat, m, (void*)transform_matrix, k, (void*)&beta, (void*)teuk_modes, m);
   #endif
}
