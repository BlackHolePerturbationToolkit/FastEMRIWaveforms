#include "stdio.h"
#include <gsl/gsl_cblas.h>
#include "matmul.hh"
#include "cuda_complex.hpp"
#include <chrono>
#include "global.h"

using namespace std::chrono;

#ifdef __CUDACC__
#include "cublas_v2.h"
#endif

#define NUM_THREADS 256


CUDA_CALLABLE_MEMBER double LeakyReLU(double x){
     double out = (x >= 0.0) ? x : 0.2*x;
     return out;
}

CUDA_KERNEL
void add_bias_relu(double *C, double *bias, int input_len, int dim2){

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
         i += diff1){

     for (int j = start2;
          j < end2;
          j += diff2){

        C[input_len*j + i] = LeakyReLU(C[input_len*j + i] + bias[j]);

  }
}
}

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
         i += diff1){

     for (int j = start2;
          j < end2;
          j += diff2){

       C[input_len*j + i] = C[input_len*j + i] + bias[j];
  }
}
}


void neural_layer(double *mat_out, double *mat_in, double *weight, double *bias, int m, int k, int n, int run_relu)
{
    //high_resolution_clock::time_point t1 = high_resolution_clock::now();
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
     cblas_dgemm (CblasColMajor,
                    CblasNoTrans, CblasNoTrans, m, n, k,
                    1.0, mat_in, m, weight, k, 0.0, mat_out, m);

    if (run_relu){
        add_bias_relu(mat_out, bias, m, n);
    } else {
        add_bias(mat_out, bias, m, n);
    }

    #endif

    //high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //duration<double> time_span = duration_cast<duration<double> >(t2 - t1);
    //printf("# Computing the inspiral took (%d,%d,%d): %lf\n", m,k,n,time_span.count());
}


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

            temp = cmplx(nn_output[ind*input_len + i], nn_output[(break_index+ind)*input_len + i]);
            complex_output[ind*input_len + i] = temp*transform_factor_inv;
         }
  }
}

void transform_output(cmplx *teuk_modes, cmplx *transform_matrix, cmplx *nn_output_mat, double *C,
                      int input_len, int break_index, double transform_factor_inv,
                      int num_teuk_modes){

  int m=input_len, k=break_index, n=num_teuk_modes;
  #ifdef __CUDACC__
  int num_blocks = std::ceil((input_len + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks, break_index);
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

    if (stat != CUBLAS_STATUS_SUCCESS) {
            exit(0);
        }

   #else

   const cmplx alpha(1.0, 0.0);
   const cmplx beta(0.0, 0.0);

   form_complex_output(nn_output_mat, C, input_len, break_index, transform_factor_inv);
   cblas_zgemm (CblasColMajor,
                  CblasNoTrans, CblasNoTrans, m, n, k,
                  (void*)&alpha, (void*)nn_output_mat, m, (void*)transform_matrix, k, (void*)&beta, (void*)teuk_modes, m);

   #endif

}
