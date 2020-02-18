#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <math.h>
#include "global.h"
#include "cublas_v2.h"
#include "elliptic.hh"
#include "kernel.hh"
#include "stdio.h"

#define gpuErrchk_here(ans) { gpuAssert_here((ans), __FILE__, __LINE__); }
inline void gpuAssert_here(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void run_layer(fod *C, fod *layer_weight, fod *layer_bias, int dim1, int dim2, int input_len);

void transform_output(cuComplex *d_teuk_modes, cuComplex *d_transform_matrix, cuComplex *d_nn_output_mat, fod *d_C,
                      int input_len, int break_index, cuComplex d_transform_factor_inv,
                      int num_teuk_modes);

void get_waveform(cuComplex *d_waveform, cuComplex *d_teuk_modes, fod *d_Phi_phi, fod *d_Phi_r,
              int *d_m, int *d_n, int input_len, int num_teuk_modes, cuComplex *d_Ylms, int num_n);

void ellpe_test();

#endif // __KERNEL_H__
