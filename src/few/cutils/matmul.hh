#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "stdlib.h"
#include "global.h"
#include "cuda_complex.hpp"

#ifdef __CUDACC__
#include "cublas_v2.h"

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
#endif

void neural_layer(double *mat_out, double *mat_in, double *layer_weight, double *bias, int m, int k, int n, int run_relu);

void transform_output(cmplx *teuk_modes, cmplx *transform_matrix, cmplx *nn_output_mat, double *C,
                      int input_len, int break_index, double transform_factor_inv,
                      int num_teuk_modes);
#endif // __MATMUL_H__
