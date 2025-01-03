import numpy as np
cimport numpy as np

from few.utils.utility import wrapper

try: 
    import cupy as cp
    gpu = True
except (ModuleNotFoundError, ImportError) as e:
    gpu = False


assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "matmul.hh":
    ctypedef void* cmplx 'cmplx'
    void neural_layer(double *mat_out, double *mat_in, double *layer_weight, double *bias, int m, int k, int n, int run_relu)
    void transform_output(cmplx *teuk_modes, cmplx *transform_matrix, cmplx *nn_output_mat, double *C,
                          int input_len, int break_index, double transform_factor_inv,
                          int num_teuk_modes)

def neural_layer_wrap(*args, **kwargs):
    # this is a special structure for converting pointers
    targs, tkwargs = wrapper(*args, **kwargs)
    
    mat_out, mat_in, weight, bias, m, k, n, run_relu = targs

    cdef size_t mat_out_in = mat_out
    cdef size_t mat_in_in = mat_in
    cdef size_t weight_in = weight
    cdef size_t bias_in = bias

    neural_layer(<double*> mat_out_in,
                 <double*> mat_in_in,
                 <double*> weight_in,
                 <double*> bias_in,
                 m, k, n, run_relu)

    return

def transform_output_wrap(*args, **kwargs):
    targs, tkwargs = wrapper(*args, **kwargs)
    (teuk_modes, transform_matrix, nn_output_mat, C,
                          input_len, break_index, transform_factor_inv,
                          num_teuk_modes) = targs

    cdef size_t teuk_modes_in = teuk_modes
    cdef size_t transform_matrix_in = transform_matrix
    cdef size_t nn_output_mat_in = nn_output_mat
    cdef size_t C_in = C

    transform_output(<cmplx*>teuk_modes_in, <cmplx*>transform_matrix_in, <cmplx*>nn_output_mat_in, <double *>C_in,
                            input_len, break_index, transform_factor_inv,
                            num_teuk_modes)
