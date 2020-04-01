import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass FastEMRIWaveformsWrap "FastEMRIWaveforms":
        FastEMRIWaveformsWrap(int, int, np.int32_t *, np.int32_t *,
             np.float64_t *, np.float64_t *,
             np.complex128_t*, int, int, double, int,
             np.int32_t*, np.int32_t*, np.int32_t*, int,
             int num_l_m_, int num_n_, np.float64_t,
             int max_init_len_, np.float64_t int_err, np.float64_t tol)

        void run_nn(np.complex128_t*, np.float64_t M, np.float64_t mu, np.float64_t p0, np.float64_t e0, np.float64_t theta, np.float64_t phi, int* out_len)


cdef extern from "src/kernel.hh":
    void test_layer(long ptr_mat_out, long ptr_mat_in, long ptr_layer_weight, long ptr_bias, int m, int n, int k, int run_bias, int run_activation);


cdef class FastEMRIWaveforms:
    cdef FastEMRIWaveformsWrap* g
    cdef max_wave_len

    def __cinit__(self, time_batch_size, num_layers,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim1,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim2,
                    np.ndarray[ndim=1, dtype=np.float64_t] flattened_weight_matrix,
                    np.ndarray[ndim=1, dtype=np.float64_t] flattened_bias_matrix,
                    np.ndarray[ndim=1, dtype=np.complex128_t] transform_matrix,
                    trans_dim1, trans_dim2,
                    transform_factor,
                    break_index,
                    np.ndarray[ndim=1, dtype=np.int32_t] l,
                    np.ndarray[ndim=1, dtype=np.int32_t] m,
                    np.ndarray[ndim=1, dtype=np.int32_t] n,
                    num_l_m,
                    num_n,
                    max_wave_len,
                    max_init_len,
                    dt,
                    int_err=1e-10,
                    tol=1e-6):

        self.max_wave_len = max_wave_len
        self.g = new FastEMRIWaveformsWrap(time_batch_size, num_layers,
                            &dim1[0], &dim2[0], &flattened_weight_matrix[0], &flattened_bias_matrix[0],
                            &transform_matrix[0],
                            trans_dim1, trans_dim2, np.float64(transform_factor), break_index,
                            &l[0], &m[0], &n[0], max_wave_len, num_l_m, num_n,
                            dt, max_init_len, int_err, tol)

    def run_nn(self, M, mu, p0, e0, theta, phi):

        cdef np.ndarray[ndim=1, dtype=np.complex128_t] waveform = np.zeros(self.max_wave_len, dtype=np.complex128)
        cdef int out_len;

        self.g.run_nn(&waveform[0], M, mu, p0, e0, theta, phi, &out_len)
        return waveform[:out_len]


def cublas_test_layer(mat_out, mat_in, layer_weight, bias, m, n, k, run_bias=1, run_activation=1):

    ptr_mat_out = mat_out.data.mem.ptr
    ptr_mat_in = mat_in.data.mem.ptr
    ptr_layer_weight = layer_weight.data.mem.ptr
    ptr_bias = bias.data.mem.ptr

    test_layer(ptr_mat_out, ptr_mat_in, ptr_layer_weight, ptr_bias, m, n, k, run_bias, run_activation)
