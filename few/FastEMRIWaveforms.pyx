import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass FastEMRIWaveformsWrap "FastEMRIWaveforms":
        FastEMRIWaveformsWrap(int, int, np.int32_t *, np.int32_t *,
             np.float32_t *, np.float32_t *,
             np.complex64_t*, int, int, float, int, np.int32_t*, np.int32_t*, np.int32_t*, int, int num_l_m_, int num_n_, np.float32_t)

        void run_nn(np.complex64_t*, np.float64_t p0, np.float64_t e0, np.float32_t theta, np.float32_t phi, int* out_len)

cdef class FastEMRIWaveforms:
    cdef FastEMRIWaveformsWrap* g
    cdef max_input_len

    def __cinit__(self, time_batch_size, num_layers,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim1,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim2,
                    np.ndarray[ndim=1, dtype=np.float32_t] flattened_weight_matrix,
                    np.ndarray[ndim=1, dtype=np.float32_t] flattened_bias_matrix,
                    np.ndarray[ndim=1, dtype=np.complex64_t] transform_matrix,
                    trans_dim1, trans_dim2,
                    transform_factor,
                    break_index,
                    np.ndarray[ndim=1, dtype=np.int32_t] l,
                    np.ndarray[ndim=1, dtype=np.int32_t] m,
                    np.ndarray[ndim=1, dtype=np.int32_t] n,
                    num_l_m,
                    num_n,
                    max_input_len,
                    dt):

        self.max_input_len = max_input_len
        self.g = new FastEMRIWaveformsWrap(time_batch_size, num_layers,
                            &dim1[0], &dim2[0], &flattened_weight_matrix[0], &flattened_bias_matrix[0],
                            &transform_matrix[0],
                            trans_dim1, trans_dim2, np.float32(transform_factor), break_index, &l[0], &m[0], &n[0], max_input_len, num_l_m, num_n, dt)

    def run_nn(self, p0, e0, theta, phi):

        cdef np.ndarray[ndim=1, dtype=np.complex64_t] waveform = np.zeros(self.max_input_len, dtype=np.complex64)
        cdef int out_len;

        self.g.run_nn(&waveform[0], p0, e0, theta, phi, &out_len)
        return waveform[:out_len]
