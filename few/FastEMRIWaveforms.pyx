import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass FastEMRIWaveformsWrap "FastEMRIWaveforms":
        FastEMRIWaveformsWrap(int, int, np.int32_t *, np.int32_t *,
             np.float32_t *, np.float32_t *,
             np.complex64_t*, int, int, float, int, np.int32_t*, np.int32_t*, np.int32_t*)

        void run_nn(np.complex64_t*, np.float32_t *, int, np.float32_t *, np.float32_t *)

cdef class FastEMRIWaveforms:
    cdef FastEMRIWaveformsWrap* g

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
                    np.ndarray[ndim=1, dtype=np.int32_t] n):

        self.g = new FastEMRIWaveformsWrap(time_batch_size, num_layers,
                            &dim1[0], &dim2[0], &flattened_weight_matrix[0], &flattened_bias_matrix[0],
                            &transform_matrix[0],
                            trans_dim1, trans_dim2, np.float32(transform_factor), break_index, &l[0], &m[0], &n[0])

    def run_nn(self, np.ndarray[ndim=1, dtype=np.float32_t] input_mat, input_len,
                     np.ndarray[ndim=1, dtype=np.float32_t] Phi_phi,
                     np.ndarray[ndim=1, dtype=np.float32_t] Phi_r):

        cdef np.ndarray[ndim=1, dtype=np.complex64_t] waveform = np.zeros(input_len, dtype=np.complex64)

        self.g.run_nn(&waveform[0], &input_mat[0], input_len, &Phi_phi[0], &Phi_r[0])
        return waveform
