import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass FastEMRIWaveformsWrap "FastEMRIWaveforms":
        FastEMRIWaveformsWrap(int, int, np.int32_t *, np.int32_t *,
             np.float32_t *, np.float32_t *,
             np.complex64_t*, int, int, float)

cdef class FastEMRIWaveforms:
    cdef FastEMRIWaveformsWrap* g

    def __cinit__(self, time_batch_size, num_layers,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim1,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim2,
                    np.ndarray[ndim=1, dtype=np.float32_t] flattened_weight_matrix,
                    np.ndarray[ndim=1, dtype=np.float32_t] flattened_bias_matrix,
                    np.ndarray[ndim=1, dtype=np.complex64_t] transform_matrix,
                    trans_dim1, trans_dim2,
                    transform_factor):

        self.g = new FastEMRIWaveformsWrap(time_batch_size, num_layers,
                            &dim1[0], &dim2[0], &flattened_weight_matrix[0], &flattened_bias_matrix[0],
                            &transform_matrix[0],
                            trans_dim1, trans_dim2, np.float32(transform_factor))
