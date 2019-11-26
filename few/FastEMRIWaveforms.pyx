import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass FastEMRIWaveformsWrap "FastEMRIWaveforms":
        FastEMRIWaveformsWrap(int, int, np.int32_t *, np.int32_t *,
             np.float32_t *, np.float32_t *)

cdef class FastEMRIWaveforms:
    cdef FastEMRIWaveformsWrap* g

    def __cinit__(self, time_batch_size, num_layers,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim1,
                    np.ndarray[ndim=1, dtype=np.int32_t] dim2,
                    np.ndarray[ndim=1, dtype=np.float32_t] flattened_weight_matrix,
                    np.ndarray[ndim=1, dtype=np.float32_t] flattened_bias_matrix):
        self.g = new GCEWrap(time_batch_size, num_layers,
                            &dim1[0], &dim2[0], &flattened_weight_matrix[0], &flattened_bias_matrix[0])
