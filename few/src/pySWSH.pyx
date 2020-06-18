import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "SWSH.hh":
    void get_spin_weighted_spher_harm(np.complex128_t*harms_out, np.int32_t *l_arr, np.int32_t *m_arr, double theta, double phi, int num)


def get_spin_weighted_spher_harm_wrap(np.ndarray[ndim=1, dtype=np.int32_t] l_arr, np.ndarray[ndim=1, dtype=np.int32_t] m_arr, theta, phi):

    cdef np.ndarray[ndim=1, dtype=np.complex128_t] harms_out = np.zeros((len(l_arr),), dtype=np.complex128)
    get_spin_weighted_spher_harm(&harms_out[0], &l_arr[0], &m_arr[0], theta, phi, len(l_arr))

    return harms_out
