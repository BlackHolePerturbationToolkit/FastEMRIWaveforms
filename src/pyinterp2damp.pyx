import numpy as np
cimport numpy as np
from libcpp.string cimport string

from few.utils.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Amplitude.hh":
    cdef cppclass AmplitudeCarrierWrap "AmplitudeCarrier":
        AmplitudeCarrierWrap(int lmax_, int nmax_, string few_dir)
        void dealloc()


    void Interp2DAmplitude(np.complex128_t *amplitude_out, double *p_arr, double *e_arr, int *l_arr, int *m_arr, int *n_arr, int num, int num_modes, AmplitudeCarrierWrap *amps_carrier);



cdef class pyAmplitudeCarrier:
    cdef AmplitudeCarrierWrap *g

    def __cinit__(self, lmax, nmax, few_dir):
        cdef string few_dir_in = str.encode(few_dir)
        self.g = new AmplitudeCarrierWrap(lmax, nmax, few_dir_in)

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    @property
    def ptr(self):
        return <long int>self.g

@pointer_adjust
def Interp2DAmplitude_wrap(p, e, l_arr, m_arr, n_arr, input_len, num_modes, amps_carrier):

    cdef np.ndarray[ndim=1, dtype=np.complex128_t] amplitude_out = np.zeros((input_len*num_modes), dtype=np.complex128)
    cdef size_t p_in = p
    cdef size_t e_in = e
    cdef size_t amps_carrier_in = amps_carrier
    cdef size_t l_arr_in = l_arr
    cdef size_t m_arr_in = m_arr
    cdef size_t n_arr_in = n_arr

    Interp2DAmplitude(&amplitude_out[0], <double *>p_in, <double *>e_in, <int *>l_arr_in, <int *>m_arr_in, <int *>n_arr_in, input_len, num_modes, <AmplitudeCarrierWrap *>amps_carrier_in);

    return amplitude_out.reshape(input_len, num_modes)
