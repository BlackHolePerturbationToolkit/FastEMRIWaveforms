import numpy as np
cimport numpy as np

from utils.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Amplitude.hh":
    cdef cppclass AmplitudeCarrierWrap "AmplitudeCarrier":
        AmplitudeCarrierWrap(int lmax_, int nmax_)
        void dealloc()


    void Interp2DAmplitude(np.complex128_t *amplitude_out, double *p_arr, double *e_arr, int num, AmplitudeCarrierWrap *amps_carrier);



cdef class pyAmplitudeCarrier:
    cdef AmplitudeCarrierWrap *g

    def __cinit__(self, lmax, nmax):
        self.g = new AmplitudeCarrierWrap(lmax, nmax)

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    @property
    def ptr(self):
        return <long int>self.g

@pointer_adjust
def Interp2DAmplitude_wrap(p, e, input_len, num_modes, amps_carrier):

    cdef np.ndarray[ndim=1, dtype=np.complex128_t] amplitude_out = np.zeros((input_len*num_modes), dtype=np.complex128)
    cdef size_t p_in = p
    cdef size_t e_in = e
    cdef size_t amps_carrier_in = amps_carrier

    Interp2DAmplitude(&amplitude_out[0], <double *>p_in, <double *>e_in, input_len, <AmplitudeCarrierWrap *>amps_carrier_in);

    return amplitude_out.reshape(input_len, num_modes)
