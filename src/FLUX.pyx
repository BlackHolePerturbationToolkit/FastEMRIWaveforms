import numpy as np
cimport numpy as np

from utils.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/FluxInspiral.hh":
    cdef cppclass FluxCarrierWrap "FluxCarrier":
        FluxCarrierWrap()
        void dealloc()

    void FLUXWrapper(np.float64_t *t, np.float64_t *p,
                     np.float64_t *e, np.float64_t *Phi_phi,
                     np.float64_t *amp_norm,
                     np.float64_t *Phi_r, np.float64_t M,
                      np.float64_t mu, np.float64_t p0,
                      np.float64_t e0, int*,
                      double tmax,
                      double dt,
                      FluxCarrierWrap* flux_carrier,
                      np.float64_t err,
                      int  DENSE_STEPPING,
                      double step_eps,
                      int init_len)



cdef class pyFluxCarrier:
    cdef FluxCarrierWrap *g

    def __cinit__(self):
        self.g = new FluxCarrierWrap()

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    @property
    def ptr(self):
        return <long int>self.g

@pointer_adjust
def flux_inspiral(M, mu, p0, e0, flux_carrier, T=1.0, dt=-1, err=1e-10, max_init_len=1000, DENSE_STEPPING=0, step_eps=1e-11):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] t = np.zeros(max_init_len, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] p = np.zeros(max_init_len, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] e = np.zeros(max_init_len, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_phi = np.zeros(max_init_len, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_r = np.zeros(max_init_len, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] amp_norm = np.zeros(max_init_len, dtype=np.float64)

    cdef int length

    cdef size_t flux_carrier_in = flux_carrier

    FLUXWrapper(&t[0], &p[0], &e[0], &Phi_phi[0], &Phi_r[0], &amp_norm[0], M, mu, p0, e0, &length, T, dt, <FluxCarrierWrap*>flux_carrier_in, err, DENSE_STEPPING, step_eps, max_init_len)

    return (t[:length], p[:length], e[:length], Phi_phi[:length], Phi_r[:length], amp_norm[:length])
