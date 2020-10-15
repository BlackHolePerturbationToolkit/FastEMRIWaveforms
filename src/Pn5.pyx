import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool

from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/Inspiral5PN.hh":
    cdef cppclass Pn5CarrierWrap "Pn5Carrier":
        Pn5CarrierWrap()
        void dealloc()

        void Pn5Wrapper(np.float64_t *t, np.float64_t *p,
                         np.float64_t *e, np.float64_t *Y,
                         np.float64_t *Phi_phi,
                         np.float64_t *Phi_theta, np.float64_t *Phi_r,
                         np.float64_t M,
                          np.float64_t mu, np.float64_t a, np.float64_t p0,
                          np.float64_t e0,  np.float64_t Y0,
                          np.float64_t Phi_phi0, np.float64_t Phi_theta0,
                          np.float64_t Phi_r0,
                          int* length,
                          double tmax,
                          double dt,
                          np.float64_t err,
                          int  DENSE_STEPPING,
                          bool use_rk4,
                          int init_len)



cdef class pyPn5Generator:
    cdef Pn5CarrierWrap *g

    def __cinit__(self):
        self.g = new Pn5CarrierWrap()

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __call__(self, M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, T=1.0, dt=-1, err=1e-10, max_init_len=1000, DENSE_STEPPING=0, use_rk4=False):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] t = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] p = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] e = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Y = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_phi = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_theta = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_r = np.zeros(max_init_len, dtype=np.float64)

        cdef int length

        self.g.Pn5Wrapper(&t[0], &p[0], &e[0], &Y[0], &Phi_phi[0], &Phi_theta[0], &Phi_r[0], M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, &length, T, dt, err, DENSE_STEPPING, use_rk4, max_init_len)

        return (t[:length], p[:length], e[:length], Y[:length], Phi_phi[:length], Phi_theta[:length], Phi_r[:length])
