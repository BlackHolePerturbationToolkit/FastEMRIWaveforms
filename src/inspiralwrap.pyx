import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool

from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/Inspiral.hh":
    cdef cppclass InspiralCarrierWrap "InspiralCarrier":
        InspiralCarrierWrap(ODECarrierWrap* test, string func_name, bool enforce_schwarz_sep_, int num_add_args_, bool convert_Y_, string few_dir) except+
        void inspiral_wrapper(double *t, double *p, double *e, double *x, double *Phi_phi, double *Phi_theta, double *Phi_r, double M, double mu, double a, double p0, double e0, double x0, double Phi_phi0, double Phi_theta0, double Phi_r0, int *length, double tmax, double dt, double err, int DENSE_STEPPING, bool use_rk4, int init_len, double* additional_args) except+
        void dealloc() except+

cdef extern from "../include/ode.hh":
    cdef cppclass ODECarrierWrap "ODECarrier":
        ODECarrierWrap(string func_name_, string few_dir_);
        void* func;
        void dealloc();
        void get_derivatives(double* pdot, double* edot, double* Ydot,
                            double* Omega_phi, double* Omega_theta, double* Omega_r,
                            double epsilon, double a, double p, double e, double Y, double* additional_args);


cdef class pyInspiralGenerator:
    cdef InspiralCarrierWrap *f
    cdef ODECarrierWrap *g
    cdef public bytes func_name
    cdef public bool enforce_schwarz_sep
    cdef public int num_add_args
    cdef public bool convert_Y
    cdef public bytes few_dir

    def __cinit__(self, func_name, enforce_schwarz_sep, num_add_args, convert_Y, few_dir):
        self.func_name = func_name
        self.enforce_schwarz_sep = enforce_schwarz_sep
        self.num_add_args = num_add_args
        self.convert_Y = convert_Y
        self.few_dir = few_dir

        self.g = new ODECarrierWrap(self.func_name, self.few_dir)
        self.f = new InspiralCarrierWrap(self.g, func_name, enforce_schwarz_sep, num_add_args, convert_Y, few_dir)

    def __reduce__(self):
        return (rebuild, (self.func_name, self.enforce_schwarz_sep, self.num_add_args, self.convert_Y, self.few_dir,))

    def __dealloc__(self):
        self.g.dealloc()
        self.f.dealloc()
        if self.f:
            del self.f
        
        if self.g:
            del self.g

    def __call__(self, M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, np.ndarray[ndim=1, dtype=np.float64_t] additional_args, T=1.0, dt=-1, err=1e-10, max_init_len=1000, DENSE_STEPPING=0, use_rk4=False):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] t = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] p = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] e = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Y = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_phi = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_theta = np.zeros(max_init_len, dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_r = np.zeros(max_init_len, dtype=np.float64)

        cdef int length

        self.f.inspiral_wrapper(&t[0], &p[0], &e[0], &Y[0], &Phi_phi[0], &Phi_theta[0], &Phi_r[0], M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, &length, T, dt, err, DENSE_STEPPING, use_rk4, max_init_len, &additional_args[0])
        
        return (t[:length], p[:length], e[:length], Y[:length], Phi_phi[:length], Phi_theta[:length], Phi_r[:length])

def rebuild(func_name, enforce_schwarz_sep, num_add_args, convert_Y, few_dir):
    c = pyInspiralGenerator(func_name, enforce_schwarz_sep, num_add_args, convert_Y, few_dir)
    return c
    