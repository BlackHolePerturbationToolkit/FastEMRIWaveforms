import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool

from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/Inspiral.hh":
    cdef cppclass InspiralCarrierWrap "InspiralCarrier":
        int nparams;
        int num_add_args;
        string func_name;
        InspiralCarrierWrap(ODECarrierWrap* test, string func_name, int nparams, int num_add_args_) except+
        void dealloc() except+
        void add_parameters_to_holder(double M, double mu, double a, double *additional_args) except+
        void initialize_integrator() except+
        void destroy_integrator_information() except+
        void reset_solver() except+
        int take_step(double *t, double *h, double* y, const double tmax) except+
        void get_derivatives(double *ydot_, double *y, int nparams_) except+

cdef extern from "../include/ode.hh":
    cdef cppclass ODECarrierWrap "ODECarrier":
        string few_dir;
        ODECarrierWrap(string func_name_, string few_dir_) except+
        void* func;
        void dealloc() except+
        void get_derivatives(double ydot[], const double y[], double epsilon, double a, double *additional_args) except+

cdef extern from "../include/ode.hh":
    cdef cppclass GetDeriv "ODECarrier":
        GetDeriv(string func, string few_dir)
        void get_derivatives(double ydot[], const double y[], double epsilon, double a, double *additional_args) except+

cdef class pyDerivative:
    cdef GetDeriv *g

    def __cinit__(self, func_name, few_dir):
        self.g = new GetDeriv(func_name.encode(), few_dir)

    def __dealloc__(self):
        if self.g:
            del self.g

    def __call__(self, epsilon, a, np.ndarray[ndim=1, dtype=np.float64_t] y, np.ndarray[ndim=1, dtype=np.float64_t] additional_args):
        
        cdef np.ndarray[ndim=1, dtype=np.float64_t] ydot = np.zeros(self.f.nparams, dtype=np.float64)
        
        self.g.get_derivatives(&ydot[0], &y[0], epsilon, a, &additional_args[0])

        return ydot



cdef class pyInspiralGenerator:
    cdef InspiralCarrierWrap *f
    cdef ODECarrierWrap *g
    cdef public bytes func_name_store
    cdef public bytes few_dir_store
    cdef public int nparams_store
    cdef public int num_add_args_store

    def __cinit__(self, func_name, nparams, num_add_args, few_dir):
        self.func_name_store = func_name
        self.nparams_store = nparams
        self.num_add_args_store = num_add_args
        self.few_dir_store = few_dir

        self.g = new ODECarrierWrap(func_name, few_dir)
        self.f = new InspiralCarrierWrap(self.g, func_name, nparams, num_add_args)

    @property
    def nparams(self):
        cdef int nparams = self.f.nparams
        return nparams

    @property
    def num_add_args(self):
        cdef int num_add_args = self.f.num_add_args
        return num_add_args

    @property
    def few_dir(self):
        cdef string few_dir = self.g.few_dir
        return few_dir

    @property
    def func_name(self):
        cdef string func_name = self.f.func_name
        return func_name

    def take_step(self, t_in, h_in, np.ndarray[ndim=1, dtype=np.float64_t] y_in, tmax):
        cdef double t = t_in
        cdef double h = h_in
        cdef int status

        print("bef", status, t, h)
        status = self.f.take_step(&t, &h, &y_in[0], tmax)
        print("af", status, t, h)
        return (status, t, h)
        
    def add_parameters_to_holder(self, M, mu, a, np.ndarray[ndim=1, dtype=np.float64_t] add_parameters_to_holder):
        self.f.add_parameters_to_holder(M, mu, a, &add_parameters_to_holder[0])

    def initialize_integrator(self):
        self.f.initialize_integrator()

    def destroy_integrator_information(self):
        self.f.destroy_integrator_information()

    def reset_solver(self):
        self.f.reset_solver()

    def get_derivatives(
        self,
        np.ndarray[ndim=1, dtype=np.float64_t] y
    ):
        assert self.nparams == len(y)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] ydot = np.zeros(self.nparams)

        self.f.get_derivatives(&ydot[0], &y[0], self.nparams)
        return ydot

    def __reduce__(self):
        return (rebuild, (self.func_name, self.nparams, self.num_add_args, self.few_dir))

    def __dealloc__(self):
        self.g.dealloc()
        self.f.dealloc()
        if self.f:
            del self.f
        
        if self.g:
            del self.g

def rebuild(func_name, nparams, num_add_args, few_dir):
    c = pyInspiralGenerator(func_name, nparams, num_add_args, few_dir)
    return c
    