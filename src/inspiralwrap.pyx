import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool as bool_c

from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/Inspiral.hh":
    cdef cppclass InspiralCarrierWrap "InspiralCarrier":
        int nparams;
        int num_add_args;
        string func_name;
        InspiralCarrierWrap(int nparams, int num_add_args_) except+
        void dealloc() except+
        void add_parameters_to_holder(double M, double mu, double a, double *additional_args) except+
        void set_error_tolerance(double err_set) except+
        void initialize_integrator() except+
        void destroy_integrator_information() except+
        void reset_solver() except+
        int take_step(double *t, double *h, double* y, const double tmax) except+
        void get_derivatives(double *ydot_, double *y, int nparams_) except+
        int get_currently_running_ode_index() except+
        void update_currently_running_ode_index(int currently_running_ode_index) except+
        int get_number_of_odes() except+
        void add_ode(string func_name, string few_dir) except+
        void get_backgrounds(int *backgrounds, int num_odes) except+
        void get_equatorial(bool_c *equatorial, int num_odes) except+
        void get_circular(bool_c *circular, int num_odes) except+
        void get_integrate_constants_of_motion(bool_c *integrate_constants_of_motion, int num_odes) except+
        void get_integrate_phases(bool_c *integrate_phases, int num_odes) except+

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
    cdef public int nparams_store
    cdef public int num_add_args_store

    def __cinit__(self, nparams, num_add_args):
        self.nparams_store = nparams
        self.num_add_args_store = num_add_args
        
        self.f = new InspiralCarrierWrap(nparams, num_add_args)
    
    @property
    def nparams(self):
        cdef int nparams = self.f.nparams
        return nparams

    @property
    def num_add_args(self):
        cdef int num_add_args = self.f.num_add_args
        return num_add_args

    #@property
    #def few_dir(self):
    #    cdef string few_dir = self.few_dir_store
    #    return few_dir

    #@property
    #def func_name(self):
    #    cdef string func_name = self.f.func_name
    #    return func_name

    @property
    def currently_running_ode_index(self):
        return self.f.get_currently_running_ode_index()

    @currently_running_ode_index.setter
    def currently_running_ode_index(self, currently_running_ode_index: int):
        self.f.update_currently_running_ode_index(currently_running_ode_index)

    @property
    def num_odes(self):
        return self.f.get_number_of_odes()

    def add_ode(self, func_name, few_dir):
        self.f.add_ode(func_name, few_dir)

    def set_error_tolerance(self, err_set):
        self.f.set_error_tolerance(err_set)

    def take_step(self, t_in, h_in, np.ndarray[ndim=1, dtype=np.float64_t] y_in, tmax):
        cdef double t = t_in
        cdef double h = h_in
        cdef int status

        status = self.f.take_step(&t, &h, &y_in[0], tmax)
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

    @property
    def backgrounds(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] backgrounds = np.zeros(self.num_odes, dtype=np.int32)

        self.f.get_backgrounds(<int*>&backgrounds[0], self.num_odes)
        return backgrounds

    @property
    def equatorial(self):
        cdef np.ndarray[ndim=1, dtype=bool_c] equatorial = np.zeros(self.num_odes, dtype=bool)

        self.f.get_equatorial(&equatorial[0], self.num_odes)
        return equatorial

    @property
    def circular(self):
        cdef np.ndarray[ndim=1, dtype=bool_c] circular = np.zeros(self.num_odes, dtype=bool)

        self.f.get_circular(&circular[0], self.num_odes)
        return circular

    @property
    def integrate_constants_of_motion(self):
        cdef np.ndarray[ndim=1, dtype=bool_c] integrate_constants_of_motion = np.zeros(self.num_odes, dtype=bool)

        self.f.get_integrate_constants_of_motion(&integrate_constants_of_motion[0], self.num_odes)
        return integrate_constants_of_motion

    @property
    def integrate_phases(self):
        cdef np.ndarray[ndim=1, dtype=bool_c] integrate_phases = np.zeros(self.num_odes, dtype=bool)

        self.f.get_integrate_phases(&integrate_phases[0], self.num_odes)
        return integrate_phases

    def __reduce__(self):
        return (rebuild, (self.nparams, self.num_add_args))

    def __dealloc__(self):
        self.f.dealloc()
        if self.f:
            del self.f

def rebuild(nparams, num_add_args):
    c = pyInspiralGenerator(nparams, num_add_args)
    return c
    