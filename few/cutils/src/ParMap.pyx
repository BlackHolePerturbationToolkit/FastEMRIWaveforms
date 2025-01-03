import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/ParameterMapAAK.hh":
    void ParMapVector(double* v_map, double* M_map, double* S_map, double* OmegaPhi, double* OmegaTheta, double* OmegaR,
                  double* p, double* e, double* iota, double M, double s, int length);

def pyParMap(np.ndarray[ndim=1, dtype=np.float64_t] OmegaPhi,
             np.ndarray[ndim=1, dtype=np.float64_t] OmegaTheta,
             np.ndarray[ndim=1, dtype=np.float64_t] OmegaR,
             np.ndarray[ndim=1, dtype=np.float64_t] p,
             np.ndarray[ndim=1, dtype=np.float64_t] e,
             np.ndarray[ndim=1, dtype=np.float64_t] iota,
             M,
             s):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] v_map = np.zeros(len(OmegaPhi), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] M_map = np.zeros(len(OmegaTheta), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] S_map = np.zeros(len(OmegaR), dtype=np.float64)

    ParMapVector(&v_map[0], &M_map[0], &S_map[0], &OmegaPhi[0], &OmegaTheta[0], &OmegaR[0], &p[0], &e[0], &iota[0], M, s, len(OmegaPhi))
    return (v_map, M_map, S_map)
