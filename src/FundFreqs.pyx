import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/FundamentalFrequencies.hh":
    void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);
    void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length)


def pyKerrGeoCoordinateFrequencies(np.ndarray[ndim=1, dtype=np.float64_t] a,
                                   np.ndarray[ndim=1, dtype=np.float64_t] p,
                                   np.ndarray[ndim=1, dtype=np.float64_t] e,
                                   np.ndarray[ndim=1, dtype=np.float64_t] x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaPhi = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaTheta = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaR = np.zeros(len(p), dtype=np.float64)

    KerrGeoCoordinateFrequenciesVectorized(&OmegaPhi[0], &OmegaTheta[0], &OmegaR[0],
                                &a[0], &p[0], &e[0], &x[0], len(p))
    return (OmegaPhi, OmegaTheta, OmegaR)


def pyGetSeparatrix(np.ndarray[ndim=1, dtype=np.float64_t] a,
                    np.ndarray[ndim=1, dtype=np.float64_t] e,
                    np.ndarray[ndim=1, dtype=np.float64_t] x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] separatrix = np.zeros_like(a)

    get_separatrix_vector(&separatrix[0], &a[0], &e[0], &x[0], len(e))

    return separatrix
