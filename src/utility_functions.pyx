import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/Utility.hh":
    void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);

    void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length);

    void KerrGeoConstantsOfMotionVectorized(double* E_out, double* L_out, double* Q_out, double* a, double* p, double* e, double* x, int n);
    void Y_to_xI_vector(double* x, double* a, double* p, double* e, double* Y, int length);

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

    cdef np.ndarray[ndim=1, dtype=np.float64_t] separatrix = np.zeros_like(e)

    get_separatrix_vector(&separatrix[0], &a[0], &e[0], &x[0], len(e))

    return separatrix

def pyKerrGeoConstantsOfMotionVectorized(np.ndarray[ndim=1, dtype=np.float64_t]  a,
                                         np.ndarray[ndim=1, dtype=np.float64_t]  p,
                                         np.ndarray[ndim=1, dtype=np.float64_t]  e,
                                         np.ndarray[ndim=1, dtype=np.float64_t]  x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] E_out = np.zeros_like(e)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] L_out = np.zeros_like(e)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] Q_out = np.zeros_like(e)

    KerrGeoConstantsOfMotionVectorized(&E_out[0], &L_out[0], &Q_out[0], &a[0], &p[0], &e[0], &x[0], len(e))

    return (E_out, L_out, Q_out)

def pyY_to_xI_vector(np.ndarray[ndim=1, dtype=np.float64_t] a,
                     np.ndarray[ndim=1, dtype=np.float64_t] p,
                     np.ndarray[ndim=1, dtype=np.float64_t] e,
                     np.ndarray[ndim=1, dtype=np.float64_t] Y):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] x = np.zeros_like(e)

    Y_to_xI_vector(&x[0], &a[0], &p[0], &e[0], &Y[0], len(e))

    return x
