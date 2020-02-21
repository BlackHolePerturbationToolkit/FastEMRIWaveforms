import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../inspiral/include/FluxInspiral.hh":
    void NITWrapper(np.float64_t *t, np.float64_t *p, np.float64_t *e, np.float64_t *Phi_phi, np.float64_t *Phi_r, np.float64_t p0, np.float64_t e0, int*)


def NIT(p0, e0):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] t = np.zeros(1000, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] p = np.zeros(1000, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] e = np.zeros(1000, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_phi = np.zeros(1000, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] Phi_r = np.zeros(1000, dtype=np.float64)

    cdef int length

    NITWrapper(&t[0], &p[0], &e[0], &Phi_phi[0], &Phi_r[0], p0, e0, &length)

    return (t[:length], p[:length], e[:length], Phi_phi[:length], Phi_r[:length])
