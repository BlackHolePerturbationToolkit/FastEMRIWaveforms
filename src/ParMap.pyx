import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/ParameterMapAAK.hh":
    void ParMapVector(double* v_map, double* M_map, double* S_map, double* OmegaPhi, double* OmegaTheta, double* OmegaR,
                  double* p, double* e, double* iota, double M, double s, int length);

    void waveform(double* hI, double* hII,
                double* tvec, double* evec, double* vvec,
                double* gimvec, double* Phivec, double* alpvec, double* nuvec, double* gimdotvec, double* OmegaPhi_spin_mapped,
                double M_phys, double mu, double lam, double qS, double phiS, double qK, double phiK, double dist,
                int length, int nmodes, bool mich);


@pointer_adjust
def pyWaveform(hI, hII, tvec, evec, vvec, gimvec, Phivec, alpvec, nuvec, gimdotvec, OmegaPhi_spin_mapped,
               M_phys, mu, lam, qS, phiS, qK, phiK, dist, length, nmodes, mich):

    cdef size_t hI_in = hI
    cdef size_t hII_in = hII
    cdef size_t tvec_in = tvec
    cdef size_t evec_in = evec
    cdef size_t vvec_in = vvec
    cdef size_t gimvec_in = gimvec
    cdef size_t Phivec_in = Phivec
    cdef size_t alpvec_in = alpvec
    cdef size_t nuvec_in = nuvec
    cdef size_t gimdotvec_in = gimdotvec
    cdef size_t OmegaPhi_spin_mapped_in = OmegaPhi_spin_mapped

    waveform(<double*>hI_in, <double*>hII_in,
                <double*>tvec_in, <double*>evec_in, <double*>vvec_in,
                <double*>gimvec_in, <double*>Phivec_in, <double*>alpvec_in, <double*>nuvec_in, <double*>gimdotvec_in, <double*> OmegaPhi_spin_mapped_in,
                M_phys, mu, lam, qS, phiS, qK, phiK, dist, length, nmodes, mich)



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
