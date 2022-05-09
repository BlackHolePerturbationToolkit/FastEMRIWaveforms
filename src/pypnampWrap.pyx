import numpy as np
cimport numpy as np

from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Zlmkn8_5PNe10.h":
    ctypedef void* cmplx 'cmplx'
    void Zlmkn8_5PNe10_wrap(cmplx *Zlmkn_out, int *l_all, int *m_all, int *k_all, int *n_all, double *q_all, double *tp_all, double *te_all, double *tY_all, double *tWr_all, double *tWth_all, double *tWph_all, int num_modes, int num_points);


@pointer_adjust
def Zlmkn8_5PNe10(Zlmkn_out, l_all, m_all, k_all, n_all, q_all, tp_all, te_all, tY_all, tWr_all, tWth_all, tWph_all, num_modes, num_points):
    cdef size_t Zlmkn_out_in = Zlmkn_out
    cdef size_t l_all_in = l_all
    cdef size_t m_all_in = m_all
    cdef size_t k_all_in = k_all
    cdef size_t n_all_in = n_all

    cdef size_t q_all_in = q_all
    cdef size_t tp_all_in = tp_all
    cdef size_t te_all_in = te_all
    cdef size_t tY_all_in = tY_all
    cdef size_t tWr_all_in = tWr_all
    cdef size_t tWth_all_in = tWth_all
    cdef size_t tWph_all_in = tWph_all

    Zlmkn8_5PNe10_wrap(<cmplx *>Zlmkn_out_in, <int *>l_all_in, <int *>m_all_in, <int *>k_all_in, <int *>n_all_in, <double *>q_all_in, <double *>tp_all_in, <double *>te_all_in, <double *>tY_all_in, <double *>tWr_all_in, <double *>tWth_all_in, <double *>tWph_all_in, num_modes, num_points)