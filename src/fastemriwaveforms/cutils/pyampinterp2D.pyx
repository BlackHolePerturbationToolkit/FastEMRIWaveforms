import numpy as np
cimport numpy as np
from libcpp.string cimport string

from few.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "AmpInterp2D.hh":
    void interp2D_wrap(double *z, const double* tx, int nx, const double* ty, int ny, double* c,
             int kx, int ky, const double* x, int mx,
             const double* y, int my, int num_indiv_c, int len_indiv_c) except+;


@pointer_adjust
def interp2D(z, tx, nx, ty, ny, c, kx, ky, x, mx, y, my, num_indiv_c, len_indiv_c):

    cdef size_t z_in = z
    cdef size_t c_in = c
    cdef size_t tx_in = tx
    cdef size_t ty_in = ty
    cdef size_t x_in = x
    cdef size_t y_in = y

    interp2D_wrap(<double*> z_in, <double*> tx_in, nx, <double*> ty_in, ny, <double*> c_in,
            kx, ky, <double*> x_in, mx,
            <double*> y_in, my, num_indiv_c, len_indiv_c);
