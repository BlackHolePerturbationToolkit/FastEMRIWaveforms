# Copyright 2017, 2022 Michael Puerrer, Jonathan Blackman.
#
#  This file is part of TPI.
#
#  TPI is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  TPI is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with TPI.  If not, see <http://www.gnu.org/licenses/>.
#

"""
    This is the TPI package for tensor product spline interpolation.
"""

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef extern from "gsl/gsl_bspline.h":
    ctypedef struct gsl_bspline_workspace:
        pass
    ctypedef struct gsl_vector:
        pass
    ctypedef struct gsl_matrix:
        pass

    void gsl_bspline_free(gsl_bspline_workspace *w);


cdef extern from "TensorProductInterpolation.h":
    cdef int TPI_FAIL;

    ctypedef struct array:
        double *vec;
        int n;

    int Interpolation_Setup_1D(
        double *xvec,
        int nx,
        gsl_bspline_workspace **bw
    );

    int Bspline_basis_1D(
        double *B_array,
        int n,
        gsl_bspline_workspace *bw,
        double x
    );

    int Bspline_basis_3rd_derivative_1D(
        double *D3_B_array,
        int n,
        gsl_bspline_workspace *bw,
        double x
    );

    void TP_Interpolation_Setup_ND(
        array *nodes,
        int n,
        gsl_bspline_workspace ***bw_out
    );

    int TP_Interpolation_ND(
        double *v,
        int n,
        double *X,
        int m,
        gsl_bspline_workspace **bw,
        double *y
    );
    
    int AssembleSplineMatrix_C(
        gsl_vector *xi,
        gsl_matrix *phi,
        gsl_vector *knots
    );
    
    int SetupSpline1D(
        double *x,
        double *y,
        int n,
        double **c,
        gsl_bspline_workspace **bw
    );

    double EvaluateSpline1D(
        double *c,
        gsl_bspline_workspace *bw,
        double xx
    );


cdef extern from "gsl/gsl_errno.h":
    ctypedef void gsl_error_handler_t(const char *reason, const char *file, 
                                      int line, int gsl_errno);
    gsl_error_handler_t *gsl_set_error_handler(gsl_error_handler_t *new_handler) except *;

cdef void handler(const char *reason, const char *file, int line, int gsl_errno) except *:
    raise ValueError("GSL error %s in %s line %d. gsl_errno = %d\n" %(reason, file, line, gsl_errno))

cdef class TP_Interpolant_ND:

    """Tensor product spline class in N dimensions.

    The constructor sets up spline data structures for a Cartesian product grid.
    ComputeSplineCoefficientsND() computes the spline coefficients given data for 
    a scalar gridfunction.
    TPInterpolationND() carries interpolates the griddata at a desired point in 
    the parameter space spanned by the grid.
    GetSplineCoefficientsND() / SetSplineCoefficientsND allow spline coefficients 
    to be returned or set, so that the setup and solution of the interpolation 
    problem can be separated, by writing the coefficient data to disk.
    The splines are cubic and use not-a-knot boundary conditions.

    """

    cdef array* nodes_c
    cdef gsl_bspline_workspace **bw_array_ptrs
    cdef nodes, n
    cdef c, knots_list

    def __init__(self, list nodes, coeffs=None, F=None):
        """Constructor

        Arguments:
          * nodes:  list of 1D arrays defining grid points in each dimension [ x1, x2, x3, ..., xn ]
                    The 1D numpy arrays x1, ... xn can have arbitrary length and spacing. 
                    Together, the arrays in nodesND define a Cartesian product grid.
          * coeffs: (optional) tensor product spline coefficients previously obtained from
                    GetSplineCoefficientsND()
          * F:      (optional) data to be interpolated on the Cartesian product grid.
                    The shape of `F` must agree with the list of lengths of 1D arrays in `nodes`.
        
        """
        self.nodes = nodes
        self.n = len(nodes) # number of parameter space dimensions
        if not np.array(list(map(lambda x: isinstance(x, np.ndarray), nodes))).all():
            raise TypeError("Expected list of numpy.ndarrays.")
        self.nodes_c = <array*> PyMem_Malloc(self.n * sizeof(array))
        if self.nodes_c == NULL:
            raise MemoryError()
        # allocate the array of B-spline workspaces; The individual workspaces will be allocated in TP_Interpolation_Setup_ND()
        self.bw_array_ptrs = <gsl_bspline_workspace **>malloc(self.n * sizeof(gsl_bspline_workspace*));
        if self.bw_array_ptrs == NULL:
            raise MemoryError()
        self.TPInterpolationSetupND()
        if coeffs is not None:
            self.SetSplineCoefficientsND(coeffs)
        if F is not None:
            self.ComputeSplineCoefficientsND(F)

        # Should do this in module __init__.py
        cdef gsl_error_handler_t *old_handler = gsl_set_error_handler(<gsl_error_handler_t *> handler);

    def __dealloc__(self):
        """Destructor

        """
        cdef gsl_bspline_workspace *bw
        # first need to deallocate the individual workspaces
        if self.bw_array_ptrs != NULL:
            for i in range(self.n):
                bw = self.bw_array_ptrs[i]
                if bw != NULL:
                    gsl_bspline_free(bw)
            free(self.bw_array_ptrs)
        if self.nodes_c != NULL:
            PyMem_Free(self.nodes_c)

    def TPInterpolationSetupND(self):
        """Allocate data structures for tensor product spline interpolation.

        """
        cdef np.ndarray[np.double_t, ndim=1] nodes1D
        cdef unsigned int i
        # allocate C datastructure and point to numpy data
        for i in range(self.n):
            nodes1D = self.nodes[i]
            self.nodes_c[i].vec = <double*> (nodes1D.data) # just copy the pointers to the numpy arrays
            self.nodes_c[i].n = len(nodes1D)
        TP_Interpolation_Setup_ND(self.nodes_c, self.n, &(self.bw_array_ptrs))

    def TPInterpolationND(self, np.ndarray[np.double_t,ndim=1] X):
        """Carry out tensor product spline interpolation at parameter space point X.

        Arguments:
          * X: a 1D numpy array of floats.

        Returns:
          * y: the interpolant evaluated at X, a float.

        """
        cdef np.ndarray[np.double_t,ndim=1] c = self.c.flatten()
        cdef double y;
        cdef int ret = TP_Interpolation_ND(<double*> c.data, len(c),
                        <double*> X.data, len(X), self.bw_array_ptrs, &y)
        cdef double x_min, x_max;
        if ret == TPI_FAIL:
            for i in range(self.n):
                x_min = self.nodes[i][0]
                x_max = self.nodes[i][-1]
                if (X[i] < x_min or X[i] > x_max):
                    raise ValueError("TP_Interpolation_ND: X[%d] = %g "
                    "is outside of knots vector [%g, %g]!\n", i, X[i], x_min, x_max);
        return y

    def __call__(self, X):
        X_array = np.atleast_1d(np.array(X, dtype=np.double))
        if len(X_array.shape) != 1:
            raise ValueError("Evaluation point X is more than one-dimensional!")
        if X_array.shape[0] != self.n:
            raise ValueError("Expected X to be array of length %d, "
            "but got length %d"%(self.n, X_array.shape[0]))

        return self.TPInterpolationND(X_array)

    def ComputeSplineCoefficientsND(self, F):
        """Compute tensor product spline coefficients on the stored grid using data F.

        Arguments:
          * F : data on the Cartesian product grid passed to the constructor. 
                The shape of `F` must agree with the list of lengths of 1D arrays in `nodes`.

        """
        nodesND = self.nodes
        dims = list(map(len, nodesND))
        d = len(dims)

        if not np.shape(F) == tuple(dims):
            raise ValueError("Data on TP grid should have shape {}".format(dims))

        # Compute 1D spline matrices and knot vectors
        inv_1d_matrices = []
        knots_list = []
        cdef unsigned int i
        for i in range(d):
            b = BsplineBasis1D(nodesND[i])
            A, knots = b.AssembleSplineMatrix()
            Ainv = np.linalg.inv(A)
            inv_1d_matrices.append(Ainv)
            knots_list.append(knots)
        self.knots_list = knots_list

        # pad boundaries with zeroes since we have 2 more equations with the not-a-knot conditions than data
        F0 = np.lib.arraypad.pad(F, 1, 'constant')

        # Solve a sequence of linear systems to obtain coefficient tensor
        tmp_result = F0
        for minv in inv_1d_matrices[::-1]:
            tmp_result = np.tensordot(minv, tmp_result, (1, d - 1))
        self.c = tmp_result

    def GetSplineCoefficientsND(self):
        return self.c
        """Return array containing tensor product spline coefficients.

        Returns:
          * c: a len(nodes) dimensional array of shape nodes + 2 
               holding the tensor product spline coefficients

        """

    def SetSplineCoefficientsND(self, coeffs):
        """Set tensor product spline coefficients to array.

        Arguments:
          * c: a len(nodes) dimensional array of shape nodes + 2 
               holding the tensor product spline coefficients

        """
        dims = list(map(lambda x: len(x) + 2, self.nodes))

        if not np.shape(coeffs) == tuple(dims):
            raise ValueError("Spline coefficients should have shape {}".format(dims))
      
        self.c = coeffs


cdef class BsplineBasis1D:

    """ Provide B-spline basis for a knots vector. 

    Calculate B-spline basis functions or their 3rd derivatives.
    Constructs spline matrix. 
    The splines are cubic and use not-a-knot boundary conditions.

    """

    cdef gsl_bspline_workspace *bw
    cdef xi, nbasis, n

    def __init__(self, xvec_in):
        """Constructor

        Arguments:
          * xvec_in: input nodes where the spline should be constructed
        """
        self.xi = xvec_in
        if self.xi is None:
          raise ValueError("Missing input nodes.")
        self.n = len(xvec_in)
        if self.n < 2:
            raise ValueError("Require at least two input nodes.")
        if (np.diff(self.xi) < 0).any():
            raise ValueError("Input nodes must be non-decreasing.")
        if (np.isnan(self.xi).any()):
            raise ValueError("At least one of the input nodes is nan.")
        self.bw = NULL
        cdef np.ndarray[np.double_t,ndim=1] xvec = xvec_in
        self.nbasis = Interpolation_Setup_1D(<double*> xvec.data, len(xvec), &self.bw)
        if self.nbasis == TPI_FAIL:
            raise ValueError("Error: Interpolation_Setup_1D(): B-spline workspace pointers"
                             " should be NULL.\n")

    def __dealloc__(self):
        """Destructor

        """
        if self.bw != NULL:
            gsl_bspline_free(self.bw)

    def EvaluateBsplines(self, x):
        """Evaluate the B-spline basis functions at point x.

        Arguments:
          * x: evaluation point

        Returns:
          * B_array: a 1D array of evaluated B-splines

        """
        cdef np.ndarray[np.double_t, ndim=1] B_array = np.zeros(self.nbasis)
        cdef int i = Bspline_basis_1D(<double*> B_array.data, self.nbasis, self.bw, x)

        if i == TPI_FAIL:
            raise ValueError("Error: Bspline_basis_1D(): x: %g is outside of knots"
            " vector with bounds [%g, %g]!\n", x, self.xi[0], self.xi[-1])

        return B_array

    def EvaluateBsplines3rdDerivatives(self, x):
        """Evaluate the 3rd derivative of B-spline basis functions at point x.

        Arguments:
          * x: evaluation point

        Returns:
          * B_array: a 1D array of evaluated 3rd derivatives of B-splines

        """
        cdef np.ndarray[np.double_t, ndim=1] DB_array = np.zeros(self.nbasis)
        cdef int i = Bspline_basis_3rd_derivative_1D(<double*> DB_array.data,
                                            self.nbasis, self.bw, x)
        if i == TPI_FAIL:
            raise ValueError("Error: Bspline_basis_3rd_derivative_1D(): x: "
            "%g is outside of knots vector with bounds [%g, %g]!\n", 
            x, self.xi[0], self.xi[-1])

        return DB_array

    def AssembleSplineMatrix(self):
        """Assemble spline matrix for cubic spline with not-a-knot boundary conditions

        Returns:
          * phi: the matrix of spline coefficients
          * knots: the vector of knots including the endpoints with multiplicity three.

        """
        # Set up B-spline workspace and knots
        pB = self.xi[0];
        pE = self.xi[-1];
        knots = np.concatenate(((pB, pB, pB), self.xi, (pE, pE, pE)))

        # Calculate the matrix of all B-spline basis functions at all gridpoints xi
        phi_internal = np.zeros((self.n, self.nbasis))
        cdef unsigned int i
        for i in range(self.n):
            phi_internal[i] = self.EvaluateBsplines(self.xi[i])

        # Prepare not-a-knot conditions from continuity of the 3rd derivative 
        # at the 2nd and the penultimate gridpoint

        # Impose these conditions in-between the first and last two points:
        xi12mean  = (self.xi[0]  + self.xi[1]) / 2.
        xi23mean  = (self.xi[1]  + self.xi[2]) / 2.
        xim32mean = (self.xi[-3] + self.xi[-2]) / 2.
        xim21mean = (self.xi[-2] + self.xi[-1]) / 2.

        # Coefficients for rows 1 and -1:
        r1  = self.EvaluateBsplines3rdDerivatives(xi12mean) \
            - self.EvaluateBsplines3rdDerivatives(xi23mean)
        rm1 = self.EvaluateBsplines3rdDerivatives(xim32mean) \
            - self.EvaluateBsplines3rdDerivatives(xim21mean)

        phi = np.vstack((r1, phi_internal, rm1))

        return phi, knots
