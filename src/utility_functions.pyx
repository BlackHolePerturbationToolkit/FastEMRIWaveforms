import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/Utility.hh":
    void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);
    
    void KerrEqSpinFrequenciesCorrVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);

    void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length);

    void KerrGeoConstantsOfMotionVectorized(double* E_out, double* L_out, double* Q_out, double* a, double* p, double* e, double* x, int n);
    void ELQ_to_pexVectorised(double *p, double *e, double *x, double *a, double *E, double *Lz, double *Q, int length)
    void Y_to_xI_vector(double* x, double* a, double* p, double* e, double* Y, int length);
    void set_threads(int num_threads);
    int get_threads();

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

def pyKerrEqSpinFrequenciesCorr(np.ndarray[ndim=1, dtype=np.float64_t] a,
                                   np.ndarray[ndim=1, dtype=np.float64_t] p,
                                   np.ndarray[ndim=1, dtype=np.float64_t] e,
                                   np.ndarray[ndim=1, dtype=np.float64_t] x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaPhi = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaTheta = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaR = np.zeros(len(p), dtype=np.float64)

    KerrEqSpinFrequenciesCorrVectorized(&OmegaPhi[0], &OmegaTheta[0], &OmegaR[0],
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

def pyELQ_to_pex(np.ndarray[ndim=1, dtype=np.float64_t] a,
                                   np.ndarray[ndim=1, dtype=np.float64_t] E,
                                   np.ndarray[ndim=1, dtype=np.float64_t] Lz,
                                   np.ndarray[ndim=1, dtype=np.float64_t] Q):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] p = np.zeros(len(E), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] e = np.zeros(len(E), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] x = np.zeros(len(E), dtype=np.float64)

    ELQ_to_pexVectorised(&p[0], &e[0], &x[0], &a[0], &E[0], &Lz[0], &Q[0], len(E))
    return (p, e, x)

def pyY_to_xI_vector(np.ndarray[ndim=1, dtype=np.float64_t] a,
                     np.ndarray[ndim=1, dtype=np.float64_t] p,
                     np.ndarray[ndim=1, dtype=np.float64_t] e,
                     np.ndarray[ndim=1, dtype=np.float64_t] Y):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] x = np.zeros_like(e)

    Y_to_xI_vector(&x[0], &a[0], &p[0], &e[0], &Y[0], len(e))

    return x

def set_threads_wrap(num_threads):
    set_threads(num_threads)

def get_threads_wrap():
    return get_threads()

####################################################################################################
# spline function from Zach
from libcpp.vector cimport vector

cdef extern from "spline.hpp":
    cdef cppclass Matrix:
        Matrix()
        Matrix(int n)
        Matrix(int n, int m)
        Matrix(int n, int m, vector[double] A)
        Matrix(int n, int m, double val)

        void set_value(int i, int j, double val)
        double& operator()(int i, int j)

    cdef cppclass ThreeTensor:
        ThreeTensor()
        ThreeTensor(int nx)
        ThreeTensor(int nx, int ny, int nz)
        ThreeTensor(int nx, int ny, int nz, double *A)
        ThreeTensor(int nx, int ny, int nz, vector[double] A)
        ThreeTensor(int nx, int ny, int nz, double val)

        int rows() const
        int cols() const
        int slcs() const
        int size() const

        void row_replace(int i, Matrix row)
        void col_replace(int j, Matrix col)
        void slc_replace(int k, Matrix slc)

        Matrix row(int i)
        vector[double] rowcol(int i, int j)
        vector[double] rowslc(int i, int k)
        Matrix col(int j)
        vector[double] colslc(int j, int k)
        Matrix slc(int k)

        void reshape(int nx, int ny, int nz)
        ThreeTensor reshaped(int nx, int ny, int nz) const

        void set_value(int i, int j, int k, double val)

        double& operator()(int i, int j, int k)

    cdef cppclass CubicSpline:
        CubicSpline(double x0, double dx, const vector[double] &y, int method) except +
        CubicSpline(const vector[double] &x, const vector[double] &y, int method) except +

        double getSplineCoefficient(int i, int j)

        double evaluate(const double x)
        double derivative(const double x)
        double derivative2(const double x)

    cdef cppclass BicubicSpline:
        BicubicSpline(const vector[double] &x, const vector[double] &y, const Matrix &z, int method) except +
        BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, const Matrix &z, int method) except +
        
        double evaluate(const double x, const double y)
        double derivative_x(const double x, const double y)
        double derivative_y(const double x, const double y)
        double derivative_xy(const double x, const double y)
        double derivative_xx(const double x, const double y)
        double derivative_yy(const double x, const double y)
        CubicSpline reduce_x(const double x)
        CubicSpline reduce_y(const double y)
        double getSplineCoefficient(int i, int j, int nx, int ny)

    cdef cppclass TricubicSpline:
        # TricubicSpline(const vector[double] &x, const vector[double] &y, const vector[double] &z, ThreeTensor &f, int method) except +
        TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, ThreeTensor &f, int method) except +
        double evaluate(const double x, const double y, const double z)
        double derivative_x(const double x, const double y, const double z)
        double derivative_y(const double x, const double y, const double z)
        double derivative_z(const double x, const double y, const double z)
        double derivative_xy(const double x, const double y, const double z)
        double derivative_xz(const double x, const double y, const double z)
        double derivative_yz(const double x, const double y, const double z)
        double derivative_xx(const double x, const double y, const double z)
        double derivative_yy(const double x, const double y, const double z)
        double derivative_zz(const double x, const double y, const double z)

        double getSplineCoefficient(int i, int j, int k, int nx, int ny, int nz)

cdef class CyCubicSpline:
    cdef CubicSpline *scpp

    def __init__(self, double x0, double dx, np.ndarray[ndim=1, dtype=np.float64_t, mode='c'] f, int method):
        cdef vector[double] fvec = vector[double](len(f))
        for i in range(len(f)):
            fvec[i] = f[i]
        self.scpp = new CubicSpline(x0, dx, fvec, method)

    def coefficient(self, int i, int j):
        return self.scpp.getSplineCoefficient(i, j)

    def eval(self, double x):
        return self.scpp.evaluate(x)

    def deriv(self, double x):
        return self.scpp.derivative(x)

    def deriv2(self, double x):
        return self.scpp.derivative2(x)
    

cdef class CyBicubicSpline:
    cdef BicubicSpline *scpp

    def __init__(self, double x0, double dx, int nx, double y0, double dy, int ny, np.ndarray[ndim=2, dtype=np.float64_t, mode='c'] f, int method):
        cdef Matrix mz = Matrix(nx + 1, ny + 1)
        for i in range(nx + 1):
            for j in range(ny + 1):
                mz.set_value(i, j, f[i, j])
        self.scpp = new BicubicSpline(x0, dx, nx, y0, dy, ny, mz, method)

    def coefficient(self, int i, int j, int nx, int ny):
        return self.scpp.getSplineCoefficient(i, j, nx, ny)
    
    def eval(self, double x, double y):
        return self.scpp.evaluate(x, y)

    def deriv_x(self, double x, double y):
        return self.scpp.derivative_x(x, y)

    def deriv_y(self, double x, double y):
        return self.scpp.derivative_y(x, y)
    
    def deriv_xx(self, double x, double y):
        return self.scpp.derivative_xx(x, y)

    def deriv_yy(self, double x, double y):
        return self.scpp.derivative_yy(x, y)

    def deriv_xy(self, double x, double y):
        return self.scpp.derivative_xy(x, y)

cdef class CyTricubicSpline:
    cdef TricubicSpline *scpp

    def __init__(self, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, np.ndarray[ndim=3, dtype=np.float64_t, mode='c'] f, int method):
        cdef ThreeTensor ftens = ThreeTensor(nx + 1, ny + 1, nz + 1, &f[0,0,0])
        # cdef ThreeTensor ftens = ThreeTensor(nx + 1, ny + 1, nz + 1)
        # for i in range(nx + 1):
        #     for j in range(ny + 1):
        #         for k in range(nz + 1):
        #             ftens.set_value(i, j, k, f[i, j, k])
        self.scpp = new TricubicSpline(x0, dx, nx, y0, dy, ny, z0, dz, nz, ftens, method)

    def coefficient(self, int i, int j, int k, int nx, int ny, int nz):
        return self.scpp.getSplineCoefficient(i, j, k, nx, ny, nz)

    def eval(self, double x, double y, double z):
        return self.scpp.evaluate(x, y, z)

    def deriv_x(self, double x, double y, double z):
        return self.scpp.derivative_x(x, y, z)

    def deriv_y(self, double x, double y, double z):
        return self.scpp.derivative_y(x, y, z)

    def deriv_z(self, double x, double y, double z):
        return self.scpp.derivative_z(x, y, z)
    
    def deriv_xx(self, double x, double y, double z):
        return self.scpp.derivative_xx(x, y, z)

    def deriv_yy(self, double x, double y, double z):
        return self.scpp.derivative_yy(x, y, z)

    def deriv_zz(self, double x, double y, double z):
        return self.scpp.derivative_yy(x, y, z)

    def deriv_xy(self, double x, double y, double z):
        return self.scpp.derivative_xy(x, y, z)

    def deriv_xz(self, double x, double y, double z):
        return self.scpp.derivative_xz(x, y, z)

    def deriv_yz(self, double x, double y, double z):
        return self.scpp.derivative_yz(x, y, z)

