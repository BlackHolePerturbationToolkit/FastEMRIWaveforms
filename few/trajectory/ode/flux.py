from .base import ODEBase
from ...utils.utility import _KerrGeoCoordinateFrequencies_kernel_inner, _get_separatrix_kernel_inner, check_for_file_download
from numba import njit

from multispline.spline import BicubicSpline, TricubicSpline
import os
from typing import Optional
import numpy as np
from numba import njit
from math import pow, sqrt, log

dir_path = os.path.dirname(os.path.realpath(__file__))

@njit(fastmath=True)
def _Edot_PN(e, yPN):
    return (96 + 292 * pow(e, 2) + 37 * pow(e, 4)) / (15. * pow(1 - pow(e, 2), 3.5)) * pow(yPN, 5)

@njit(fastmath=True)
def _Ldot_PN(e, yPN):
    return (4 * (8 + 7 * pow(e, 2))) / (5. * pow(-1 + pow(e, 2), 2)) * pow(yPN, 7. / 2.)

@njit(fastmath=True)
def _schwarz_jac_kernel(p, e, Edot, Ldot):
    pdot = (-2 * (Edot * sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p)) * (3 + pow(e, 2) - p) * pow(p, 1.5) + Ldot * pow(-4 + p, 2) * sqrt(-3 - pow(e, 2) + p))) / (4 * pow(e, 2) - pow(-6 + p, 2))
    if e > 0:
            edot = -((Edot * sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p)) * pow(p, 1.5) *
                        (18 + 2 * pow(e, 4) - 3 * pow(e, 2) * (-4 + p) - 9 * p + pow(p, 2)) +
                    (-1 + pow(e, 2)) * Ldot * sqrt(-3 - pow(e, 2) + p) * (12 + 4 * pow(e, 2) - 8 * p + pow(p, 2))) /
                    (e * (4 * pow(e, 2) - pow(-6 + p, 2)) * p))
    else:
            edot = 0.0
    return pdot, edot

class SchwarzEccFlux(ODEBase):
    def __init__(self, *args, file_directory: Optional[str]=None, **kwargs):
        super().__init__(*args, **kwargs)
        # construct the BicubicSpline object from the expected file
        if file_directory is None:
            self.file_dir = os.path.join(dir_path,"../../../few/files/")
        else:
            self.file_dir = file_directory

        fp = "FluxNewMinusPNScaled_fixed_y_order.dat"
        
        check_for_file_download(fp, self.file_dir)

        data = np.loadtxt(os.path.join(self.file_dir, fp))
        x = np.unique(data[:,0])
        y = np.unique(data[:,1])
        
        self.Edot_interp = BicubicSpline(x, y, data[:,2].reshape(33, 50).T)
        self.Ldot_interp = BicubicSpline(x, y, data[:,3].reshape(33, 50).T)

    @property
    def equatorial(self):
        return True
    
    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        return 0.1

    def evaluate_rhs(self, p: float, e: float, x: float, *args) -> list[float]:
        if 6 + 2*e > p:
            return np.zeros(3)


        # directly evaluate the numba kernel for speed
        Omega_phi, Omega_theta, Omega_r = _KerrGeoCoordinateFrequencies_kernel_inner(0., p, e, x)
        yPN = Omega_phi**(2/3)

        y1 = np.log((p - 2. * e - 2.1))

        Edot_PN = _Edot_PN(e, yPN)
        Ldot_PN = _Ldot_PN(e, yPN) 

        Edot = -(self.Edot_interp(y1, e)* yPN**6 + Edot_PN)
        Ldot = -(self.Ldot_interp(y1, e)* yPN**(9/2) + Ldot_PN)
        
        pdot, edot = _schwarz_jac_kernel(p, e, Edot, Ldot)
        return [pdot, edot, 0., Omega_phi, Omega_theta, Omega_r]

# TODO add ELQ


@njit(fastmath=True)
def _pdot_PN(p, e, risco, p_sep):
    return ((8. * pow(1. - (e * e), 1.5) * (8. + 7. * (e * e))) / (5. * p * (((p - risco)*(p - risco)) - ((-risco + p_sep)*(-risco + p_sep)))))

@njit(fastmath=True)
def _edot_PN(p, e, risco, p_sep):
    return ((pow(1. - (e * e), 1.5) * (304. + 121. * (e * e))) / (15. * (p*p) * (((p - risco)*(p - risco)) - ((-risco + p_sep)*(-risco + p_sep)))))

@njit(fastmath=True)
def _p_to_u(p, p_sep):
    return log((p - p_sep + 4.0 - 0.05)/4)


class KerrEccEqFlux(ODEBase):
    def __init__(self, *args, file_directory: Optional[str]=None, **kwargs):
        super().__init__(*args, **kwargs)
        # construct the BicubicSpline object from the expected file
        if file_directory is None:
            self.file_dir = os.path.join(dir_path,"../../../few/files/")
        else:
            self.file_dir = file_directory

        self.files = [
            "KerrEqEcc_x0.dat",
            "KerrEqEcc_x1.dat",
            "KerrEqEcc_x2.dat",
            "KerrEqEcc_pdot_grid.dat",
            "KerrEqEcc_edot_grid.dat"
        ]
        for fp in self.files:
            check_for_file_download(fp, self.file_dir)

        x = np.loadtxt(os.path.join(self.file_dir, self.files[0]))
        y = np.loadtxt(os.path.join(self.file_dir, self.files[1]))
        z = np.loadtxt(os.path.join(self.file_dir, self.files[2]))

        pdot = np.loadtxt(os.path.join(self.file_dir, self.files[3])).reshape(x.size, y.size, z.size)
        edot = np.loadtxt(os.path.join(self.file_dir, self.files[4])).reshape(x.size, y.size, z.size)

        self.pdot_interp = TricubicSpline(x, y, z, pdot)
        self.edot_interp = TricubicSpline(x, y, z, edot)

    @property
    def equatorial(self):
        return True

    @property
    def separatrix_buffer_dist(self):
        return 0.05


    def evaluate_rhs(self, p: float, e: float, x: float, *args) -> list[float]:

        # directly evaluate the numba kernel for speed
        Omega_phi, Omega_theta, Omega_r = _KerrGeoCoordinateFrequencies_kernel_inner(self.a, p, e, x)

        p_sep = _get_separatrix_kernel_inner(self.a, e, x)
        if e < 0 or p < p_sep:
             return [0., 0., 0., 0., 0., 0.,]

        risco = _get_separatrix_kernel_inner(self.a, 0., x)
        u = _p_to_u(p, p_sep)
        w = e**0.5
        a_sign = self.a * x

        pdot = self.pdot_interp(a_sign, w, u) * _pdot_PN(p, e, risco, p_sep)
        edot = self.edot_interp(a_sign, w, u) * _edot_PN(p, e, risco, p_sep)

        if e < 1e-6:
            edot = 0.

        return [pdot, edot, 0., Omega_phi, Omega_theta, Omega_r]
    
# TODO add ELQ
