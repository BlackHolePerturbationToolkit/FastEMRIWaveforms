from .base import ODEBase
from ...utils.utility import get_fundamental_frequencies, get_separatrix, check_for_file_download, ELQ_to_pex
from numba import njit

from multispline.spline import BicubicSpline, TricubicSpline
import os
from typing import Optional, Union
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
    """
    Schwarzschild eccentric flux ODE.

    Args:
        file_directory: The directory where the ODE data files are stored. Defaults to the FEW installation directory.
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """
    def __init__(self, *args, file_directory: Optional[str]=None, use_ELQ: bool=False, **kwargs):
        super().__init__(*args, file_directory=file_directory, use_ELQ=use_ELQ, **kwargs)
        # construct the BicubicSpline object from the expected file
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

    @property
    def supports_ELQ(self):
        return True

    def evaluate_rhs(self, y: Union[list[float], np.ndarray]) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x= ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]

        if e < 0 or p < 6 + 2*e:
            return [0., 0., 0., 0., 0., 0.,]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(0., p, e, x)
        yPN = Omega_phi**(2/3)

        y1 = np.log((p - 2. * e - 2.1))

        Edot_PN = _Edot_PN(e, yPN)
        Ldot_PN = _Ldot_PN(e, yPN)

        Edot = -(self.Edot_interp(y1, e)* yPN**6 + Edot_PN)
        Ldot = -(self.Ldot_interp(y1, e)* yPN**(9/2) + Ldot_PN)

        if self.use_ELQ:
            y1dot, y2dot = Edot, Ldot
        else:
            y1dot, y2dot = _schwarz_jac_kernel(p, e, Edot, Ldot)

        y3dot = 0.

        return [y1dot, y2dot, y3dot, Omega_phi, Omega_theta, Omega_r]


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
    """
    Kerr eccentric equatorial flux ODE.

    Args:
        file_directory: The directory where the ODE data files are stored. Defaults to the FEW installation directory.
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """
    def __init__(self, *args, file_directory: Optional[str]=None, use_ELQ: bool=False, **kwargs):
        super().__init__(*args,file_directory=file_directory, use_ELQ=use_ELQ, **kwargs)
        self.files = [
            "KerrEqEcc_x0.dat",
            "KerrEqEcc_x1.dat",
            "KerrEqEcc_x2.dat",
            "KerrEqEcc_pdot.dat",
            "KerrEqEcc_edot.dat"
        ]
        for fp in self.files:
            check_for_file_download(fp, self.file_dir)

        x = np.loadtxt(os.path.join(self.file_dir, self.files[0]))
        y = np.loadtxt(os.path.join(self.file_dir, self.files[1]))
        z = np.loadtxt(os.path.join(self.file_dir, self.files[2]))

        pdot = np.loadtxt(os.path.join(self.file_dir, self.files[3])).reshape(x.size, y.size, z.size)
        edot = np.loadtxt(os.path.join(self.file_dir, self.files[4])).reshape(x.size, y.size, z.size)

        self.pdot_interp = TricubicSpline(x, y, z, np.log(-pdot))
        self.edot_interp = TricubicSpline(x, y, z, edot)

    @property
    def equatorial(self):
        return True

    @property
    def separatrix_buffer_dist(self):
        return 0.05

    @property
    def supports_ELQ(self):
        return False

    def evaluate_rhs(self, y: Union[list[float], np.ndarray]) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            raise NotImplementedError
        else:
            p, e, x = y[:3]

        if e < 0:
             return [0., 0., 0., 0., 0., 0.,]

        p_sep = get_separatrix(self.a, e, x)

        if p < p_sep:
             return [0., 0., 0., 0., 0., 0.,]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)

        risco = get_separatrix(self.a, 0., x)
        u = _p_to_u(p, p_sep)
        w = e**0.5
        a_sign = self.a * x

        pdot = -np.exp(self.pdot_interp(a_sign, w, u)) * _pdot_PN(p, e, risco, p_sep)
        edot = self.edot_interp(a_sign, w, u) * _edot_PN(p, e, risco, p_sep)

        if e < 1e-6:
            edot = 0.

        return [pdot, edot, 0., Omega_phi, Omega_theta, Omega_r]
