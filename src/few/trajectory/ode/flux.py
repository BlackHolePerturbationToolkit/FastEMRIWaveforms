from .base import ODEBase
from ...utils.utility import get_fundamental_frequencies, get_separatrix, ELQ_to_pex
from ...utils.globals import get_file_manager
from numba import njit

from multispline.spline import BicubicSpline, TricubicSpline
import os
from typing import Union
import numpy as np
from math import pow, sqrt, log

dir_path = os.path.dirname(os.path.realpath(__file__))

@njit
def _Edot_PN(e, yPN):
    return (
        (96 + 292 * pow(e, 2) + 37 * pow(e, 4))
        / (15.0 * pow(1 - pow(e, 2), 3.5))
        * pow(yPN, 5)
    )


@njit
def _Ldot_PN(e, yPN):
    return (
        (4 * (8 + 7 * pow(e, 2))) / (5.0 * pow(-1 + pow(e, 2), 2)) * pow(yPN, 7.0 / 2.0)
    )


@njit
def _schwarz_jac_kernel(p, e, Edot, Ldot):
    pdot = (
        -2
        * (
            Edot
            * sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p))
            * (3 + pow(e, 2) - p)
            * pow(p, 1.5)
            + Ldot * pow(-4 + p, 2) * sqrt(-3 - pow(e, 2) + p)
        )
    ) / (4 * pow(e, 2) - pow(-6 + p, 2))
    if e > 0:
        edot = -(
            (
                Edot
                * sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p))
                * pow(p, 1.5)
                * (18 + 2 * pow(e, 4) - 3 * pow(e, 2) * (-4 + p) - 9 * p + pow(p, 2))
                + (-1 + pow(e, 2))
                * Ldot
                * sqrt(-3 - pow(e, 2) + p)
                * (12 + 4 * pow(e, 2) - 8 * p + pow(p, 2))
            )
            / (e * (4 * pow(e, 2) - pow(-6 + p, 2)) * p)
        )
    else:
        edot = 0.0
    return pdot, edot


class SchwarzEccFlux(ODEBase):
    """
    Schwarzschild eccentric flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)
        # construct the BicubicSpline object from the expected file
        fp = "FluxNewMinusPNScaled_fixed_y_order.dat"

        data = np.loadtxt(get_file_manager().get_file(fp))
        x = np.unique(data[:, 0])
        y = np.unique(data[:, 1])

        self.Edot_interp = BicubicSpline(x, y, data[:, 2].reshape(33, 50).T)
        self.Ldot_interp = BicubicSpline(x, y, data[:, 3].reshape(33, 50).T)

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

    def interpolate_flux_grids(
              self, p: float, e: float, Omega_phi: float
              ) -> tuple[float]:

        y1 = np.log((p - 2. * e - 2.1))
        yPN = Omega_phi**(2/3)

        Edot_PN = _Edot_PN(e, yPN)
        Ldot_PN = _Ldot_PN(e, yPN)

        Edot = -(self.Edot_interp(y1, e)* yPN**6 + Edot_PN)
        Ldot = -(self.Ldot_interp(y1, e)* yPN**(9/2) + Ldot_PN)

        return Edot, Ldot

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)

        Edot, Ldot = self.interpolate_flux_grids(p, e, Omega_phi)

        return [Edot, Ldot, 0., Omega_phi, Omega_theta, Omega_r]


@njit(fastmath=True)
def _pdot_PN(p, e, risco, p_sep):
    return (8.0 * pow(1.0 - (e * e), 1.5) * (8.0 + 7.0 * (e * e))) / (
        5.0 * p * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )


@njit(fastmath=True)
def _edot_PN(p, e, risco, p_sep):
    return (pow(1.0 - (e * e), 1.5) * (304.0 + 121.0 * (e * e))) / (
        15.0
        * (p * p)
        * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )


@njit(fastmath=True)
def _p_to_u(p, p_sep):
    return log((p - p_sep + 4.0 - 0.05) / 4)


class KerrEccEqFlux(ODEBase):
    """
    Kerr eccentric equatorial flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)
        self.files = [
            "KerrEqEcc_x0.dat",
            "KerrEqEcc_x1.dat",
            "KerrEqEcc_x2.dat",
            "KerrEqEcc_pdot.dat",
            "KerrEqEcc_edot.dat",
        ]
        fm = get_file_manager()
        fm.prefetch_files_by_list(self.files)

        x = np.loadtxt(fm.get_file(self.files[0]))
        y = np.loadtxt(fm.get_file(self.files[1]))
        z = np.loadtxt(fm.get_file(self.files[2]))

        pdot = np.loadtxt(fm.get_file(self.files[3])).reshape(x.size, y.size, z.size)
        edot = np.loadtxt(fm.get_file(self.files[4])).reshape(x.size, y.size, z.size)

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

    @property
    def flux_output_convention(self):
        return "pex"

    def interpolate_flux_grids(
            self, p: float, e: float, x: float
        ) -> tuple[float]:

        risco = get_separatrix(self.a, 0., x)
        u = _p_to_u(p, self.p_sep_cache)
        w = e**0.5
        a_sign = self.a * x

        pdot = -np.exp(self.pdot_interp(a_sign, w, u)) * _pdot_PN(p, e, risco, self.p_sep_cache)
        edot = self.edot_interp(a_sign, w, u) * _edot_PN(p, e, risco, self.p_sep_cache)

        return pdot, edot

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            raise NotImplementedError
        else:
            p, e, x = y[:3]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)

        pdot, edot = self.interpolate_flux_grids(p, e, x)

        return [pdot, edot, 0.0, Omega_phi, Omega_theta, Omega_r]
