from .base import ODEBase
from ...utils.utility import get_fundamental_frequencies, get_separatrix, ELQ_to_pex
from ...utils.globals import get_file_manager
from numba import njit
from few.utils.mappings import kerrecceq_forward_map, apex_of_uwyz, apex_of_UWYZ

import h5py

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

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 3.817 - np.log((p - 2. * e - 2.1))
        dist_e = 0.75 - e

        if dist_p < 0 or dist_e < 0:
            mult = -1
        else:
            mult = 1

        dist = mult * min(abs(dist_p), abs(dist_e))
        return dist

    def interpolate_flux_grids(
              self, p: float, e: float, Omega_phi: float
              ) -> tuple[float]:

        if e > 0.755:
            raise ValueError("Interpolation: e out of bounds.")

        y1 = np.log((p - 2. * e - 2.1))
        
        if y1 < 1.3686394258811698 or y1 > 3.817712325956905:  # bounds described in 2104.04582
            raise ValueError("Interpolation: p out of bounds.")
        
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

@njit
def _EdotPN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    pdot_V = 32./5. * p**(-5) * (1-e**2)**1.5 * (1 + 73/24 * e**2 + 37/96 * e**4)
    return pdot_V

@njit
def _LdotPN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    pdot_V = 32./5. * p**(-7/2) * (1-e**2)**1.5 * (1 + 7./8. * e**2)
    return pdot_V


class KerrEccEqFlux(ODEBase):
    """
    Kerr eccentric equatorial flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
        downsample: List of two 3-tuples of integers to downsample the flux grid in u, w, z. The first list element
        refers to the inner grid, the second to the outer. Useful for testing error convergence. Defaults to None (no downsampling).
    """

    def __init__(self, *args, use_ELQ: bool = False, downsample=None, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        fp = "KerrEccEqFluxData.h5"

        if downsample is None:
            downsample = [(1,1,1),(1,1,1)]

        downsample_inner = downsample[0]
        downsample_outer = downsample[1]
        
        fm = get_file_manager()
        file_path = fm.get_file(fp)

        with h5py.File(file_path, "r") as fluxData:
            regionA = fluxData['regionA']
            u = np.linspace(0,1,regionA.attrs['NU'])[::downsample_inner[0]]
            w = np.linspace(0,1,regionA.attrs['NW'])[::downsample_inner[1]]
            z = np.linspace(0,1,regionA.attrs['NZ'])[::downsample_inner[2]]

            ugrid, wgrid, zgrid = np.asarray(np.meshgrid(u, w, z, indexing='ij')).reshape(3,-1)
            agrid, pgrid, egrid, xgrid = apex_of_uwyz(ugrid, wgrid, np.ones_like(zgrid), zgrid)
            EdotPN = _EdotPN_alt(pgrid, egrid).reshape(u.size, w.size, z.size)
            LdotPN = _LdotPN_alt(pgrid, egrid).reshape(u.size, w.size, z.size)

            # normalise by PN contribution
            Edot = regionA['Edot'][()][::downsample_inner[0],::downsample_inner[1], ::downsample_inner[2]] / EdotPN
            Ldot = regionA['Ldot'][()][::downsample_inner[0],::downsample_inner[1], ::downsample_inner[2]] / LdotPN

            self.Edot_interp_A = TricubicSpline(u, w, z, Edot)
            self.Ldot_interp_A = TricubicSpline(u, w, z, Ldot)

            regionB = fluxData['regionB']
            u = np.linspace(0,1,regionB.attrs['NU'])[::downsample_outer[0]]
            w = np.linspace(0,1,regionB.attrs['NW'])[::downsample_outer[1]]
            z = np.linspace(0,1,regionB.attrs['NZ'])[::downsample_outer[2]]

            ugrid, wgrid, zgrid = np.asarray(np.meshgrid(u, w, z, indexing='ij')).reshape(3,-1)
            agrid, pgrid, egrid, xgrid = apex_of_UWYZ(ugrid, wgrid, np.ones_like(zgrid), zgrid, True)

            EdotPN = _EdotPN_alt(pgrid, egrid).reshape(u.size, w.size, z.size)
            LdotPN = _LdotPN_alt(pgrid, egrid).reshape(u.size, w.size, z.size)

            # normalise by PN contribution
            Edot = regionB['Edot'][()][::downsample_outer[0],::downsample_outer[1], ::downsample_outer[2]] / EdotPN
            Ldot = regionB['Ldot'][()][::downsample_outer[0],::downsample_outer[1], ::downsample_outer[2]] / LdotPN

            self.Edot_interp_B = TricubicSpline(u, w, z, Edot)
            self.Ldot_interp_B = TricubicSpline(u, w, z, Ldot)

    @property
    def equatorial(self):
        return True

    @property
    def separatrix_buffer_dist(self):
        return 2e-3

    @property
    def supports_ELQ(self):
        return True

    @property
    def flux_output_convention(self):
        return "ELQ"

    def interpolate_flux_grids(
            self, p: float, e: float, x: float
        ) -> tuple[float]:
        # handle xI = -1 case
        if x == -1:
            a_in = -self.a
        else:
            a_in = self.a

        u, w, _, z, in_region_A = kerrecceq_forward_map(a_in, p, e, 1., pLSO=self.p_sep_cache, kind="flux")

        if u < 0 or u > 1 + 1e-8 or np.isnan(u):
            raise ValueError("Interpolation: p out of bounds.")
        if w < 0 or w > 1 + 1e-8:
            raise ValueError("Interpolation: e out of bounds.")

        if in_region_A:
            Edot = -self.Edot_interp_A(u, w, z) * _EdotPN_alt(p, e)
            Ldot = -self.Ldot_interp_A(u, w, z) * _LdotPN_alt(p, e)
        else:
            Edot = -self.Edot_interp_B(u, w, z) * _EdotPN_alt(p, e)
            Ldot = -self.Ldot_interp_B(u, w, z) * _LdotPN_alt(p, e)

        if a_in < 0:
            Ldot *= -1

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

        Edot, Ldot = self.interpolate_flux_grids(p, e, x)

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r]

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

class KerrEccEqFluxLegacy(ODEBase):
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
