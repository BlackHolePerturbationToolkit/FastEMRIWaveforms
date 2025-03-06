from .base import ODEBase
from ...utils.utility import get_fundamental_frequencies, get_separatrix, ELQ_to_pex
from ...utils.globals import get_file_manager
from numba import njit
from few.utils.mappings.kerrecceq import (kerrecceq_flux_forward_map, 
                                apex_of_uwyz, 
                                apex_of_UWYZ, 
                                z_of_a, 
                                w_of_euz_flux, 
                                p_of_u_flux, 
                                u_where_w_is_unity, 
                                u_of_p_flux,
                                EMAX,
                                PMAX_REGIONB,
                                AMAX)
from few.utils.mappings.jacobian import ELdot_to_PEdot_Jacobian

from few.utils.utility import _brentq_jit, _get_separatrix_kernel_inner
from few.utils.exceptions import TrajectoryOffGridException

import h5py

from multispline.spline import BicubicSpline, TricubicSpline
import os
from typing import Union
import numpy as np
from math import pow, log

PMAX = PMAX_REGIONB - 1e-5
PISCO_MIN = get_separatrix(AMAX, 0, 1) + 1e-5

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

        self.flux_output_convention = "ELQ"

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

    def min_p(self, e, x = 1, a = 0):
        return 6 + 2*e + self.separatrix_buffer_dist

    def max_p(self, e, x = 1, a = 0):
        return np.exp(3.817712325956905) + 2.1 + 2.0 * e

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 3.817 - np.log((p - 2.0 * e - 2.1))
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

        y1 = np.log((p - 2.0 * e - 2.1))

        if (
            y1 < 1.3686394258811698 or y1 > 3.817712325956905
        ):  # bounds described in 2104.04582
            raise ValueError("Interpolation: p out of bounds.")

        yPN = Omega_phi ** (2 / 3)

        Edot_PN = _Edot_PN(e, yPN)
        Ldot_PN = _Ldot_PN(e, yPN)

        Edot = -(self.Edot_interp(y1, e) * yPN**6 + Edot_PN)
        Ldot = -(self.Ldot_interp(y1, e) * yPN ** (9 / 2) + Ldot_PN)

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

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r]


@njit 
def _PN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2)**1.5
    Edot = (
        32.0
        / 5.0
        * p ** (-5)
        * oneme2
        * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    )
    Ldot = 32.0 / 5.0 * p ** (-7 / 2) * (1 - e**2) ** 1.5 * (1 + 7.0 / 8.0 * e**2)
    return Edot, Ldot

@njit
def _emax_w(e, args):
    a = args[0]
    p = args[1]
    z = args[2]
    psep = _get_separatrix_kernel_inner(a, e, 1)
    u = u_of_p_flux(p, psep)
    w = w_of_euz_flux(e, u, z)
    return w-1

class KerrEccEqFlux(ODEBase):
    """
    Kerr eccentric equatorial flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
        downsample: List of two 3-tuples of integers to downsample the flux grid in u, w, z. The first list element
        refers to the inner grid, the second to the outer. Useful for testing error convergence. Defaults to None (no downsampling).
    """

    def __init__(self, *args, use_ELQ: bool = False, downsample=None, flux_output_convention="pex", **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        self.flux_output_convention = flux_output_convention

        fp = "KerrEccEqFluxData.h5"

        if downsample is None:
            downsample = [(1, 1, 1), (1, 1, 1)]

        downsample_inner = downsample[0]
        downsample_outer = downsample[1]

        fm = get_file_manager()
        file_path = fm.get_file(fp)

        # set cache of separatrix to None as placeholder
        self.p_sep_cache = None

        with h5py.File(file_path, "r") as fluxData:
            regionA = fluxData["regionA"]
            u = np.linspace(0, 1, regionA.attrs["NU"])[:: downsample_inner[0]]
            w = np.linspace(0, 1, regionA.attrs["NW"])[:: downsample_inner[1]]
            z = np.linspace(0, 1, regionA.attrs["NZ"])[:: downsample_inner[2]]

            ugrid, wgrid, zgrid = np.asarray(
                np.meshgrid(u, w, z, indexing="ij")
            ).reshape(3, -1)
            agrid, pgrid, egrid, xgrid = apex_of_uwyz(
                ugrid, wgrid, np.ones_like(zgrid), zgrid
            )

            # normalise by PN contribution
            Edot = (
                regionA["Edot"][()][
                    :: downsample_inner[0],
                    :: downsample_inner[1],
                    :: downsample_inner[2],
                ]
            )
            Ldot = (
                regionA["Ldot"][()][
                    :: downsample_inner[0],
                    :: downsample_inner[1],
                    :: downsample_inner[2],
                ]
            )

            if flux_output_convention == "pex":
                # calculate pdot and edot from Edot and Ldot
                Edothere = (Edot).flatten()
                Ldothere = (Ldot).flatten()
                xgrid = np.sign(agrid)
                xgrid[xgrid == 0] = 1
                
                Ldothere = Ldothere * xgrid
                agrid = np.abs(agrid)

                out_pdot_edot = np.asarray([ELdot_to_PEdot_Jacobian(agrid[i], pgrid[i], egrid[i], xgrid[i], Edothere[i], Ldothere[i]) for i in range(Edothere.size)])
                    
                # check whether there are no nans in the output and Edot and Ldot
                if np.isnan(out_pdot_edot).any() or np.isnan(Edot).any() or np.isnan(Ldot).any():
                    raise ValueError("Interpolation: nans in pdot, edot or Edot, Ldot.")
                
                pdot = out_pdot_edot[:, 0].reshape(u.size, w.size, z.size)
                edot = out_pdot_edot[:, 1].reshape(u.size, w.size, z.size)

                risco = get_separatrix(agrid.flatten(), np.zeros_like(agrid.flatten()), xgrid.flatten())
                psep = get_separatrix(agrid.flatten(), egrid.flatten(), xgrid.flatten())
                pdot_pn = _pdot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size)
                edot_pn = _edot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size)

                self.pdot_interp_A = TricubicSpline(u, w, z, pdot / pdot_pn)
                self.edot_interp_A = TricubicSpline(u, w, z, edot / edot_pn)
                
            else:
                EdotPN, LdotPN = _PN_alt(pgrid, egrid)
                EdotPN = EdotPN.reshape(u.size, w.size, z.size)
                LdotPN = LdotPN.reshape(u.size, w.size, z.size)

                self.Edot_interp_A = TricubicSpline(u, w, z, Edot / EdotPN)
                self.Ldot_interp_A = TricubicSpline(u, w, z, Ldot / LdotPN)

            regionB = fluxData["regionB"]
            u = np.linspace(0, 1, regionB.attrs["NU"])[:: downsample_outer[0]]
            w = np.linspace(0, 1, regionB.attrs["NW"])[:: downsample_outer[1]]
            z = np.linspace(0, 1, regionB.attrs["NZ"])[:: downsample_outer[2]]

            ugrid, wgrid, zgrid = np.asarray(
                np.meshgrid(u, w, z, indexing="ij")
            ).reshape(3, -1)
            agrid, pgrid, egrid, xgrid = apex_of_UWYZ(
                ugrid, wgrid, np.ones_like(zgrid), zgrid, True
            )

            # normalise by PN contribution
            Edot = (
                regionB["Edot"][()][
                    :: downsample_outer[0],
                    :: downsample_outer[1],
                    :: downsample_outer[2],
                ]
            )
            Ldot = (
                regionB["Ldot"][()][
                    :: downsample_outer[0],
                    :: downsample_outer[1],
                    :: downsample_outer[2],
                ]
            )

            if self.flux_output_convention == "pex":
                # calculate pdot and edot from Edot and Ldot
                Edothere = (Edot).flatten()
                Ldothere = (Ldot).flatten()
                xgrid = np.sign(agrid)
                xgrid[xgrid == 0] = 1
                
                Ldothere = Ldothere * xgrid
                agrid = np.abs(agrid)

                out_pdot_edot = np.asarray([ELdot_to_PEdot_Jacobian(agrid[i], pgrid[i], egrid[i], xgrid[i], Edothere[i], Ldothere[i]) for i in range(Edothere.size)])
                
                # check whether there are no nans in the output and Edot and Ldot
                if np.isnan(out_pdot_edot).any() or np.isnan(Edot).any() or np.isnan(Ldot).any():
                    raise ValueError("Interpolation: nans in pdot, edot or Edot, Ldot.")
                
                pdot = out_pdot_edot[:, 0].reshape(u.size, w.size, z.size)
                edot = out_pdot_edot[:, 1].reshape(u.size, w.size, z.size)

                risco = get_separatrix(agrid.flatten(), np.zeros_like(agrid.flatten()), xgrid.flatten())
                psep = get_separatrix(agrid.flatten(), egrid.flatten(), xgrid.flatten())
                pdot_pn = _pdot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size)
                edot_pn = _edot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size)

                self.pdot_interp_B = TricubicSpline(u, w, z, pdot / pdot_pn)
                self.edot_interp_B = TricubicSpline(u, w, z, edot / edot_pn)
            else:
                EdotPN, LdotPN = _PN_alt(pgrid, egrid)
                EdotPN = EdotPN.reshape(u.size, w.size, z.size)
                LdotPN = LdotPN.reshape(u.size, w.size, z.size)

                self.Edot_interp_B = TricubicSpline(u, w, z, Edot / EdotPN)
                self.Ldot_interp_B = TricubicSpline(u, w, z, Ldot / LdotPN)

    @property
    def equatorial(self):
        return True

    @property
    def separatrix_buffer_dist(self):
        return 2e-3

    @property
    def supports_ELQ(self):
        return True
    
    def isvalid_x(self, x):
        if np.any(np.abs(x) != 1):
            raise ValueError("Interpolation: x out of bounds. Must be either 1 or -1.")
        
    def isvalid_e(self, e):
        if np.any(e > EMAX) or np.any(e < 0):
            raise ValueError(f"Interpolation: e out of bounds. Must be between 0 and {EMAX}.")
    
    def isvalid_p(self, p):
        if np.any(p > PMAX) or np.any(p < PISCO_MIN + self.separatrix_buffer_dist):
            raise ValueError(f"Interpolation: p out of bounds. Must be between {PISCO_MIN + self.separatrix_buffer_dist} and {PMAX}.")
    
    def isvalid_a(self, a):
        if np.any(np.abs(a) > AMAX):
            raise ValueError(f"Interpolation: a out of bounds. Must be between {-AMAX} and {AMAX}.")

    def _min_p(self, e, x, a):
        if x == -1:
            a_in = -a
        else:
            a_in = a

        z = z_of_a(a_in)
        p_sep = _get_separatrix_kernel_inner(a, e, x)

        if w_of_euz_flux(e, 0., z) > 1:
            u_min = u_where_w_is_unity(e, z, kind="flux")
        else:
            u_min = 0.

        return max(p_of_u_flux(u_min, p_sep), p_sep + self.separatrix_buffer_dist) + 1e-5

    def _max_p(self, e, x, a):        
        return PMAX
    
    def min_p(self, e = 0, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return self._min_p(e, x, a)
    
    def max_p(self, e = 0, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return self._max_p(e, x, a)

    def _min_e(self, p, x, a):            
        return 0.0

    def _max_e(self, p, x, a):
        if x == -1:
            a_in = -a
        else:
            a_in = a

        p_sep_min_buffer = get_separatrix(a_in, 0, 1) + self.separatrix_buffer_dist
        if p < p_sep_min_buffer:
            raise ValueError(f"Interpolation: p out of bounds. Must be greater than innermost stable circular orbit + buffer = {p_sep_min_buffer}.")
        
        p_min = self._min_p(EMAX, x, a)
        if p > p_min:
            emax = EMAX
        else:
            tol = 1e-13
            z = z_of_a(a_in)
            emax = _brentq_jit(_emax_w, 0, EMAX, (a_in, p, z), tol)
        return emax

    def min_e(self, p = 20, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._min_e(p, x, a)    

    def max_e(self, p = 20, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._max_e(p, x, a)
    
    def bounds_p(self, e = 0, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return [self._min_p(e, x, a), self._max_p(e, x, a)]
    
    def bounds_e(self, p = 20, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return [self._min_e(p, x, a), self._max_e(p, x, a)]

    def interpolate_flux_grids(self, p: float, e: float, x: float = 1, a: float = 0) -> tuple[float]:
        # handle xI = -1 case
        if x == -1:
            a_in = -a
        else:
            a_in = a

        u, w, _, z, in_region_A = kerrecceq_flux_forward_map(
            a_in, p, e, 1.0, self.p_sep_cache
        )
        
        if u < 0 or u > 1 + 1e-8 or np.isnan(u):
            raise ValueError("Interpolation: p out of bounds.")
        if w < 0:
            raise TrajectoryOffGridException("Interpolation: e out of bounds.")
        if w > 1 + 1e-8:
            if self.integrate_backwards:
                raise ValueError("Interpolation: e out of bounds.")
            else:
                raise TrajectoryOffGridException("Interpolation: e out of bounds.")

        if z < 0 or z > 1 + 1e-8:
            raise TrajectoryOffGridException("Interpolation: a out of bounds.")

        if self.flux_output_convention == "ELQ":
            EdotPN, LdotPN = _PN_alt(p, e)
            if in_region_A:
                Edot = -self.Edot_interp_A(u, w, z) * EdotPN
                Ldot = -self.Ldot_interp_A(u, w, z) * LdotPN
            else:
                Edot = -self.Edot_interp_B(u, w, z) * EdotPN
                Ldot = -self.Ldot_interp_B(u, w, z) * LdotPN

            if a_in < 0:
                Ldot *= -1

            return Edot, Ldot

        else:
            risco = get_separatrix(a_in, 0.0, 1.0)
            p_sep = self.p_sep_cache
            pdotPN = _pdot_PN(p, e, risco, p_sep)
            edotPN = _edot_PN(p, e, risco, p_sep)
            if in_region_A:
                pdot = -self.pdot_interp_A(u, w, z) * pdotPN
                edot = -self.edot_interp_A(u, w, z) * edotPN
            else:
                pdot = -self.pdot_interp_B(u, w, z) * pdotPN
                edot = -self.edot_interp_B(u, w, z) * edotPN

            return pdot, edot
    
    def interpolate_ELQ_flux(self, p: float, e: float, x: float = 1, a: float = 0) -> tuple[float]:
        # handle xI = -1 case
        self.isvalid_x(x)

        if x == -1:
            a_in = -a
        else:
            a_in = a

        u, w, _, z, in_region_A = kerrecceq_flux_forward_map(
            a_in, p, e, 1.0, pLSO=self.p_sep_cache
        )

        if u < 0 or u > 1 + 1e-8 or np.isnan(u):
            raise ValueError(f"Interpolation: p={p} out of bounds.")
        if w < 0 or w > 1 + 1e-8:
            raise ValueError(f"Interpolation: e={e} out of bounds.")
        if z < 0 or z > 1 + 1e-8:
            raise ValueError(f"Interpolation: a={a} out of bounds.")

        EdotPN, LdotPN = _PN_alt(p, e)
        if in_region_A:
            Edot = self.Edot_interp_A(u, w, z) * EdotPN
            Ldot = self.Ldot_interp_A(u, w, z) * LdotPN
        else:
            Edot = self.Edot_interp_B(u, w, z) * EdotPN
            Ldot = self.Ldot_interp_B(u, w, z) * LdotPN

        if a_in < 0:
            Ldot *= -1
        
        if isinstance(Edot, float):
            Qdot = 0.0
        else:
            Qdot = np.zeros(Edot.shape)

        return Edot, Ldot, Qdot

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            a, E, L, Q = y[:4]
            p, e, x = ELQ_to_pex(a, E, L, Q)
        else:
            a, p, e, x = y[:4]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(a, p, e, x)

        Edot, Ldot = self.interpolate_flux_grids(p, e, x, a=self.a)

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r]


@njit
def _pdot_PN(p, e, risco, p_sep):
    return (8.0 * (1.0 - (e * e))** 1.5 * (8.0 + 7.0 * (e * e))) / (
        5.0 * p * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )

@njit
def _edot_PN(p, e, risco, p_sep):
    return (((1.0 - (e * e)) ** 1.5) * (304.0 + 121.0 * (e * e))) / (
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

    def interpolate_flux_grids(self, p: float, e: float, x: float) -> tuple[float]:
        risco = get_separatrix(self.a, 0.0, x)
        u = _p_to_u(p, self.p_sep_cache)
        w = e**0.5
        a_sign = self.a * x

        pdot = -np.exp(self.pdot_interp(a_sign, w, u)) * _pdot_PN(
            p, e, risco, self.p_sep_cache
        )
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
