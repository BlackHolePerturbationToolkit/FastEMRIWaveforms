from math import log, pow
from typing import Optional, Union

import h5py
import numpy as np
from multispline.spline import BicubicSpline, TricubicSpline
from numba import njit

from ...utils.exceptions import TrajectoryOffGridException
from ...utils.geodesic import (
    ELQ_to_pex,
    _get_separatrix_kernel_inner,
    _KerrGeoCoordinateFrequencies_kernel_inner,
    get_fundamental_frequencies,
    get_separatrix,
)
from ...utils.globals import get_file_manager
from ...utils.mappings.jacobian import ELdot_to_PEdot_Jacobian
from ...utils.mappings.kerrecceq import (
    AMAX,
    DELTAPMIN,
    EMAX,
    PMAX_REGIONB,
    _kerrecceq_flux_forward_map,
    apex_of_UWYZ,
    apex_of_uwyz,
    p_of_u_flux,
    u_of_p_flux,
    u_where_w_is_unity,
    w_of_euz_flux,
    z_of_a,
)
from ...utils.utility import _brentq_jit
from .base import ODEBase

PMAX = PMAX_REGIONB
PISCO_MIN = get_separatrix(AMAX, 0, 1)

PISCO_MIN_SCHW = get_separatrix(0, 0, 1) + 1e-5
PMAX_SCHW = 47.6
EMAX_SCHW = 0.755


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

    def isvalid_x(self, x):
        if np.any(x != 1):
            raise ValueError("Interpolation: x out of bounds. Must be 1.")

    def isvalid_e(self, e, e_buffer=[0, 0]):
        emax = EMAX_SCHW - e_buffer[1]
        emin = e_buffer[0]
        if np.any(e > emax) or np.any(e < emin):
            raise ValueError(
                f"Interpolation: e out of bounds. Must be between {emin} and {emax}."
            )

    def isvalid_p(self, p, p_buffer=[0, 0]):
        pmax = PMAX_SCHW - p_buffer[1]
        pmin = PISCO_MIN_SCHW + self.separatrix_buffer_dist + p_buffer[0]
        if np.any(p > pmax) or np.any(p < pmin):
            raise ValueError(
                f"Interpolation: p out of bounds. Must be between {pmin} and {pmax}."
            )

    def isvalid_a(self, a):
        if np.any(a != 0.0):
            raise ValueError("Interpolation: a out of bounds. Must be 0.")

    def min_p(self, e, x=1, a=0):
        return 6 + 2 * e + self.separatrix_buffer_dist

    def max_p(self, e, x=1, a=0):
        return PMAX_SCHW + 2.0 * e

    def bounds_p(self, e, x=1, a=0, p_buffer=[0, 0]):
        return [self.min_p(e, x, a) + p_buffer[0], self.max_p(e, x, a) - p_buffer[1]]

    def max_e(self, p, x=1, a=0):
        return EMAX_SCHW

    def isvalid_pex(self, p=20, e=0, x=1, a=0, p_buffer=[0, 0], e_buffer=[0, 0]):
        self.isvalid_x(x)
        self.isvalid_e(e, e_buffer=e_buffer)
        self.isvalid_a(a)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer)
        assert p >= pmin and p <= pmax, (
            f"Interpolation: p out of bounds. Must be between {pmin + p_buffer[0]} and {pmax - p_buffer[1]}."
        )

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
        self, p: float, e: float, Omega_phi: float, pLSO: float = None
    ) -> tuple[float]:
        if pLSO is None:
            pLSO = 6.0 + 2.0 * e

        if e > 0.755:
            raise ValueError("Interpolation: e out of bounds.")

        y1 = np.log((p - 2.0 * e - 2.1))

        if (
            y1 < 1.3686394258811698 or y1 > 3.817712325956905
        ):  # bounds described in 2104.04582
            raise ValueError(f"Interpolation: p={p} out of bounds.")

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

        Edot, Ldot = self.interpolate_flux_grids(p, e, Omega_phi, pLSO=self.p_sep_cache)

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r]


@njit
def _PN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2) ** 1.5
    Edot = 32.0 / 5.0 * p ** (-5) * oneme2 * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    Ldot = 32.0 / 5.0 * p ** (-7 / 2) * oneme2 * (1 + 7.0 / 8.0 * e**2)
    return Edot, Ldot


@njit
def _EdotPN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2) ** 1.5
    Edot = 32.0 / 5.0 * p ** (-5) * oneme2 * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    return Edot


@njit
def _LdotPN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2) ** 1.5
    Ldot = 32.0 / 5.0 * p ** (-7 / 2) * oneme2 * (1 + 7.0 / 8.0 * e**2)
    return Ldot


@njit
def _emax_w(e, args):
    """
    Function for root-finding the maximum e-value on the domain for a given a, p, x = 1.
    """
    a = args[0]
    p = args[1]
    z = args[2]
    psep = _get_separatrix_kernel_inner(a, e, 1)
    u = u_of_p_flux(p, psep)
    w = w_of_euz_flux(e, u, z)
    return w - 1


@njit
def _emax_sep(e, args):
    """
    Function for foot-finding the e-value on the separatrix for a given a, p, x = 1.
    """
    a = args[0]
    p = args[1]
    psep = _get_separatrix_kernel_inner(a, e, 1)
    return p - psep


class KerrEccEqFlux(ODEBase):
    """
    Kerr eccentric equatorial flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
        downsample: List of two 3-tuples of integers to downsample the flux grid in u, w, z. The first list element
        refers to the inner grid, the second to the outer. Useful for testing error convergence. Defaults to None (no downsampling).
    """

    def __init__(
        self,
        *args,
        use_ELQ: bool = False,
        downsample=None,
        flux_output_convention="pex",
        **kwargs,
    ):
        super().__init__(*args, use_ELQ=use_ELQ, downsample=downsample, **kwargs)

        self.flux_output_convention = flux_output_convention

        fp = "KerrEccEqFluxData.h5"

        if downsample is None:
            downsample = [(1, 1, 1), (1, 1, 1)]

        downsample_inner = downsample[0]
        downsample_outer = downsample[1]

        fm = get_file_manager()
        file_path = fm.get_file(fp)

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
            Edot = regionA["Edot"][()][
                :: downsample_inner[0],
                :: downsample_inner[1],
                :: downsample_inner[2],
            ]
            Ldot = regionA["Ldot"][()][
                :: downsample_inner[0],
                :: downsample_inner[1],
                :: downsample_inner[2],
            ]

            if flux_output_convention == "pex":
                # calculate pdot and edot from Edot and Ldot
                Edothere = (Edot).flatten()
                Ldothere = (Ldot).flatten()
                xgrid = np.sign(agrid)
                xgrid[xgrid == 0] = 1

                Ldothere = Ldothere * xgrid
                agrid = np.abs(agrid)

                out_pdot_edot = np.asarray(
                    [
                        ELdot_to_PEdot_Jacobian(
                            agrid[i],
                            pgrid[i],
                            egrid[i],
                            xgrid[i],
                            Edothere[i],
                            Ldothere[i],
                        )
                        for i in range(Edothere.size)
                    ]
                )

                # check whether there are no nans in the output and Edot and Ldot
                if (
                    np.isnan(out_pdot_edot).any()
                    or np.isnan(Edot).any()
                    or np.isnan(Ldot).any()
                ):
                    raise ValueError("Interpolation: nans in pdot, edot or Edot, Ldot.")

                pdot = out_pdot_edot[:, 0].reshape(u.size, w.size, z.size)
                edot = out_pdot_edot[:, 1].reshape(u.size, w.size, z.size)

                risco = get_separatrix(
                    agrid.flatten(), np.zeros_like(agrid.flatten()), xgrid.flatten()
                )
                psep = get_separatrix(agrid.flatten(), egrid.flatten(), xgrid.flatten())
                pdot_pn = _pdot_PN(
                    pgrid.flatten(), egrid.flatten(), risco, psep
                ).reshape(u.size, w.size, z.size)
                edot_pn = _edot_PN(
                    pgrid.flatten(), egrid.flatten(), risco, psep
                ).reshape(u.size, w.size, z.size)

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
            Edot = regionB["Edot"][()][
                :: downsample_outer[0],
                :: downsample_outer[1],
                :: downsample_outer[2],
            ]
            Ldot = regionB["Ldot"][()][
                :: downsample_outer[0],
                :: downsample_outer[1],
                :: downsample_outer[2],
            ]

            if self.flux_output_convention == "pex":
                # calculate pdot and edot from Edot and Ldot
                Edothere = (Edot).flatten()
                Ldothere = (Ldot).flatten()
                xgrid = np.sign(agrid)
                xgrid[xgrid == 0] = 1

                Ldothere = Ldothere * xgrid
                agrid = np.abs(agrid)

                out_pdot_edot = np.asarray(
                    [
                        ELdot_to_PEdot_Jacobian(
                            agrid[i],
                            pgrid[i],
                            egrid[i],
                            xgrid[i],
                            Edothere[i],
                            Ldothere[i],
                        )
                        for i in range(Edothere.size)
                    ]
                )

                # check whether there are no nans in the output and Edot and Ldot
                if (
                    np.isnan(out_pdot_edot).any()
                    or np.isnan(Edot).any()
                    or np.isnan(Ldot).any()
                ):
                    raise ValueError("Interpolation: nans in pdot, edot or Edot, Ldot.")

                pdot = out_pdot_edot[:, 0].reshape(u.size, w.size, z.size)
                edot = out_pdot_edot[:, 1].reshape(u.size, w.size, z.size)

                risco = get_separatrix(
                    agrid.flatten(), np.zeros_like(agrid.flatten()), xgrid.flatten()
                )
                psep = get_separatrix(agrid.flatten(), egrid.flatten(), xgrid.flatten())
                pdot_pn = _pdot_PN(
                    pgrid.flatten(), egrid.flatten(), risco, psep
                ).reshape(u.size, w.size, z.size)
                edot_pn = _edot_PN(
                    pgrid.flatten(), egrid.flatten(), risco, psep
                ).reshape(u.size, w.size, z.size)

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
        return 2 * DELTAPMIN

    @property
    def separatrix_buffer_dist_grid(self):
        return DELTAPMIN

    @property
    def supports_ELQ(self):
        return True

    def isvalid_x(self, x):
        if np.any(np.abs(x) != 1):
            raise ValueError("Interpolation: x out of bounds. Must be either 1 or -1.")

    def isvalid_e(self, e, e_buffer=[0, 0]):
        emax = EMAX - e_buffer[1]
        emin = e_buffer[0]
        if np.any(e > emax) or np.any(e < emin):
            raise ValueError(
                f"Interpolation: e out of bounds. Must be between {emin} and {emax}."
            )

    def isvalid_p(self, p, p_buffer=[0, 0]):
        pmax = PMAX - p_buffer[1]
        pmin = PISCO_MIN + self.separatrix_buffer_dist + p_buffer[0]
        if np.any(p > pmax) or np.any(p < pmin):
            raise ValueError(
                f"Interpolation: p out of bounds. Must be between {pmin} and {pmax}."
            )

    def isvalid_a(self, a, a_buffer=[0, 0]):
        amax = AMAX - a_buffer[1]
        amin = -AMAX + a_buffer[0]
        if np.any(a > amax) or np.any(a < amin):
            raise ValueError(
                f"Interpolation: a out of bounds. Must be between {amin} and {amax}."
            )

    def _min_p(self, e, x, a):
        if x == -1:
            a_in = -a
        else:
            a_in = a

        z = z_of_a(a_in)
        p_sep = _get_separatrix_kernel_inner(a, e, x)

        if w_of_euz_flux(e, 0.0, z) > 1:
            u_min = u_where_w_is_unity(e, z, kind="flux")
        else:
            u_min = 0.0

        return max(p_of_u_flux(u_min, p_sep), p_sep + self.separatrix_buffer_dist)

    def _max_p(self, e, x, a):
        return PMAX

    def min_p(self, e=0, x=1, a=0):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return self._min_p(e, x, a)

    def max_p(self, e=0, x=1, a=0):
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
            raise ValueError(
                f"Interpolation: p out of bounds. Must be greater than innermost stable circular orbit + buffer = {p_sep_min_buffer}."
            )

        p_min = self._min_p(EMAX, x, a)
        if p > p_min:
            emax = EMAX
        else:
            tol = 1e-13
            z = z_of_a(a_in)
            emax = _brentq_jit(_emax_w, 0, EMAX, (a_in, p, z), tol)

            # if you lie below the separatrix, then you are limited by the max e-value on the separatrix
            if get_separatrix(a_in, emax, 1) > p:
                emax = _brentq_jit(
                    _emax_sep, 0, emax, (a_in, p - self.separatrix_buffer_dist), tol
                )
        return emax

    def min_e(self, p=20, x=1, a=0):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._min_e(p, x, a)

    def max_e(self, p=20, x=1, a=0):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._max_e(p, x, a)

    def _min_a(self, p, e, x):
        return -AMAX

    def _max_a(self, p, e, x):
        return AMAX

    def min_a(self, p=20, e=0, x=1):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return self._min_a(p, e, x)

    def max_a(self, p=20, e=0, x=1):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return self._max_a(p, e, x)

    def bounds_a(self, p=20, e=0, x=1, a_buffer=[0, 0]):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return [self._min_a(p, e, x) + a_buffer[0], self._max_a(p, e, x) - a_buffer[1]]

    def bounds_p(self, e=0, x=1, a=0, p_buffer=[0, 0]):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return [self._min_p(e, x, a) + p_buffer[0], self._max_p(e, x, a) - p_buffer[1]]

    def bounds_e(self, p=20, x=1, a=0, e_buffer=[0, 0]):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return [self._min_e(p, x, a) + e_buffer[0], self._max_e(p, x, a) - e_buffer[1]]

    def isvalid_pex(
        self, p=20, e=0, x=1, a=0, p_buffer=[0, 0], e_buffer=[0, 0], a_buffer=[0, 0]
    ):
        self.isvalid_x(x)
        self.isvalid_e(e, e_buffer=e_buffer)
        self.isvalid_a(a, a_buffer=a_buffer)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer)
        assert p >= pmin and p <= pmax, (
            f"Interpolation: p {p} out of bounds. Must be between {pmin} and {pmax}."
        )

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)

        e_max = self._max_e(p, x, self.a)

        # Subtract a small value to avoid numerical issues at the boundary
        dist_p = (PMAX - 1e-9) - p
        dist_e = (e_max - 1e-9) - e

        if dist_p < 0 or dist_e < 0:
            mult = -1
        else:
            mult = 1

        dist = mult * min(abs(dist_p), abs(dist_e))
        return dist

    def interpolate_flux_grids(
        self,
        p: float,
        e: float,
        x: float = 1,
        a: float = 0,
        pLSO: Optional[float] = None,
    ) -> tuple[float]:
        if pLSO is None:
            pLSO = get_separatrix(a, e, x)

        edge_buffer = -1e-8

        # handle xI = -1 case
        if x == -1:
            a_in = -a
        else:
            a_in = a

        u, w, _, z, in_region_A = _kerrecceq_flux_forward_map(a_in, p, e, 1.0, pLSO)

        if u < edge_buffer or u > 1 - edge_buffer or np.isnan(u):
            raise ValueError("Interpolation: p out of bounds.")
        if w < edge_buffer:
            raise TrajectoryOffGridException("Interpolation: e out of bounds.")
        if w > 1 - edge_buffer:
            if self.integrate_backwards:
                raise ValueError("Interpolation: e out of bounds.")
            else:
                raise TrajectoryOffGridException("Interpolation: e out of bounds.")

        if z < edge_buffer or z > 1 - edge_buffer:
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
            p_sep = pLSO
            pdotPN = _pdot_PN(p, e, risco, p_sep)
            edotPN = _edot_PN(p, e, risco, p_sep)
            if in_region_A:
                pdot = -self.pdot_interp_A(u, w, z) * pdotPN
                edot = -self.edot_interp_A(u, w, z) * edotPN
            else:
                pdot = -self.pdot_interp_B(u, w, z) * pdotPN
                edot = -self.edot_interp_B(u, w, z) * edotPN

            return pdot, edot

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)
        else:
            p, e, x = y[:3]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)

        Edot, Ldot = self.interpolate_flux_grids(
            p, e, x, a=self.a, pLSO=self.p_sep_cache
        )

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r]

    def evaluate_rhs_batch(self, y: np.ndarray, a) -> tuple:
        """Batched right-hand side over ``S`` systems (the CPU reference of the device RHS).

        Args:
            y: ``(6, S)`` states ``(p, e, x, Phi_phi, Phi_theta, Phi_r)`` (pex only).
            a: spin, scalar or ``(S,)`` (one per system).
        Returns:
            ``(ydot (6, S), status (S,))`` with status 0 ok, 1 ``e < 0``, 2 inside the
            separatrix, 3 off the flux grid; non-zero-status columns are NaN. Equals the
            scalar ``__call__`` column by column (no exceptions, no Python per-system loop).
        """
        if self.use_ELQ or self.flux_output_convention != "pex":
            raise NotImplementedError("evaluate_rhs_batch: pex convention only")
        y = np.ascontiguousarray(np.asarray(y, dtype=float))
        S = y.shape[1]
        a_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=float), (S,)))
        grids = (tricubic_grid(self.pdot_interp_A), tricubic_grid(self.edot_interp_A),
                 tricubic_grid(self.pdot_interp_B), tricubic_grid(self.edot_interp_B))
        ydot = np.empty((6, S))
        status = np.zeros(S, dtype=np.int64)
        _kerr_ecc_eq_rhs_batch(y, a_arr, ydot, status, *grids)
        if self.integrate_backwards:
            ydot *= -1.0
        return ydot, status


@njit
def _pdot_PN(p, e, risco, p_sep):
    return (8.0 * (1.0 - (e * e)) ** 1.5 * (8.0 + 7.0 * (e * e))) / (
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



# ----------------------------------------------------------------------
# Batched (numSys) pieces: numba CPU reference of the device kernels
# ----------------------------------------------------------------------

def tricubic_grid(spline):
    """``(coeffs, x0, dx, nx, y0, dy, ny, z0, dz, nz)`` of a multispline ``TricubicSpline``.

    ``coeffs`` has shape ``(nx, ny, 64 * nz)`` (cells); cell ``(i, j, k)`` holds the 64
    monomial coefficients of the local normalised coordinates at ``64 k + 16 mx + 4 my + mz``
    (verified against ``TricubicSpline.__call__`` to 1.5e-15).
    """
    C = np.asarray(spline.coefficients)
    return (C, float(spline.x0), float(spline.dx), int(C.shape[0]),
            float(spline.y0), float(spline.dy), int(C.shape[1]),
            float(spline.z0), float(spline.dz), int(C.shape[2] // 64))


@njit(fastmath=False)
def _tricubic_one(C, x0, dx, nx, y0, dy, ny, z0, dz, nz, x, y, z):
    fx = (x - x0) / dx
    fy = (y - y0) / dy
    fz = (z - z0) / dz
    i = min(max(int(np.floor(fx)), 0), nx - 1)
    j = min(max(int(np.floor(fy)), 0), ny - 1)
    k = min(max(int(np.floor(fz)), 0), nz - 1)
    tx, ty, tz = fx - i, fy - j, fz - k
    base = 64 * k
    out = 0.0
    txp = 1.0
    for mx in range(4):
        typ = 1.0
        for my in range(4):
            tzp = 1.0
            acc = 0.0
            for mz in range(4):
                acc += C[i, j, base + 16 * mx + 4 * my + mz] * tzp
                tzp *= tz
            out += acc * txp * typ
            typ *= ty
        txp *= tx
    return out


@njit(fastmath=False)
def _tricubic_batch(C, x0, dx, nx, y0, dy, ny, z0, dz, nz, xs, ys, zs, out):
    for q in range(xs.shape[0]):
        out[q] = _tricubic_one(C, x0, dx, nx, y0, dy, ny, z0, dz, nz, xs[q], ys[q], zs[q])


def tricubic_eval_batch(grid, x, y, z):
    """Evaluate a :func:`tricubic_grid` at arrays of points (numba, no Python loop)."""
    x = np.ascontiguousarray(np.asarray(x, dtype=float))
    out = np.empty(x.shape[0])
    _tricubic_batch(*grid, x, np.ascontiguousarray(np.asarray(y, dtype=float)),
                    np.ascontiguousarray(np.asarray(z, dtype=float)), out)
    return out


@njit(fastmath=False)
def _kerr_ecc_eq_rhs_batch(y, a_arr, ydot, status,
                           gpa, gea, gpb, geb):
    edge_buffer = -1e-8
    for s in range(y.shape[1]):
        p, e, x = y[0, s], y[1, s], y[2, s]
        a = a_arr[s]
        if not (e >= 0.0):
            status[s] = 1
            ydot[:, s] = np.nan
            continue
        p_sep = _get_separatrix_kernel_inner(a, e, x, 1e-13)
        if not (p > p_sep):
            status[s] = 2
            ydot[:, s] = np.nan
            continue
        Om_phi, Om_theta, Om_r = _KerrGeoCoordinateFrequencies_kernel_inner(a, p, e, x)
        a_in = -a if x == -1 else a
        u, w, _yy, z, in_region_A = _kerrecceq_flux_forward_map(a_in, p, e, 1.0, p_sep)
        if (u < edge_buffer or u > 1 - edge_buffer or np.isnan(u) or w < edge_buffer
                or w > 1 - edge_buffer or z < edge_buffer or z > 1 - edge_buffer):
            status[s] = 3
            ydot[:, s] = np.nan
            continue
        risco = _get_separatrix_kernel_inner(a_in, 0.0, 1.0, 1e-13)
        pdotPN = _pdot_PN(p, e, risco, p_sep)
        edotPN = _edot_PN(p, e, risco, p_sep)
        if in_region_A:
            pdot = -_tricubic_one(gpa[0], gpa[1], gpa[2], gpa[3], gpa[4], gpa[5], gpa[6], gpa[7], gpa[8], gpa[9], u, w, z) * pdotPN
            edot = -_tricubic_one(gea[0], gea[1], gea[2], gea[3], gea[4], gea[5], gea[6], gea[7], gea[8], gea[9], u, w, z) * edotPN
        else:
            pdot = -_tricubic_one(gpb[0], gpb[1], gpb[2], gpb[3], gpb[4], gpb[5], gpb[6], gpb[7], gpb[8], gpb[9], u, w, z) * pdotPN
            edot = -_tricubic_one(geb[0], geb[1], geb[2], geb[3], geb[4], geb[5], geb[6], geb[7], geb[8], geb[9], u, w, z) * edotPN
        ydot[0, s] = pdot
        ydot[1, s] = edot
        ydot[2, s] = 0.0
        ydot[3, s] = Om_phi
        ydot[4, s] = Om_theta
        ydot[5, s] = Om_r
