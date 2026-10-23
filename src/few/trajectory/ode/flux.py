r"""Flux-interpolated trajectory models: Kerr eccentric equatorial and Schwarzschild eccentric.

- :class:`KerrEccEqFlux`: the trajectory model of
  :class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux` (and of the LISA
  global-fit EMRI path). Tricubic splines (``multispline``) of the adiabatic fluxes of
  ``KerrEccEqFluxData.h5`` on two grids in the mapped coordinates ``(u, w, z)`` of
  :mod:`few.utils.mappings.kerrecceq` (region A near the separatrix, region B out to
  ``p = 200``), normalised by PN-like factors (:func:`_pdot_PN`, :func:`_edot_PN`).
- :class:`SchwarzEccFlux`: bicubic splines of the Schwarzschild eccentric fluxes of
  ``FluxNewMinusPNScaled_fixed_y_order.dat`` (arXiv:2104.04582).
- :class:`KerrEccEqFluxLegacy`: the superseded Kerr eccentric equatorial model (not a stock
  option; its constructor currently raises ``AttributeError``).

The module also holds the numba pieces of the experimental batched right-hand side
(:meth:`KerrEccEqFlux.evaluate_rhs_batch`, :func:`tricubic_grid`,
:func:`tricubic_eval_batch`), the CPU reference used by
:class:`~few.trajectory.integrate.BatchedIntegrate`.

Module constants: ``PMAX`` (200, the outer ``p`` edge of the Kerr grid), ``PISCO_MIN``
(the prograde ISCO at the maximum spin ``AMAX = 0.999``, the smallest ``p`` anywhere on the
Kerr grid), and the Schwarzschild limits ``PISCO_MIN_SCHW``, ``PMAX_SCHW`` (47.6) and
``EMAX_SCHW`` (0.755).
"""

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
    """Peters-Mathews energy flux used by :class:`SchwarzEccFlux` (``yPN = Omega_phi**(2/3)``)."""
    return (
        (96 + 292 * pow(e, 2) + 37 * pow(e, 4))
        / (15.0 * pow(1 - pow(e, 2), 3.5))
        * pow(yPN, 5)
    )


@njit
def _Ldot_PN(e, yPN):
    """Peters-Mathews angular-momentum flux used by :class:`SchwarzEccFlux`."""
    return (
        (4 * (8 + 7 * pow(e, 2))) / (5.0 * pow(-1 + pow(e, 2), 2)) * pow(yPN, 7.0 / 2.0)
    )


class SchwarzEccFlux(ODEBase):
    """Schwarzschild eccentric flux ODE (the trajectory of the Schwarzschild waveform models).

    Bicubic splines in ``(y1, e)``, ``y1 = log(p - 2 e - 2.1)``, of the energy and
    angular-momentum fluxes minus their Peters-Mathews parts, read from
    ``FluxNewMinusPNScaled_fixed_y_order.dat`` (33 x 50 grid, arXiv:2104.04582). The
    fluxes are ``(Edot, Ldot)`` (``flux_output_convention = "ELQ"``), so a pex
    integration applies the Jacobian to ``(pdot, edot)`` in
    :meth:`~few.trajectory.ode.base.ODEBase.__call__`. Domain: ``a = 0``, ``x = 1``,
    ``0 <= e <= 0.755``, ``p`` between ``6 + 2e + 0.1`` (the separatrix buffer is 0.1) and
    the grid edge ``y1 <= 3.8177``.

    Args:
        *args: Forwarded to :class:`~few.trajectory.ode.base.ODEBase`.
        use_ELQ: If ``True``, the ODE takes and returns ``(E, L, Q)``. Default ``False``.
        **kwargs: Forwarded to :class:`~few.trajectory.ode.base.ODEBase`.

    Raises:
        few.utils.exceptions.FileManagerException: If the flux file cannot be found or
            fetched.

    Example:
        .. code-block:: python

            from few.trajectory.ode import SchwarzEccFlux

            rhs = SchwarzEccFlux()
            rhs.add_fixed_parameters(1e6, 1e1, 0.0)
            pdot, edot, xdot, Om_phi, Om_theta, Om_r = rhs([10.0, 0.3, 1.0])
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
        """bool: ``True`` (equatorial orbits only)."""
        return True

    @property
    def background(self):
        """str: ``"Schwarzschild"``."""
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        """float: ``0.1``, the stop distance in ``p`` above the separatrix ``6 + 2e``."""
        return 0.1

    @property
    def supports_ELQ(self):
        """bool: ``True``."""
        return True

    def isvalid_x(self, x):
        """Raise ``ValueError`` unless ``x == 1``."""
        if np.any(x != 1):
            raise ValueError("Interpolation: x out of bounds. Must be 1.")

    def isvalid_e(self, e, e_buffer=[0, 0]):
        """Raise ``ValueError`` unless ``e_buffer[0] <= e <= 0.755 - e_buffer[1]``."""
        emax = EMAX_SCHW - e_buffer[1]
        emin = e_buffer[0]
        if np.any(e > emax) or np.any(e < emin):
            raise ValueError(
                f"Interpolation: e out of bounds. Must be between {emin} and {emax}."
            )

    def isvalid_p(self, p, p_buffer=[0, 0]):
        """Raise ``ValueError`` unless ``p`` is in the grid range, shrunk by ``p_buffer``.

        The range is ``[PISCO_MIN_SCHW + 0.1, 47.6]``.
        """
        pmax = PMAX_SCHW - p_buffer[1]
        pmin = PISCO_MIN_SCHW + self.separatrix_buffer_dist + p_buffer[0]
        if np.any(p > pmax) or np.any(p < pmin):
            raise ValueError(
                f"Interpolation: p out of bounds. Must be between {pmin} and {pmax}."
            )

    def isvalid_a(self, a):
        """Raise ``ValueError`` unless ``a == 0``."""
        if np.any(a != 0.0):
            raise ValueError("Interpolation: a out of bounds. Must be 0.")

    def min_p(self, e, x=1, a=0):
        """Return the smallest ``p``: the separatrix plus buffer, ``6 + 2e + 0.1``."""
        return 6 + 2 * e + self.separatrix_buffer_dist

    def max_p(self, e, x=1, a=0):
        """Return the largest ``p`` used by the start check, ``47.6 + 2e``."""
        return PMAX_SCHW + 2.0 * e

    def bounds_p(self, e, x=1, a=0, p_buffer=[0, 0]):
        """Return ``[min_p + p_buffer[0], max_p - p_buffer[1]]`` (no ``x``, ``e``, ``a`` checks)."""
        return [self.min_p(e, x, a) + p_buffer[0], self.max_p(e, x, a) - p_buffer[1]]

    def max_e(self, p, x=1, a=0):
        """Return the largest eccentricity, ``0.755``."""
        return EMAX_SCHW

    def isvalid_pex(self, p=20, e=0, x=1, a=0, p_buffer=[0, 0], e_buffer=[0, 0]):
        """Check a start point: ``x``, ``e``, ``a`` (``ValueError``), ``p`` (``AssertionError``)."""
        self.isvalid_x(x)
        self.isvalid_e(e, e_buffer=e_buffer)
        self.isvalid_a(a)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer)
        assert p >= pmin and p <= pmax, (
            f"Interpolation: p out of bounds. Must be between {pmin + p_buffer[0]} and {pmax - p_buffer[1]}."
        )

    def distance_to_outer_boundary(self, y):
        """Return the signed distance to the grid's outer edge (``y1 = 3.817`` or ``e = 0.75``).

        Args:
            y (np.ndarray): State.

        Returns:
            float: ``min(|3.817 - y1|, |0.75 - e|)``, negative if either edge is passed.
        """
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
        """Interpolate the energy and angular-momentum fluxes at ``(p, e)``.

        Args:
            p: Semi-latus rectum in units of :math:`M`.
            e: Eccentricity.
            Omega_phi: Dimensionless azimuthal frequency (sets ``yPN = Omega_phi**(2/3)``).
            pLSO: Separatrix; unused apart from its default ``6 + 2e``.

        Returns:
            tuple[float]: ``(Edot, Ldot)``, negative (energy and angular momentum are lost),
            per unit of the radiation-reaction time.

        Raises:
            ValueError: If ``e > 0.755`` or ``y1 = log(p - 2e - 2.1)`` is outside
                ``[1.3686, 3.8177]``.
        """
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
        """Return ``[Edot, Ldot, 0, Omega_phi, Omega_theta, Omega_r]`` at the state ``y``.

        The fluxes are in the ELQ convention; :meth:`~few.trajectory.ode.base.ODEBase.__call__`
        converts them to ``(pdot, edot)`` for a pex integration.

        Args:
            y: State, ``(p, e, x)`` or ``(E, L, Q)`` first.

        Returns:
            list: Six values (fluxes, then the dimensionless frequencies).
        """
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
    """Leading-order ``(Edot, Ldot)`` magnitudes, the normalisation of the ELQ flux grids.

    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2) ** 1.5
    Edot = 32.0 / 5.0 * p ** (-5) * oneme2 * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    Ldot = 32.0 / 5.0 * p ** (-7 / 2) * oneme2 * (1 + 7.0 / 8.0 * e**2)
    return Edot, Ldot


# TODO check: keep/remove/document -- no callers (duplicates the Edot half of _PN_alt)
@njit
def _EdotPN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2) ** 1.5
    Edot = 32.0 / 5.0 * p ** (-5) * oneme2 * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    return Edot


# TODO check: keep/remove/document -- no callers (duplicates the Ldot half of _PN_alt)
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
    """Root-finding target of :meth:`KerrEccEqFlux._max_e`: ``w(e) - 1`` at fixed ``(a, p, z)``.

    ``args = (a, p, z)``; the root is the largest grid eccentricity at ``p`` (``x = 1``).
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
    """Root-finding target of :meth:`KerrEccEqFlux._max_e`: ``p - p_sep(a, e, 1)``.

    ``args = (a, p)``; the root is the eccentricity whose separatrix is ``p`` (``x = 1``).
    """
    a = args[0]
    p = args[1]
    psep = _get_separatrix_kernel_inner(a, e, 1)
    return p - psep


class KerrEccEqFlux(ODEBase):
    r"""Kerr eccentric equatorial flux ODE (the trajectory of ``FastKerrEccentricEquatorialFlux``).

    Loads ``KerrEccEqFluxData.h5`` once (through the FEW file manager) and builds four
    ``multispline`` tricubic splines on the mapped coordinates ``(u, w, z)`` of
    :mod:`few.utils.mappings.kerrecceq` (``u`` from ``p``, ``w`` from ``e``, ``z`` from the
    signed spin): region A, ``p <= p_sep + 9.001`` near the separatrix, and region B out to
    ``p = 200``. With the default ``flux_output_convention="pex"`` the grid fluxes
    ``(Edot, Ldot)`` are converted to ``(pdot, edot)`` at the nodes and stored divided by the
    PN-like factors :func:`_pdot_PN` / :func:`_edot_PN`; with ``"ELQ"`` the splines hold
    ``(Edot, Ldot)`` over :func:`_PN_alt`.

    Domain: :math:`|a| \le 0.999`, ``x = +1`` (prograde) or ``-1`` (retrograde; evaluated
    as prograde with spin ``-a``), :math:`0 \le e \le 0.9` within the grid's ``w <= 1``
    edge, ``p`` from :meth:`_min_p` (at least ``p_sep + 0.002``, the separatrix buffer of
    ``2 * DELTAPMIN``) up to 200. :meth:`interpolate_flux_grids` raises
    :class:`~few.utils.exceptions.TrajectoryOffGridException` when a trajectory leaves the
    grid in ``e`` or ``a``, ``ValueError`` in ``p``.

    Args:
        *args: Forwarded to :class:`~few.trajectory.ode.base.ODEBase`.
        use_ELQ: If ``True``, the ODE takes and returns ``(E, L, Q)``. Default ``False``.
        downsample (list, optional): Two 3-tuples of integer strides ``(u, w, z)`` that
            downsample the flux grids, region A first, region B second; useful for testing
            interpolation-error convergence. Default ``None`` (no downsampling).
        flux_output_convention (str, optional): ``"pex"`` (default; splines of ``pdot``,
            ``edot``) or ``"ELQ"`` (splines of ``Edot``, ``Ldot``, converted by the Jacobian
            in :meth:`~few.trajectory.ode.base.ODEBase.__call__`).
        **kwargs: Forwarded to :class:`~few.trajectory.ode.base.ODEBase` (ignored there).

    Raises:
        ValueError: If the flux data produce NaNs in the pex conversion, or for an invalid
            ``flux_output_convention``.
        few.utils.exceptions.FileManagerException: If the flux file cannot be found or
            fetched.

    Example:
        .. code-block:: python

            from few.trajectory.ode import KerrEccEqFlux

            rhs = KerrEccEqFlux()
            rhs.add_fixed_parameters(1e6, 1e1, 0.9)       # m1, m2 [Msun], a
            pdot, edot, xdot, Om_phi, Om_theta, Om_r = rhs([10.0, 0.3, 1.0])
            emax = rhs.max_e(10.0, a=0.9)                  # largest e on the grid at p = 10
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
        """bool: ``True`` (``x = +1`` or ``-1`` only)."""
        return True

    @property
    def separatrix_buffer_dist(self):
        """float: ``2 * DELTAPMIN = 0.002``, the stop distance in ``p`` above the separatrix."""
        return 2 * DELTAPMIN

    @property
    def separatrix_buffer_dist_grid(self):
        """float: ``DELTAPMIN = 0.001``, the inner edge of the flux grid above the separatrix."""
        return DELTAPMIN

    @property
    def supports_ELQ(self):
        """bool: ``True``."""
        return True

    def isvalid_x(self, x):
        """Raise ``ValueError`` unless ``|x| == 1``."""
        if np.any(np.abs(x) != 1):
            raise ValueError("Interpolation: x out of bounds. Must be either 1 or -1.")

    def isvalid_e(self, e, e_buffer=[0, 0]):
        """Raise ``ValueError`` unless ``e_buffer[0] <= e <= 0.9 - e_buffer[1]`` (``EMAX``)."""
        emax = EMAX - e_buffer[1]
        emin = e_buffer[0]
        if np.any(e > emax) or np.any(e < emin):
            raise ValueError(
                f"Interpolation: e out of bounds. Must be between {emin} and {emax}."
            )

    def isvalid_p(self, p, p_buffer=[0, 0]):
        """Raise ``ValueError`` unless ``p`` is within the global grid range.

        The range is ``[PISCO_MIN + 0.002 + p_buffer[0], 200 - p_buffer[1]]``, independent
        of ``(e, x, a)``; :meth:`bounds_p` gives the ``(e, x, a)``-dependent range.
        """
        pmax = PMAX - p_buffer[1]
        pmin = PISCO_MIN + self.separatrix_buffer_dist + p_buffer[0]
        if np.any(p > pmax) or np.any(p < pmin):
            raise ValueError(
                f"Interpolation: p out of bounds. Must be between {pmin} and {pmax}."
            )

    def isvalid_a(self, a, a_buffer=[0, 0]):
        """Raise ``ValueError`` unless ``-0.999 + a_buffer[0] <= a <= 0.999 - a_buffer[1]``."""
        amax = AMAX - a_buffer[1]
        amin = -AMAX + a_buffer[0]
        if np.any(a > amax) or np.any(a < amin):
            raise ValueError(
                f"Interpolation: a out of bounds. Must be between {amin} and {amax}."
            )

    def _min_p(self, e, x, a):
        """Smallest ``p`` on the flux grid at ``(e, x, a)`` (no input checks).

        The larger of the grid's inner edge (where ``w = 1`` for large ``e``, else ``u = 0``)
        and ``p_sep + separatrix_buffer_dist``.

        Args:
            e (float): Eccentricity.
            x (float): ``+1`` or ``-1``.
            a (float): Dimensionless spin.

        Returns:
            float: The minimum ``p`` in units of :math:`M`.
        """
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
        """Largest ``p`` on the flux grid: ``PMAX = 200`` for every ``(e, x, a)``."""
        return PMAX

    def min_p(self, e=0, x=1, a=0):
        """Smallest ``p`` on the grid at ``(e, x, a)``, after checking them (``ValueError``)."""
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return self._min_p(e, x, a)

    def max_p(self, e=0, x=1, a=0):
        """Largest ``p`` on the grid (200), after checking ``(e, x, a)`` (``ValueError``)."""
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return self._max_p(e, x, a)

    def _min_e(self, p, x, a):
        """Smallest eccentricity on the grid: ``0.0``."""
        return 0.0

    def _max_e(self, p, x, a):
        """Largest eccentricity on the flux grid at ``(p, x, a)`` (no input checks).

        ``EMAX = 0.9`` where the grid allows it; closer to the separatrix, the root of
        ``w(e) = 1`` (:func:`_emax_w`), further limited so that ``p`` stays above the
        separatrix plus buffer (:func:`_emax_sep`).

        Args:
            p (float): Semi-latus rectum in units of :math:`M`.
            x (float): ``+1`` or ``-1``.
            a (float): Dimensionless spin.

        Returns:
            float: The maximum eccentricity.

        Raises:
            ValueError: If ``p`` is below the circular separatrix plus buffer.
        """
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
        """Smallest eccentricity (0), after checking ``(p, x, a)`` (``ValueError``)."""
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._min_e(p, x, a)

    def max_e(self, p=20, x=1, a=0):
        """Largest eccentricity on the grid at ``(p, x, a)`` (:meth:`_max_e`), after checks.

        Raises:
            ValueError: If ``x``, ``p`` or ``a`` is out of range, or ``p`` is below the
                circular separatrix plus buffer.
        """
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._max_e(p, x, a)

    def _min_a(self, p, e, x):
        """Smallest spin on the grid: ``-AMAX = -0.999``."""
        return -AMAX

    def _max_a(self, p, e, x):
        """Largest spin on the grid: ``AMAX = 0.999``."""
        return AMAX

    def min_a(self, p=20, e=0, x=1):
        """Smallest spin (-0.999), after checking ``(p, e, x)`` (``ValueError``)."""
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return self._min_a(p, e, x)

    def max_a(self, p=20, e=0, x=1):
        """Largest spin (0.999), after checking ``(p, e, x)`` (``ValueError``)."""
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return self._max_a(p, e, x)

    def bounds_a(self, p=20, e=0, x=1, a_buffer=[0, 0]):
        """Return the spin range ``[-0.999 + a_buffer[0], 0.999 - a_buffer[1]]`` after checks."""
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return [self._min_a(p, e, x) + a_buffer[0], self._max_a(p, e, x) - a_buffer[1]]

    def bounds_p(self, e=0, x=1, a=0, p_buffer=[0, 0]):
        """Return the ``p`` range ``[_min_p + p_buffer[0], 200 - p_buffer[1]]`` at ``(e, x, a)``.

        Raises:
            ValueError: If ``x``, ``e`` or ``a`` is out of range.
        """
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return [self._min_p(e, x, a) + p_buffer[0], self._max_p(e, x, a) - p_buffer[1]]

    def bounds_e(self, p=20, x=1, a=0, e_buffer=[0, 0]):
        """Return the ``e`` range ``[e_buffer[0], _max_e - e_buffer[1]]`` at ``(p, x, a)``.

        Raises:
            ValueError: If ``x``, ``p`` or ``a`` is out of range.
        """
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return [self._min_e(p, x, a) + e_buffer[0], self._max_e(p, x, a) - e_buffer[1]]

    def isvalid_pex(
        self, p=20, e=0, x=1, a=0, p_buffer=[0, 0], e_buffer=[0, 0], a_buffer=[0, 0]
    ):
        """Check that a start point lies on the flux grid (called before every trajectory).

        Args:
            p (float, optional): Semi-latus rectum in units of :math:`M`. Default ``20``.
            e (float, optional): Eccentricity. Default ``0``.
            x (float, optional): ``+1`` or ``-1``. Default ``1``.
            a (float, optional): Dimensionless spin. Default ``0``.
            p_buffer (list[float], optional): Margins on the ``p`` range (a backwards
                integration passes ``[-1e-6, 0]``). Default ``[0, 0]``.
            e_buffer (list[float], optional): Margins on the ``e`` range. Default ``[0, 0]``.
            a_buffer (list[float], optional): Margins on the ``a`` range. Default ``[0, 0]``.

        Raises:
            ValueError: If ``x``, ``e`` (against ``EMAX`` only) or ``a`` is out of range.
            AssertionError: If ``p`` is outside :meth:`bounds_p` (an ``assert``).
        """
        self.isvalid_x(x)
        self.isvalid_e(e, e_buffer=e_buffer)
        self.isvalid_a(a, a_buffer=a_buffer)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer)
        assert p >= pmin and p <= pmax, (
            f"Interpolation: p {p} out of bounds. Must be between {pmin} and {pmax}."
        )

    def distance_to_outer_boundary(self, y):
        """Return the signed distance to the grid's outer edge (backwards stop check).

        Args:
            y (np.ndarray): State.

        Returns:
            float: ``min(|200 - p|, |e_max(p) - e|)`` (each edge moved in by ``1e-9``),
            negative once either edge is passed.
        """
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
        """Interpolate the orbital-element (or ELQ) fluxes at ``(p, e, x, a)``.

        Maps the point to ``(u, w, z)`` (retrograde ``x = -1`` as prograde with spin
        ``-a``), checks it lies on the grid (with a ``1e-8`` tolerance), picks region A or B,
        and returns the spline value times the PN-like normalisation, with the sign of a
        decaying orbit.

        Args:
            p: Semi-latus rectum in units of :math:`M`.
            e: Eccentricity.
            x: ``+1`` or ``-1``. Default ``1``.
            a: Dimensionless spin. Default ``0``.
            pLSO: Separatrix ``p_sep(a, e, x)``. Default ``None`` computes it (the RHS
                passes the cached ``p_sep_cache``).

        Returns:
            tuple[float]: ``(pdot, edot)`` with ``flux_output_convention="pex"``, or
            ``(Edot, Ldot)`` with ``"ELQ"``; derivatives with respect to the
            radiation-reaction time.

        Raises:
            ValueError: If ``p`` is off the grid (``u`` outside ``[0, 1]`` or NaN), or ``e``
                is above the grid edge (``w > 1``) during a backwards integration.
            few.utils.exceptions.TrajectoryOffGridException: If ``w < 0``, if ``w > 1``
                in a forward integration, or if ``a`` is off the grid (``z`` outside
                ``[0, 1]``). It is not a ``ValueError``, so the integrator does not retry
                and it reaches the caller.
        """
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
        """Return the right-hand side at the state ``y`` (the per-stage call of the integrator).

        Evaluates the scalar Kerr geodesic frequencies
        (:func:`~few.utils.geodesic.get_fundamental_frequencies`) and the interpolated
        fluxes (:meth:`interpolate_flux_grids`, using the cached separatrix
        ``p_sep_cache``).

        Args:
            y: State, ``(p, e, x)`` (or ``(E, L, Q)`` with ``use_ELQ``) first.

        Returns:
            list: ``[pdot, edot, 0.0, Omega_phi, Omega_theta, Omega_r]`` in the default
            "pex" convention (``[Edot, Ldot, ...]`` in "ELQ"); ``xdot = 0`` (equatorial),
            the frequencies dimensionless.
        """
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

        Equals the scalar :meth:`~few.trajectory.ode.base.ODEBase.__call__` column by
        column (no exceptions, no Python per-system loop); used by
        :class:`~few.trajectory.integrate.BatchedIntegrate`. The four spline grids are
        extracted once per instance (``_tricubic_grids``).

        Args:
            y: ``(6, S)`` states ``(p, e, x, Phi_phi, Phi_theta, Phi_r)`` (pex only).
            a (float or np.ndarray): Spin, scalar or ``(S,)`` (one per system).

        Returns:
            tuple: ``(ydot (6, S), status (S,))`` with status 0 ok, 1 ``e < 0``, 2 inside
            the separatrix, 3 off the flux grid, 4 a scalar kernel raised (e.g.
            ``e >= 1``); non-zero-status columns are NaN. Negated when
            ``integrate_backwards``.

        Raises:
            NotImplementedError: With ``use_ELQ`` or the "ELQ" flux convention.
        """
        if self.use_ELQ or self.flux_output_convention != "pex":
            raise NotImplementedError("evaluate_rhs_batch: pex convention only")
        y = np.ascontiguousarray(np.asarray(y, dtype=float))
        S = y.shape[1]
        a_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=float), (S,)))
        # multispline's .coefficients copies the full array out of C++ on EVERY access
        # (268 MB per region-A spline): build the four grids once per instance
        grids = getattr(self, "_tricubic_grids", None)
        if grids is None:
            grids = (tricubic_grid(self.pdot_interp_A), tricubic_grid(self.edot_interp_A),
                     tricubic_grid(self.pdot_interp_B), tricubic_grid(self.edot_interp_B))
            self._tricubic_grids = grids
        ydot = np.empty((6, S))
        status = np.zeros(S, dtype=np.int64)
        _kerr_ecc_eq_rhs_batch(y, a_arr, ydot, status, *grids)
        if self.integrate_backwards:
            ydot *= -1.0
        return ydot, status


@njit
def _pdot_PN(p, e, risco, p_sep):
    """PN-like ``pdot`` normalisation of the :class:`KerrEccEqFlux` splines.

    ``8 (1 - e^2)^1.5 (8 + 7 e^2) / (5 p ((p - risco)^2 - (p_sep - risco)^2))``, with
    ``risco`` the circular separatrix and ``p_sep`` the separatrix at ``e``; it diverges at
    the separatrix, so the spline holds a smooth ratio.
    """
    return (8.0 * (1.0 - (e * e)) ** 1.5 * (8.0 + 7.0 * (e * e))) / (
        5.0 * p * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )


@njit
def _edot_PN(p, e, risco, p_sep):
    """PN-like ``edot`` normalisation of the :class:`KerrEccEqFlux` splines.

    ``(1 - e^2)^1.5 (304 + 121 e^2) / (15 p^2 ((p - risco)^2 - (p_sep - risco)^2))``.
    """
    return (((1.0 - (e * e)) ** 1.5) * (304.0 + 121.0 * (e * e))) / (
        15.0
        * (p * p)
        * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )


@njit(fastmath=True)
def _p_to_u(p, p_sep):
    """Legacy ``p`` -> ``u`` grid map of :class:`KerrEccEqFluxLegacy`."""
    return log((p - p_sep + 4.0 - 0.05) / 4)


# TODO check: keep/remove/document -- superseded; no callers; data deprecated; init AttributeError
class KerrEccEqFluxLegacy(ODEBase):
    """
    Kerr eccentric equatorial flux ODE (legacy grids, superseded by :class:`KerrEccEqFlux`).

    Args:
        use_ELQ: Not supported (``supports_ELQ`` is ``False``). Defaults to False.

    Warning:
        The class cannot be constructed as written: it redefines ``flux_output_convention``
        as a read-only property, and :class:`~few.trajectory.ode.base.ODEBase` assigns that
        attribute in its constructor, which raises ``AttributeError``.
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

    Args:
        spline (multispline.spline.TricubicSpline): The spline. Its ``coefficients`` copy
            the whole array out of C++ on every access, so call this once per spline.

    Returns:
        tuple: ``(coeffs, x0, dx, nx, y0, dy, ny, z0, dz, nz)``.
    """
    C = np.asarray(spline.coefficients)
    return (C, float(spline.x0), float(spline.dx), int(C.shape[0]),
            float(spline.y0), float(spline.dy), int(C.shape[1]),
            float(spline.z0), float(spline.dz), int(C.shape[2] // 64))


@njit(fastmath=False)
def _tricubic_one(C, x0, dx, nx, y0, dy, ny, z0, dz, nz, x, y, z):
    """Evaluate a :func:`tricubic_grid` at one point (cell index clamped to the grid)."""
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
    """Evaluate a :func:`tricubic_grid` at the points ``(xs, ys, zs)`` into ``out``."""
    for q in range(xs.shape[0]):
        out[q] = _tricubic_one(C, x0, dx, nx, y0, dy, ny, z0, dz, nz, xs[q], ys[q], zs[q])


def tricubic_eval_batch(grid, x, y, z):
    """Evaluate a :func:`tricubic_grid` at arrays of points (numba, no Python loop).

    Args:
        grid (tuple): Output of :func:`tricubic_grid`.
        x (array-like): First coordinates, shape ``(N,)``.
        y (array-like): Second coordinates, shape ``(N,)``.
        z (array-like): Third coordinates, shape ``(N,)``.

    Returns:
        np.ndarray: Spline values, shape ``(N,)``.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=float))
    out = np.empty(x.shape[0])
    _tricubic_batch(*grid, x, np.ascontiguousarray(np.asarray(y, dtype=float)),
                    np.ascontiguousarray(np.asarray(z, dtype=float)), out)
    return out


@njit(fastmath=False)
def _kerr_ecc_eq_rhs_one(p, e, x, a, gpa, gea, gpb, geb, out):
    """One column; returns the status (0 ok, 1 e<0, 2 inside separatrix, 3 off grid)."""
    edge_buffer = -1e-8
    if not (e >= 0.0):
        return 1
    p_sep = _get_separatrix_kernel_inner(a, e, x, 1e-13)
    if not (p > p_sep):
        return 2
    Om_phi, Om_theta, Om_r = _KerrGeoCoordinateFrequencies_kernel_inner(a, p, e, x)
    a_in = -a if x == -1 else a
    u, w, _yy, z, in_region_A = _kerrecceq_flux_forward_map(a_in, p, e, 1.0, p_sep)
    if (u < edge_buffer or u > 1 - edge_buffer or np.isnan(u) or w < edge_buffer
            or w > 1 - edge_buffer or z < edge_buffer or z > 1 - edge_buffer):
        return 3
    risco = _get_separatrix_kernel_inner(a_in, 0.0, 1.0, 1e-13)
    pdotPN = _pdot_PN(p, e, risco, p_sep)
    edotPN = _edot_PN(p, e, risco, p_sep)
    if in_region_A:
        pdot = -_tricubic_one(gpa[0], gpa[1], gpa[2], gpa[3], gpa[4], gpa[5], gpa[6], gpa[7], gpa[8], gpa[9], u, w, z) * pdotPN
        edot = -_tricubic_one(gea[0], gea[1], gea[2], gea[3], gea[4], gea[5], gea[6], gea[7], gea[8], gea[9], u, w, z) * edotPN
    else:
        pdot = -_tricubic_one(gpb[0], gpb[1], gpb[2], gpb[3], gpb[4], gpb[5], gpb[6], gpb[7], gpb[8], gpb[9], u, w, z) * pdotPN
        edot = -_tricubic_one(geb[0], geb[1], geb[2], geb[3], geb[4], geb[5], geb[6], geb[7], geb[8], geb[9], u, w, z) * edotPN
    out[0] = pdot
    out[1] = edot
    out[2] = 0.0
    out[3] = Om_phi
    out[4] = Om_theta
    out[5] = Om_r
    return 0


@njit(fastmath=False)
def _kerr_ecc_eq_rhs_batch(y, a_arr, ydot, status, gpa, gea, gpb, geb):
    """Status 4: a scalar kernel RAISED for this column (no separatrix bracket, elliptic
    modulus out of range, ...); the scalar path raises, a batch keeps it per column."""
    out = np.empty(6)
    for s in range(y.shape[1]):
        try:
            st = _kerr_ecc_eq_rhs_one(y[0, s], y[1, s], y[2, s], a_arr[s], gpa, gea, gpb, geb, out)
        except Exception:
            st = 4
        status[s] = st
        if st == 0:
            ydot[:, s] = out
        else:
            ydot[:, s] = np.nan
