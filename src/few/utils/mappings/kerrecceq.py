r"""Interpolation-grid coordinates of the Kerr eccentric equatorial (KerrEccEq) model.

The flux (:class:`~few.trajectory.ode.flux.KerrEccEqFlux`) and amplitude
(:class:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq`) data of
``FastKerrEccentricEquatorialFlux`` are tabulated on unit cubes. This module maps the
"apex" coordinates :math:`(a, p, e, x_I)` to the interpolation coordinates
:math:`(u, w, y, z)` (:func:`kerrecceq_forward_map`) and back
(:func:`kerrecceq_backward_map`), in two regions of :math:`p`:

- Region A (near the separatrix, :math:`p \le p_{\rm sep} + \Delta p_{\max}`):

  .. math::

      u = \mathrm{sgn}(c)\,|c|^{\alpha}, \qquad
      c = \log_2 \frac{p - p_{\rm sep} + \Delta p_{\max} - 2\Delta p_{\min}}
          {\Delta p_{\max} - \Delta p_{\min}},

      w = \frac{e}{e_{\max}(u, z)}, \qquad
      e_{\max}(u, z) = e_{\rm sep} + (e_{\max} - e_{\rm sep})\sqrt{z + u^{\beta}(1 - z)},

  so ``u = 0`` at :math:`p_{\rm sep} + \Delta p_{\min}` and ``u = 1`` at
  :math:`p_{\rm sep} + \Delta p_{\max}`, and the eccentricity range shrinks to
  :math:`e_{\rm sep}` near the separatrix of a rapidly prograde-spinning hole.
  :math:`(\alpha, \beta) = (1/2, 2)` for the fluxes and :math:`(1/3, 3)` for the
  amplitudes.
- Region B (far): ``u`` (``U``) is linear in an inverse square root of ``p`` and
  ``w = e / 0.9``. Fluxes: :math:`U \propto 9^{-1/2} - (p - p_{\rm sep})^{-1/2}`, from
  ``U = 0`` at :math:`p - p_{\rm sep} = 9` to ``U = 1`` at ``p = 200``. Amplitudes:
  :math:`U \propto p_c^{-1/2} - p^{-1/2}` with :math:`p_c = p_{\rm sep} + 9`, from ``U = 0``
  at :math:`p_c` to ``U = 1`` at :math:`p_c + 200`.
- Both regions: :math:`z = (\chi(a) - \chi(a_{\max})) / (\chi(a_{\min}) - \chi(a_{\max}))`
  with :math:`\chi(a) = (1 - a)^{1/3}`, so ``z = 0`` at ``a = 0.999`` (prograde) and
  ``z = 1`` at ``a = -0.999`` (retrograde); :math:`y = (x - x_{\min}) / (1 - x_{\min})`
  with :math:`x_{\min} = 0.05` (only ``x = 1``, ``y = 1``, is supported).

Spin convention: ``a`` is the SIGNED spin, negative for retrograde orbits, with
``xI = +1`` (the convention of
:meth:`~few.utils.baseclasses.KerrEccentricEquatorial.sanity_check_init`); the separatrix
is evaluated as :func:`~few.utils.geodesic.get_separatrix` of ``(|a|, e, sign(a))``.

Production callers: :meth:`~few.trajectory.ode.flux.KerrEccEqFlux.interpolate_flux_grids`
(the scalar :func:`_kerrecceq_flux_forward_map`, once per right-hand side), the flux grid
set-up (:func:`apex_of_uwyz`, :func:`apex_of_UWYZ`), the amplitude call
(``kind="amplitude"``), the initial-parameter check of
:class:`~few.utils.baseclasses.KerrEccentricEquatorial`, and the LISA Analysis Tools EMRI
domain mask.
"""

from math import log
from typing import Optional, Union

import numpy as np
from numba import njit

from ..geodesic import get_separatrix

# Grid constants. Region A: XMIN lower x of the y coordinate; AMAX / AMIN spin range of
# the z coordinate; DELTAPMIN / DELTAPMAX distance to the separatrix at u = 0 / u = 1;
# EMAX / ESEP largest eccentricity far from / at the separatrix (prograde a = AMAX);
# ALPHA_* / BETA_* the u and e-range exponents of the flux / amplitude grids.
# Region B: DPC_REGIONB offset of the amplitude grid's inner edge from the separatrix;
# PMAX_REGIONB outer p (flux) or outer offset (amplitudes); *_REGIONB spin and
# eccentricity ranges; DELTAPMIN_REGIONB inner p - p_sep of the flux grid.
XMIN = 0.05
AMAX = 0.999
AMIN = -AMAX
DELTAPMIN = 0.001
DELTAPMAX = 9 + DELTAPMIN
EMAX = 0.9
ESEP = 0.25

ALPHA_FLUX = 1.0 / 2.0
BETA_FLUX = 2.0
ALPHA_AMP = 1.0 / 3.0
BETA_AMP = 3.0

DPC_REGIONB = DELTAPMAX - 0.001
PMAX_REGIONB = 200
AMAX_REGIONB = 0.999
AMIN_REGIONB = -AMAX_REGIONB
EMAX_REGIONB = 0.9
DELTAPMIN_REGIONB = 9


# TODO check: keep/remove/document -- no callers; superseded by kerrecceq_forward_map
def kerrecceq_legacy_p_to_u(a, p, e, xI, use_gpu=False):
    """Convert the separation :math:`p` to the legacy grid coordinate :math:`u`.

    Legacy mapping ``u = log((p - p_sep + 3.95) / 4)``, clipped at 0, with
    ``p_sep = get_separatrix(a, e, xI)``.

    Args:
        a (double scalar or 1D xp.ndarray): Dimensionless spin.
        p (double scalar or 1D xp.ndarray): Values of separation,
            :math:`p`, to convert.
        e (double scalar or 1D xp.ndarray): Associated eccentricity values
            of :math:`p` necessary for conversion.
        xI (double scalar or 1D xp.ndarray): Cosine of the inclination.
        use_gpu (bool, optional): If True, use Cupy/GPUs. Default is False.

    """
    if use_gpu:
        import cupy as xp
    else:
        import numpy as xp

    scalar = False
    if not hasattr(p, "__len__"):
        scalar = True

    delta_p = 0.05
    alpha = 4.0

    pLSO = get_separatrix(a, e, xI)
    beta = alpha - delta_p
    u = xp.log((p + beta - pLSO) / alpha)

    if xp.any(u < -1e9):
        raise ValueError("u values are too far below zero.")

    # numerical errors
    if scalar:
        u = max(u, 0)
    else:
        u[u < 0.0] = 0.0

    return u


@njit
def _kerrecceq_flux_forward_map(
    a: float,
    p: float,
    e: float,
    xI: float,
    pLSO: float,
):
    r"""Scalar forward map of the flux grids, for the per-step ODE evaluations.

    The jit-compiled hot path of
    :meth:`~few.trajectory.ode.flux.KerrEccEqFlux.interpolate_flux_grids` (called once per
    right-hand side): region A (:math:`p \le p_{\rm sep} + \Delta p_{\max}`, flux
    exponents) or region B (flux variant). No input checks.

    Args:
        a: Signed spin (the caller passes ``-a`` for ``x = -1``).
        p: Semi-latus rectum in units of :math:`M`.
        e: Eccentricity.
        xI: Cosine of the inclination (the caller passes ``1.0``).
        pLSO: Separatrix at the same point (the caller's cached value).

    Returns:
        tuple: ``(u, w, y, z, in_region_A)``, four floats and a bool.
    """
    if p <= pLSO + DELTAPMAX:
        return *_uwyz_of_apex_kernel(a, p, e, xI, pLSO, ALPHA_FLUX, BETA_FLUX), True
    else:
        return *_UWYZ_of_apex_kernel(
            a,
            p,
            e,
            xI,
            pLSO,
            True,
        ), False


def kerrecceq_forward_map(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    xI: Union[float, np.ndarray],
    pLSO: Optional[Union[float, np.ndarray]] = None,
    return_mask: bool = False,
    kind: str = "flux",
):
    """Map apex coordinates :math:`(a, p, e, x_I)` to the KerrEccEq grid coordinates.

    Returns the interpolation coordinates ``(u, w, y, z)`` (see the module docstring for the
    formulas), which lie in ``[0, 1]`` on the grid: region A where
    ``p <= p_sep + DELTAPMAX``, region B elsewhere. Values outside ``[0, 1]`` mean the
    point is off the grid; ``u < 0`` inside the inner edge ``p_sep + DELTAPMIN`` of region
    A, and ``u`` is NaN where the logarithm's argument is negative
    (``p < p_sep - DELTAPMAX + 2 DELTAPMIN``). Used by
    :meth:`~few.utils.baseclasses.KerrEccentricEquatorial.sanity_check_init` (``kind="flux"``),
    :meth:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq.get_amplitudes`
    (``kind="amplitude"``) and the LISA Analysis Tools EMRI domain mask.

    Args:
        a: Signed dimensionless spin, negative for retrograde orbits.
        p: Semi-latus rectum in units of :math:`M`.
        e: Eccentricity.
        xI: Cosine of the inclination; every value must be ``1`` (retrograde orbits use
            a negative ``a``).
        pLSO: Ignored: the separatrix is always recomputed, as
            ``get_separatrix(|a|, e, sign(a) * xI)`` (``sign(0) = +1``).
        return_mask: Also return the region mask, ``True`` in region A. Default ``False``.
        kind: ``"flux"`` (default; exponents ``alpha = 1/2``, ``beta = 2`` and the flux
            region-B ``U``) or ``"amplitude"`` (``1/3``, ``3`` and the amplitude ``U``).

    Returns:
        tuple: ``(u, w, y, z)``, or ``(u, w, y, z, mask)`` with ``return_mask=True``: 1-D
        ``np.ndarray`` of the common input length (at least 1). The outputs are allocated
        like ``a``, so pass a float ``a`` (an integer ``a`` would truncate them).

    Raises:
        ValueError: If any ``xI != 1`` ("Only xI = 1 is supported."), or if ``kind`` is
            neither ``"flux"`` nor ``"amplitude"``.

    Note:
        Inputs are not broadcast: pass all scalars or 1-D arrays of one common length
        (mixing scalars and arrays fails with an ``AssertionError`` or ``IndexError``).

    Example:
        >>> from few.utils.mappings.kerrecceq import (
        ...     kerrecceq_backward_map,
        ...     kerrecceq_forward_map,
        ... )
        >>> u, w, y, z = kerrecceq_forward_map(0.3, 8.0, 0.4, 1.0)
        >>> a, p, e, x = kerrecceq_backward_map(u, w, y, z)
        >>> round(p.item(), 8), round(e.item(), 8)
        (8.0, 0.4)
    """
    xp = np  # TODO: gpu

    if np.any(xI != 1):
        raise ValueError("Only xI = 1 is supported.")

    if kind == "flux":
        is_flux = True
        alpha = ALPHA_FLUX
        beta = BETA_FLUX
    elif kind == "amplitude":
        is_flux = False
        alpha = ALPHA_AMP
        beta = BETA_AMP
    else:
        raise ValueError

    a = xp.atleast_1d(xp.asarray(a))
    p = xp.atleast_1d(xp.asarray(p))
    e = xp.atleast_1d(xp.asarray(e))
    xI = xp.atleast_1d(xp.asarray(xI))

    u = xp.zeros_like(a)
    w = xp.zeros_like(a)
    y = xp.zeros_like(a)
    z = xp.zeros_like(a)

    a_sep_in = xp.abs(a)

    asign = xp.sign(a)
    asign[asign == 0] = 1

    xI_sep_in = asign * xI

    # compute separatrix at all points
    pLSO = get_separatrix(a_sep_in, e, xI_sep_in)

    # handle regions A and B
    near = p <= pLSO + DELTAPMAX
    if xp.any(near):
        out = _uwyz_of_apex_kernel(
            a[near], p[near], e[near], xI[near], pLSO[near], alpha, beta
        )
        u[near] = out[0]
        w[near] = out[1]
        y[near] = out[2]
        z[near] = out[3]

    far = np.bitwise_not(near)
    if xp.any(far):
        out = _UWYZ_of_apex_kernel(a[far], p[far], e[far], xI[far], pLSO[far], is_flux)
        u[far] = out[0]
        w[far] = out[1]
        y[far] = out[2]
        z[far] = out[3]

    if return_mask:
        return u, w, y, z, near
    else:
        return u, w, y, z


def kerrecceq_backward_map(
    u: Union[float, np.ndarray],
    w: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    regionA: bool = True,
    kind: str = "flux",
):
    """Map KerrEccEq grid coordinates ``(u, w, y, z)`` back to apex coordinates.

    The inverse of :func:`kerrecceq_forward_map` for one region at a time: region A
    (:func:`apex_of_uwyz`) or region B (:func:`apex_of_UWYZ`); the caller must know the
    region of the points (e.g. from the forward map's mask). Used by the Amplitude tutorial
    to build grids.

    Args:
        u: Radial grid coordinate (``U`` in region B).
        w: Eccentricity grid coordinate.
        y: Inclination grid coordinate; every value must be ``1`` (``xI = 1``).
        z: Spin grid coordinate.
        regionA: Invert the region-A map (default ``True``) or the region-B map.
        kind: ``"flux"`` (default) or ``"amplitude"`` grid parameters.

    Returns:
        tuple: ``(a, p, e, xI)``: signed spin, ``p`` in units of :math:`M`, eccentricity and
        ``xI`` (``1``), as ``np.ndarray`` of the input length.

    Raises:
        ValueError: If any ``y != 1`` ("Only xI = 1 is supported."), or if ``kind`` is
            neither ``"flux"`` nor ``"amplitude"``.

    Note:
        Pass 1-D arrays (e.g. the forward map's outputs). The scalar shortcut (inputs
        without ``__len__``) fails in the separatrix step of :func:`apex_of_uwyz` /
        :func:`apex_of_UWYZ` (item assignment on the NumPy scalar returned by ``np.sign``).

    Example:
        >>> from few.utils.mappings.kerrecceq import (
        ...     kerrecceq_backward_map,
        ...     kerrecceq_forward_map,
        ... )
        >>> u, w, y, z = kerrecceq_forward_map(0.3, 45.0, 0.4, 1.0, kind="amplitude")
        >>> a, p, e, x = kerrecceq_backward_map(u, w, y, z, regionA=False, kind="amplitude")
        >>> round(p.item(), 8)
        45.0
    """
    xp = np  # TODO: gpu

    if np.any(np.asarray(y) != 1):
        raise ValueError("Only xI = 1 is supported.")

    is_flux = kind == "flux"
    is_amp = kind == "amplitude"
    if is_flux:
        alpha = ALPHA_FLUX
        beta = BETA_FLUX
    elif is_amp:
        alpha = ALPHA_AMP
        beta = BETA_AMP
    else:
        raise ValueError

    # if scalar directly evaluate the kernel for speed
    if not hasattr(u, "__len__"):
        if regionA:
            return apex_of_uwyz(u, w, y, z, alpha=alpha, beta=beta)
        else:
            return apex_of_UWYZ(u, w, y, z, is_flux)

    # else, we have multiple points
    u = xp.atleast_1d(xp.asarray(u))
    w = xp.atleast_1d(xp.asarray(w))
    y = xp.atleast_1d(xp.asarray(y))
    z = xp.atleast_1d(xp.asarray(z))

    if regionA:
        return apex_of_uwyz(u, w, y, z, alpha=alpha, beta=beta)
    else:
        return apex_of_UWYZ(u, w, y, z, is_flux)


@njit
def u_of_p(p, pLSO, alpha):
    """Region-A ``u`` of ``p``: signed power ``alpha`` of the log2 distance to the separatrix.

    ``u = 0`` at ``pLSO + DELTAPMIN``, ``u = 1`` at ``pLSO + DELTAPMAX``; scalar or array.
    """
    check_term = (
        np.log(p - pLSO + DELTAPMAX - 2 * DELTAPMIN) - log(DELTAPMAX - DELTAPMIN)
    ) / log(2)
    sgn = np.sign(check_term)
    return sgn * (sgn * check_term) ** alpha


@njit
def u_of_p_flux(p, pLSO):
    """Scalar :func:`u_of_p` with the flux exponent ``ALPHA_FLUX = 1/2``."""
    check_term = (
        np.log(p - pLSO + DELTAPMAX - 2 * DELTAPMIN) - log(DELTAPMAX - DELTAPMIN)
    ) / log(2)
    if check_term < 0:
        return -((-check_term) ** ALPHA_FLUX)
    else:
        return check_term**ALPHA_FLUX


@njit
def y_of_x(x):
    """Inclination grid coordinate ``y = (x - XMIN) / (1 - XMIN)`` (``y = 1`` at ``x = 1``)."""
    return (x - XMIN) / (1 - XMIN)


@njit
def chi_of_a(a):
    """Spin variable ``chi = (1 - a)^(1/3)`` of the ``z`` coordinate (signed ``a``)."""
    return (1 - a) ** (1 / 3)


@njit
def chi2_of_a(a):
    """Alternative spin variable ``(1 - a)^(2/3)`` (for :func:`z2_of_a`; unused by FEW)."""
    return (1 - a) ** (2 / 3)


@njit
def z_of_a(a):
    """Region-A spin coordinate ``z`` in ``[0, 1]``: 0 at ``a = AMAX``, 1 at ``a = AMIN``."""
    chimax = chi_of_a(AMIN)
    chimin = chi_of_a(AMAX)
    return (chi_of_a(a) - chimin) / (chimax - chimin)


@njit
def z2_of_a(a):
    """Alternative spin coordinate built on :func:`chi2_of_a` (unused by FEW)."""
    chimax = chi2_of_a(AMIN)
    chimin = chi2_of_a(AMAX)
    return (chi2_of_a(a) - chimin) / (chimax - chimin)


@njit
def Secc_of_uz(
    u,
    z,
    beta,
):
    """Largest grid eccentricity of region A at ``(u, z)``.

    ``ESEP + (EMAX - ESEP) * sqrt(z + u^beta (1 - z))`` (signed square root): ``ESEP`` at
    ``u = z = 0``, ``EMAX`` at ``u = 1`` or ``z = 1``.
    """
    check_part = z + u**beta * (1 - z)
    sgn = np.sign(check_part)
    return ESEP + (EMAX - ESEP) * sgn * np.sqrt(sgn * check_part)


@njit
def w_of_euz(e, u, z, beta):
    """Region-A eccentricity coordinate ``w = e / Secc_of_uz(u, z, beta)``."""
    return e / Secc_of_uz(u, z, beta)


@njit
def w_of_euz_flux(e, u, z):
    """:func:`w_of_euz` with the flux exponent ``BETA_FLUX = 2``."""
    return e / Secc_of_uz(u, z, BETA_FLUX)


@njit
def p_of_u(u, pLSO, alpha):
    """Inverse of :func:`u_of_p` for ``u >= 0``.

    ``p = pLSO + DELTAPMIN + (DELTAPMAX - DELTAPMIN) * (2^(u^(1/alpha)) - 1)``.
    """
    return (pLSO + DELTAPMIN) + (DELTAPMAX - DELTAPMIN) * (
        np.exp(u ** (1 / alpha) * log(2)) - 1
    )


@njit
def p_of_u_flux(u, pLSO):
    """:func:`p_of_u` with the flux exponent ``ALPHA_FLUX = 1/2``."""
    return (pLSO + DELTAPMIN) + (DELTAPMAX - DELTAPMIN) * (
        np.exp(u ** (1 / ALPHA_FLUX) * log(2)) - 1
    )


@njit
def x_of_y(y):
    """Inverse of :func:`y_of_x`."""
    return y * (1 - XMIN) + XMIN


@njit
def a_of_chi2(chi):
    """Inverse of :func:`chi2_of_a` (unused by FEW)."""
    return 1 - chi ** (1.5)


@njit
def a_of_z2(z):
    """Inverse of :func:`z2_of_a` (unused by FEW)."""
    chimax = chi2_of_a(AMIN)
    chimin = chi2_of_a(AMAX)
    return a_of_chi2(chimin + z * (chimax - chimin))


@njit
def a_of_chi(chi):
    """Inverse of :func:`chi_of_a`: ``a = 1 - chi^3``."""
    return 1 - chi**3


@njit
def a_of_z(z):
    """Inverse of :func:`z_of_a`: the signed spin of a region-A ``z``."""
    chimax = chi_of_a(AMIN)
    chimin = chi_of_a(AMAX)
    return a_of_chi(chimin + z * (chimax - chimin))


@njit
def e_of_uwz(u, w, z, beta):
    """Inverse of :func:`w_of_euz`: ``e = w * Secc_of_uz(u, z, beta)``."""
    return Secc_of_uz(u, z, beta) * w


@njit
def e_of_uwz_flux(u, w, z):
    """:func:`e_of_uwz` with the flux exponent ``BETA_FLUX = 2``."""
    return Secc_of_uz(u, z, BETA_FLUX) * w


@njit
def u_where_w_is_unity(e, z, kind="flux"):
    """Region-A ``u`` at which the eccentricity ``e`` lies on the grid edge (``w = 1``).

    Solves ``Secc_of_uz(u, z, beta) = e`` for ``u``. Used by
    :meth:`~few.trajectory.ode.flux.KerrEccEqFlux._min_p` when ``e`` is above the edge at
    ``u = 0``.

    Args:
        e (float): Eccentricity.
        z (float): Spin coordinate.
        kind (str, optional): ``"flux"`` (default, ``beta = 2``) or ``"amplitude"``
            (``beta = 3``).

    Returns:
        float: ``u``; ``0.0`` if ``z == 1``; NaN if ``e < ESEP``.
    """
    if kind == "flux":
        beta = BETA_FLUX
    elif kind == "amplitude":
        beta = BETA_AMP

    if z == 1:
        return 0.0

    if e < ESEP:
        return np.nan

    part = ((e - ESEP) / (EMAX - ESEP)) ** 2
    inside_root = (part - z) / (1 - z)
    return inside_root ** (1 / beta)


@njit
def _uwyz_of_apex_kernel(
    a,
    p,
    e,
    x,
    pLSO,
    alpha,
    beta,
):
    """Region-A forward map: ``(u, w, y, z)`` for signed spin ``a`` and separatrix ``pLSO``."""
    u = u_of_p(p, pLSO, alpha)
    y = y_of_x(x)
    z = z_of_a(a)
    w = w_of_euz(e, u, z, beta)
    return u, w, y, z


def apex_of_uwyz(
    u,
    w,
    y,
    z,
    alpha=ALPHA_FLUX,
    beta=BETA_FLUX,
):
    """Region-A inverse map: ``(u, w, y, z)`` to ``(a, p, e, x)``.

    ``a = a_of_z(z)`` (signed), ``x = x_of_y(y)``, ``e = e_of_uwz(u, w, z, beta)``, then the
    separatrix ``get_separatrix(|a|, e, sign(a))`` and ``p = p_of_u(u, pLSO, alpha)``. Used
    by :class:`~few.trajectory.ode.flux.KerrEccEqFlux` to place its flux grid nodes.

    Args:
        u (np.ndarray): Radial coordinate.
        w (np.ndarray): Eccentricity coordinate.
        y (np.ndarray): Inclination coordinate (``1`` for ``x = 1``).
        z (np.ndarray): Spin coordinate.
        alpha (float, optional): ``u`` exponent. Default ``ALPHA_FLUX`` (``1/2``).
        beta (float, optional): Eccentricity-range exponent. Default ``BETA_FLUX`` (``2``).

    Returns:
        tuple: ``(a, p, e, x)`` arrays of the input shape.
    """
    a = a_of_z(z)
    x = x_of_y(y)
    e = e_of_uwz(
        u,
        w,
        z,
        beta,
    )
    a = np.asarray(a)
    a_in = np.abs(a)
    x_in = np.sign(a)
    x_in[x_in == 0] = 1

    pLSO = get_separatrix(a_in, e, x_in)
    p = p_of_u(u, pLSO, alpha)
    return a, p, e, x


# # Region B


@njit
def U_of_p_flux(p, pLSO):
    """Region-B flux ``U``, linear in ``(p - pLSO)^(-1/2)``.

    ``U = 0`` at ``p - pLSO = DELTAPMIN_REGIONB`` (9), ``U = 1`` at ``p = PMAX_REGIONB`` (200).
    """
    return ((DELTAPMIN_REGIONB) ** (-0.5) - (p - pLSO) ** (-0.5)) / (
        (DELTAPMIN_REGIONB) ** (-0.5) - (PMAX_REGIONB - pLSO) ** (-0.5)
    )


@njit
def p_of_U_flux(U, pLSO):
    """Inverse of :func:`U_of_p_flux`."""
    return (
        (DELTAPMIN_REGIONB) ** (-0.5)
        - U * ((DELTAPMIN_REGIONB) ** (-0.5) - (PMAX_REGIONB - pLSO) ** (-0.5))
    ) ** (-2) + pLSO


@njit
def U_of_p_amplitude(p, pLSO):
    """Region-B amplitude ``U``: linear in ``p^(-1/2)``; 0 at ``pLSO + 9``, 1 at ``pLSO + 209``."""
    pc = pLSO + DPC_REGIONB
    return (pc ** (-0.5) - p ** (-0.5)) / (pc ** (-0.5) - (PMAX_REGIONB + pc) ** (-0.5))


@njit
def p_of_U_amplitude(U, pLSO):
    """Inverse of :func:`U_of_p_amplitude`."""
    pc = pLSO + DPC_REGIONB
    return (pc ** (-0.5) - U * (pc ** (-0.5) - (PMAX_REGIONB + pc) ** (-0.5))) ** (-2)


@njit
def W_of_e(e):
    """Region-B eccentricity coordinate ``W = e / EMAX_REGIONB`` (``0.9``)."""
    return e / EMAX_REGIONB


@njit
def e_of_W(w):
    """Inverse of :func:`W_of_e`."""
    return w * EMAX_REGIONB


@njit
def Y_of_x(x):
    """Region-B inclination coordinate (same as :func:`y_of_x`)."""
    return y_of_x(x)


@njit
def x_of_Y(y):
    """Inverse of :func:`Y_of_x`."""
    return x_of_y(y)


@njit
def Z_of_a(a):
    """Region-B spin coordinate (the :func:`z_of_a` form with the ``*_REGIONB`` spin range)."""
    chimax = chi_of_a(AMIN_REGIONB)
    chimin = chi_of_a(AMAX_REGIONB)
    return (chi_of_a(a) - chimin) / (chimax - chimin)


@njit
def a_of_Z(z):
    """Inverse of :func:`Z_of_a`."""
    chimax = chi_of_a(AMIN_REGIONB)
    chimin = chi_of_a(AMAX_REGIONB)
    return a_of_chi(chimin + z * (chimax - chimin))


@njit
def _UWYZ_of_apex_kernel(
    a,
    p,
    e,
    x,
    pLSO,
    is_flux,
):
    """Region-B forward map: returns ``(U, W, Y, Z)`` (flux or amplitude ``U``)."""
    if is_flux:
        u = U_of_p_flux(p, pLSO)
    else:
        u = U_of_p_amplitude(p, pLSO)
    y = Y_of_x(x)
    z = Z_of_a(a)
    w = W_of_e(e)
    return u, w, y, z


def apex_of_UWYZ(
    u,
    w,
    y,
    z,
    is_flux,
):
    """Region-B inverse map: ``(U, W, Y, Z)`` to ``(a, p, e, x)``.

    Like :func:`apex_of_uwyz` with the region-B coordinates; ``is_flux`` selects the flux
    (``True``) or amplitude ``U``. Used by :class:`~few.trajectory.ode.flux.KerrEccEqFlux`
    to place its region-B flux grid nodes.

    Args:
        u (np.ndarray): Radial coordinate ``U``.
        w (np.ndarray): Eccentricity coordinate ``W``.
        y (np.ndarray): Inclination coordinate.
        z (np.ndarray): Spin coordinate ``Z``.
        is_flux (bool): Flux (``True``) or amplitude grid.

    Returns:
        tuple: ``(a, p, e, x)`` arrays of the input shape.
    """
    a = a_of_Z(z)
    x = x_of_Y(y)
    e = e_of_W(w)

    a = np.asarray(a)
    a_in = np.abs(a)
    x_in = np.sign(a)
    x_in[x_in == 0] = 1

    pLSO = get_separatrix(a_in, e, x_in)
    if is_flux:
        p = p_of_U_flux(u, pLSO)
    else:
        p = p_of_U_amplitude(u, pLSO)
    return a, p, e, x
