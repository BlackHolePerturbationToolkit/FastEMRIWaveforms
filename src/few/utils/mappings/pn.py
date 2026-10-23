r"""Conversions between the inclination parameters :math:`Y = \cos\iota` and :math:`x_I = \cos I`.

:math:`x_I = \cos I` describes the inclination of the orbit from the equatorial plane (the
parameter of the FEW geodesic, flux and Kerr waveform code); :math:`Y = \cos\iota` with
:math:`\cos\iota = L_z/\sqrt{L_z^2 + Q}` is the inclination parameter of the 5PN trajectory
(:class:`~few.trajectory.ode.pn5.PN5`) and of the AAK waveform. :func:`Y_to_xI` goes through
the constants of motion of a given :math:`Y` (the numba kernels ``_PN_E``, ``_PN_L``,
``_PN_C``); :func:`xI_to_Y` uses :func:`~few.utils.geodesic.get_kerr_geo_constants_of_motion`.
:func:`Y_to_xI` returns ``Y = +-1`` (equatorial) and ``Y = 0`` (polar) unchanged.

In the private kernels the first argument ``q`` is the dimensionless spin :math:`a` of the
primary (the name comes from the original code; it is not a mass ratio).
"""

from math import sqrt
from typing import Union

import numpy as np
from numba import njit

from ..geodesic import get_kerr_geo_constants_of_motion


@njit(fastmath=False)
def _d1(q, p, e, Y):
    """Auxiliary function ``d`` at periapsis ``r = p/(1 + e)`` (``q`` is the spin ``a``).

    The ``*1`` / ``*2`` kernels (``d, f, g, h`` at periapsis / apoapsis) feed
    :func:`_PN_E`; each raises ``ValueError`` if ``|q| >= 1``, ``|e| >= 1``, ``p <= 1`` or
    ``|Y| > 1``.
    """
    p2 = p * p
    q2 = q * q
    inv_ep4 = 1.0 / ((1.0 + e) ** 4)

    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _d1")
    elif e == 0.0:
        return (p2 + q2 - 2.0 * p) * p2
    else:
        return (
            p2 * ((q2 * e * e) + (2.0 * q2 - 2.0 * p) * e + p2 + q2 - 2.0 * p) * inv_ep4
        )


@njit(fastmath=False)
def _f1(q, p, e, Y):
    """Auxiliary function ``f`` at periapsis (see :func:`_d1`)."""
    p2 = p * p
    q2 = q * q
    inv_ep4 = 1.0 / ((1.0 + e) ** 4)

    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _f1")
    elif e == 0.0:
        return p2 * p2 + p2 * q2 + 2.0 * p * q2
    else:
        return (
            2.0
            * p
            * ((1.0 + e) ** 2 * (e + p / 2.0 + 1.0) * q2 + (p2 * p) / 2.0)
            * inv_ep4
        )


@njit(fastmath=False)
def _g1(q, p, e, Y):
    """Auxiliary function ``g`` at periapsis (see :func:`_d1`)."""
    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _g1")
    elif e == 0.0:
        return 2.0 * q * p
    else:
        return 2.0 * q * p / (1.0 + e)


@njit(fastmath=False)
def _h1(q, p, e, Y):
    """Auxiliary function ``h`` at periapsis (see :func:`_d1`); depends on ``Y``."""
    q2 = q * q
    p2 = p * p
    Y2 = Y * Y
    y2 = 1.0 - Y2
    inv_ep2 = 1.0 / ((1.0 + e) ** 2)

    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _h1")
    elif e == 0.0 and Y == 1.0:
        return 2.0 * q * p
    elif e == 0.0:
        return (p2 - 2.0 * p + y2 * q2) / Y2
    elif Y == 1.0:
        return (-2.0 * p * e + p2 - 2.0 * p) * inv_ep2
    else:
        return (
            (y2 * q2 * e * e + (2.0 * y2 * q2 - 2.0 * p) * e + y2 * q2 + p2 - 2.0 * p)
            * inv_ep2
            / Y2
        )


@njit(fastmath=False)
def _d2(q, p, e, Y):
    """Auxiliary function ``d`` at apoapsis ``r = p/(1 - e)`` (see :func:`_d1`)."""
    q2 = q * q
    p2 = p * p
    inv_em4 = 1.0 / ((-1.0 + e) ** 4)

    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _d2")
    elif e == 0.0:
        return 4.0 * p2 * p + 2.0 * p * q2 - 6.0 * p2
    else:
        return (
            p2
            * ((q2 * e * e) + (-2.0 * q2 + 2.0 * p) * e + p2 + q2 - 2.0 * p)
            * inv_em4
        )


@njit(fastmath=False)
def _f2(q, p, e, Y):
    """Auxiliary function ``f`` at apoapsis (see :func:`_d1`)."""
    q2 = q * q
    p2 = p * p
    inv_em4 = 1.0 / ((-1.0 + e) ** 4)

    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _f2")
    elif e == 0.0:
        return (2.0 * p + 2.0) * q2 + 4.0 * p2 * p
    else:
        return (
            -2.0
            * ((e - p / 2.0 - 1.0) * (-1.0 + e) ** 2 * q2 - (p2 * p) / 2.0)
            * p
            * inv_em4
        )


@njit(fastmath=False)
def _g2(q, p, e, Y):
    """Auxiliary function ``g`` at apoapsis (see :func:`_d1`)."""
    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _g2")
    elif e == 0.0:
        return 2.0 * q
    else:
        return -2.0 * q * p / (-1.0 + e)


@njit(fastmath=False)
def _h2(q, p, e, Y):
    """Auxiliary function ``h`` at apoapsis (see :func:`_d1`); depends on ``Y``."""
    q2 = q * q
    p2 = p * p
    Y2 = Y * Y
    y2 = 1.0 - Y2
    inv_em2 = 1.0 / ((-1.0 + e) ** 2)

    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _h2")
    elif e == 0.0 and Y == 1.0:
        return 2.0 * p - 2.0
    elif e == 0.0:
        return (2.0 * p - 2.0) / Y2
    elif Y == 1.0:
        return (2.0 * p * e + p2 - 2.0 * p) * inv_em2
    else:
        return (
            (y2 * q2 * e * e + (-2.0 * y2 * q2 + 2.0 * p) * e + y2 * q2 + p2 - 2.0 * p)
            * inv_em2
            / Y2
        )


@njit(fastmath=False)
def _PN_E(q, p, e, Y):
    """Energy per unit mass of the orbit ``(q = a, p, e, Y)``.

    Retrograde (sign flip of the square-root term) when exactly one of ``Y`` and ``q`` is
    negative.
    """
    d1, f1, g1, h1 = _d1(q, p, e, Y), _f1(q, p, e, Y), _g1(q, p, e, Y), _h1(q, p, e, Y)
    d2, f2, g2, h2 = _d2(q, p, e, Y), _f2(q, p, e, Y), _g2(q, p, e, Y), _h2(q, p, e, Y)

    kappa = d1 * h2 - d2 * h1
    epsilon = d1 * g2 - d2 * g1
    rho = f1 * h2 - f2 * h1
    eta = f1 * g2 - f2 * g1
    sigma = g1 * h2 - g2 * h1

    rhs_sqrt = sigma * (
        sigma * epsilon * epsilon + rho * epsilon * kappa - eta * kappa * kappa
    )
    rhs_numer = kappa * rho + 2.0 * epsilon * sigma
    rhs_denom = rho * rho + 4.0 * eta * sigma
    # Determine if orbit is prograde (1) or retrograde (-1)
    grade = - 1 if (Y < 0) ^ (q < 0)   else 1 # Retrograde if (Y<0) xor (q<0), else prograde
    E_square = (rhs_numer -grade* 2.0 * sqrt(rhs_sqrt)) / rhs_denom
    return sqrt(E_square)


@njit(fastmath=False)
def _PN_L(q, p, e, Y):
    """Axial angular momentum per unit mass :math:`L_z` (sign of ``Y``) of ``(q = a, p, e, Y)``."""
    E = _PN_E(q, p, e, Y)
    h2 = _h2(q, p, e, Y)
    rhs_first_term = -_g2(q, p, e, Y) * E / h2
    rhs_sqrt = rhs_first_term**2 + (_f2(q, p, e, Y) * E * E - _d2(q, p, e, Y)) / h2

    sgnY = 1 if Y > 0 else -1
    return rhs_first_term + sgnY * sqrt(rhs_sqrt)


@njit(fastmath=False)
def _PN_C(q, p, e, Y):
    """Carter constant ``Q = L_z^2 (1/Y^2 - 1)`` of ``(q = a, p, e, Y)``; ``0`` for ``Y = 1``."""
    L = _PN_L(q, p, e, Y)
    if Y == 1.0:
        return 0.0
    else:
        return L * L * ((1.0 / (Y * Y)) - 1.0)


@njit(fastmath=False)
def _Y_to_xI_kernel_inner(a, p, e, Y):
    """Scalar kernel of :func:`Y_to_xI`: ``x_I`` from the ``(E, L_z, Q)`` of ``Y``.

    Returns ``Y`` unchanged for ``|Y| == 1`` or ``Y == 0``.
    """
    if abs(Y) == 1 or Y == 0.0:
        return Y

    E = _PN_E(a, p, e, Y)
    Lz = _PN_L(a, p, e, Y)
    Q = _PN_C(a, p, e, Y)

    a2 = a * a
    Lz2 = Lz * Lz
    E2m1 = (E - 1) * (E + 1.0)
    QpLz2ma2omE2 = Q + Lz2 + a2 * E2m1
    denomsqr = QpLz2ma2omE2 + sqrt(QpLz2ma2omE2 * QpLz2ma2omE2 - 4.0 * Lz2 * a2 * E2m1)
    xI = sqrt(2.0) * Lz / sqrt(denomsqr)
    return xI


@njit
def _Y_to_xI_kernel(xI, a, p, e, Y):
    """Array loop of :func:`_Y_to_xI_kernel_inner`; fills ``xI`` in place (0 where ``Y == 0``)."""
    for i in range(len(xI)):
        Y_h = Y[i]
        if Y_h != 0.0:
            xI[i] = _Y_to_xI_kernel_inner(a[i], p[i], e[i], Y_h)
        else:
            xI[i] = 0.0


def Y_to_xI(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    Y: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Convert :math:`Y = \cos\iota` to :math:`x_I = \cos I`.

    :math:`x_I = \cos I`, with :math:`I` the inclination of the orbit from the equatorial
    plane; :math:`Y = \cos\iota = L_z/\sqrt{L_z^2 + Q}`. Computes :math:`(E, L_z, Q)` of the
    orbit with the given ``Y`` and converts them to :math:`x_I` with the relation used by
    :func:`~few.utils.geodesic.ELQ_to_pex`. ``Y = +-1`` and ``Y = 0`` are returned unchanged.
    Inverse: :func:`xI_to_Y`. Used by the trajectory for models that integrate ``Y``
    (``convert_Y``, e.g. :class:`~few.trajectory.ode.pn5.PN5`).

    Args:
        a: Dimensionless spin, ``|a| < 1``. Scalar (broadcast to the length of ``e`` on the
            array path) or an array of that length.
        p: Semi-latus rectum in units of :math:`M`, ``p > 1``. A scalar selects the scalar
            path (all inputs scalars); an array selects the array path.
        e: Eccentricity, ``|e| < 1``; an array of the length of ``p`` on the array path.
        Y: :math:`Y = \cos\iota`, ``|Y| <= 1``; an array of the length of ``p`` on the
            array path.

    Returns:
        float or np.ndarray: :math:`x_I`; a float on the scalar path, else a 1-D array of
        the length of ``e``.

    Raises:
        ValueError: From the numba kernels if ``|a| >= 1``, ``|e| >= 1``, ``p <= 1`` or
            ``|Y| > 1`` (``Y`` neither ``+-1`` nor ``0``).
        AssertionError: If ``a`` is an array of another length than ``e``.

    Example:
        >>> from few.utils.mappings.pn import Y_to_xI, xI_to_Y
        >>> xI = Y_to_xI(0.3, 8.0, 0.4, 0.3)
        >>> round(float(xI_to_Y(0.3, 8.0, 0.4, xI)), 8)
        0.3
    """

    # determines shape of input
    if not hasattr(p, "__len__"):
        x = _Y_to_xI_kernel_inner(a, p, e, Y)
    else:
        p_in = np.atleast_1d(p)
        e_in = np.atleast_1d(e)
        Y_in = np.atleast_1d(Y)

        # cast spin values if necessary
        if not hasattr(a, "__len__"):
            a_in = np.full_like(e_in, a)
        else:
            a_in = np.atleast_1d(a)

        assert len(a_in) == len(e_in)

        x = np.empty_like(e_in)
        _Y_to_xI_kernel(x, a_in, p_in, e_in, Y_in)

    return x


def xI_to_Y(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Convert :math:`x_I = \cos I` to :math:`Y = \cos\iota = L_z/\sqrt{L_z^2 + Q}`.

    Evaluates the Kerr constants of motion with
    :func:`~few.utils.geodesic.get_kerr_geo_constants_of_motion` (same scalar / array rules;
    polar orbits ``x = 0`` are not supported there) and returns
    :math:`L_z/\sqrt{L_z^2 + Q}`. Inverse: :func:`Y_to_xI`. Used by
    :class:`~few.waveform.waveform.GenerateEMRIWaveform` for models that expect ``Y``.

    Args:
        a: Dimensionless spin. Scalar (broadcast on the array path) or an array.
        p: Semi-latus rectum in units of :math:`M`; scalar or array.
        e: Eccentricity.
        x: :math:`x_I = \cos I`.

    Returns:
        float or np.ndarray: :math:`Y`, a NumPy float on the scalar path, else a 1-D array
        of the length of ``p``.

    Example:
        >>> from few.utils.mappings.pn import xI_to_Y
        >>> float(xI_to_Y(0.3, 8.0, 0.4, 1.0))
        1.0
    """

    # get constants of motion
    E, L, Q = get_kerr_geo_constants_of_motion(a, p, e, x)

    Y = L / np.sqrt(L**2 + Q)
    return Y
