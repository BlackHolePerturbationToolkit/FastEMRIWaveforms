from math import sqrt
from typing import Union

import numpy as np
from numba import njit

from ..geodesic import get_kerr_geo_constants_of_motion


@njit(fastmath=False)
def _d1(q, p, e, Y):
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
    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _g1")
    elif e == 0.0:
        return 2.0 * q * p
    else:
        return 2.0 * q * p / (1.0 + e)


@njit(fastmath=False)
def _h1(q, p, e, Y):
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
    if abs(q) >= 1.0 or abs(e) >= 1.0 or p <= 1.0 or abs(Y) > 1.0:
        raise ValueError("Parameter errors: _g2")
    elif e == 0.0:
        return 2.0 * q
    else:
        return -2.0 * q * p / (-1.0 + e)


@njit(fastmath=False)
def _h2(q, p, e, Y):
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

    E_square = (rhs_numer - 2.0 * sqrt(rhs_sqrt)) / rhs_denom
    return sqrt(E_square)


@njit(fastmath=False)
def _PN_L(q, p, e, Y):
    E = _PN_E(q, p, e, Y)
    h2 = _h2(q, p, e, Y)
    rhs_first_term = -_g2(q, p, e, Y) * E / h2
    rhs_sqrt = rhs_first_term**2 + (_f2(q, p, e, Y) * E * E - _d2(q, p, e, Y)) / h2

    return rhs_first_term + sqrt(rhs_sqrt)


@njit(fastmath=False)
def _PN_C(q, p, e, Y):
    L = _PN_L(q, p, e, Y)
    if Y == 1.0:
        return 0.0
    else:
        return L * L * ((1.0 / (Y * Y)) - 1.0)


@njit(fastmath=False)
def _Y_to_xI_kernel_inner(a, p, e, Y):
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
    r"""Convert from :math:`Y=\cos{\iota}` to :math:`x_I=\cos{I}`.

    Converts between the two different inclination parameters. :math:`\cos{I}\equiv x_I`,
    where :math:`I` describes the orbit's inclination from the equatorial plane.
    :math:`\cos{\iota}\equiv Y`, where :math:`\cos{\iota}=L/\sqrt{L^2 + Q}`.

    arguments:
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p: Values of separation,
            :math:`p`.
        e: Values of eccentricity,
            :math:`e`.
        Y: Values of PN cosine of inclination :math:`Y`.

    returns:
    :math:`x=\cos{I}` values with shape based on input shapes.

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
    r"""Convert from :math:`x_I=\cos{I}` to :math:`Y=\cos{\iota}`.

    Converts between the two different inclination parameters. :math:`\cos{I}\equiv x_I`,
    where :math:`I` describes the orbit's inclination from the equatorial plane.
    :math:`\cos{\iota}\equiv Y`, where :math:`\cos{\iota}=L/\sqrt{L^2 + Q}`.

    arguments:
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p: Values of separation,
            :math:`p`.
        e: Values of eccentricity,
            :math:`e`.
        x: Values of cosine of the
            inclination, :math:`x=\cos{I}`.

    returns:
        :math:`Y` values with shape based on input shapes.

    """

    # get constants of motion
    E, L, Q = get_kerr_geo_constants_of_motion(a, p, e, x)

    Y = L / np.sqrt(L**2 + Q)
    return Y
