"""
Efficient implementations of complete elliptic integrals, obtained by applying
the duplication theorem to compute Carlson's symmetric forms.

Derived from Carlson's algorithms in the SLATEC library.
"""

from math import acos, acosh, log, sqrt

from numba import njit


@njit(fastmath=False)
def RF(x: float, y: float, z: float, tol: float = 5e-4) -> float:
    r"""
    Computes the Carlson symmetric form of the
    elliptic integral of the first kind, :math:`R_F`.

    Args:
        x: First argument, :math:`x >= 0`.
        y: Second argument, :math:`y >= 0`.
        z: Third argument, :math:`z >= 0`.
        tol: Numerical tolerance parameter (defaults to 5e-4)

    Returns:
        The Carlson symmetric form :math:`R_F`.
    """
    # Taylor expansion coefficients
    c1 = 1.0 / 24.0
    c2 = 3.0 / 44.0
    c3 = 1.0 / 14.0

    maxval = 1000
    while maxval > tol:
        xroot = sqrt(x)
        yroot = sqrt(y)
        zroot = sqrt(z)
        lamda = xroot * (yroot + zroot) + yroot * zroot

        x = (x + lamda) * 0.25
        y = (y + lamda) * 0.25
        z = (z + lamda) * 0.25

        mu = (x + y + z) / 3.0
        xdev = 2.0 - (mu + x) / mu
        ydev = 2.0 - (mu + y) / mu
        zdev = 2.0 - (mu + z) / mu
        maxval = max(abs(xdev), abs(ydev), abs(zdev))

    # Taylor expansion
    e2 = xdev * ydev - zdev * zdev
    e3 = xdev * ydev * zdev
    s = 1.0 + (c1 * e2 - 0.1 - c2 * e3) * e2 + c3 * e3
    drf = s / sqrt(mu)

    return drf


@njit(fastmath=False)
def RC(x: float, y: float) -> float:
    r"""
    Computes the reduced Carlson symmetric form of the
    elliptic integral of the first kind, :math:`R_C`.

    Unlike the other Carlson symmetric forms provided, this one
    can be computed analytically in terms of transcendental functions.

    Args:
        x: First argument, :math:`x >= 0`.
        y: Second argument, :math:`y >= 0`.

    Returns:
        The Carlson symmetric form :math:`R_J`.
    """

    diff = x - y
    # These bounds may need adjustment depending on the precision
    if diff < -1e-8:
        return acos(sqrt(x / y)) / sqrt(-diff)
    elif diff > 1e-8:
        return acosh(sqrt(x / y)) / sqrt(diff)
    else:
        return 1 / sqrt(y)


@njit(fastmath=False)
def RJ(x: float, y: float, z: float, p: float, tol: float = 5e-4) -> float:
    r"""
    Computes the Carlson symmetric form of the
    elliptic integral of the second kind, :math:`R_J`.

    No more than one of :math:`(x, y, z)` may be equal to zero.

    Args:
        x: First argument, :math:`x >= 0`.
        y: Second argument, :math:`y >= 0`.
        z: Third argument, :math:`z >= 0`.
        p: Fourth argument, :math:`p > 0`.
        tol: Numerical tolerance parameter (defaults to 5e-4)

    Returns:
        The Carlson symmetric form :math:`R_J`.
    """
    # Taylor expansion coefficients
    c1 = 3.0 / 14.0
    c2 = 1.0 / 3.0
    c3 = 3.0 / 22.0
    c4 = 3.0 / 26.0

    sigma = 0.0
    power4 = 1.0

    maxval = 1000
    while maxval > tol:
        xroot = sqrt(x)
        yroot = sqrt(y)
        zroot = sqrt(z)
        lamda = xroot * (yroot + zroot) + yroot * zroot
        alfa = (p * (xroot + yroot + zroot) + xroot * yroot * zroot) ** 2
        beta = p * (p + lamda) ** 2

        sigma += power4 * RC(alfa, beta)

        power4 *= 0.25
        x = (x + lamda) * 0.25
        y = (y + lamda) * 0.25
        z = (z + lamda) * 0.25
        p = (p + lamda) * 0.25

        mu = (x + y + z + 2 * p) * 0.2
        xdev = (mu - x) / mu
        ydev = (mu - y) / mu
        zdev = (mu - z) / mu
        pdev = (mu - p) / mu
        maxval = max(abs(xdev), abs(ydev), abs(zdev), abs(pdev))

    # Taylor expansion
    ea = xdev * (ydev + zdev) + ydev * zdev
    eb = xdev * ydev * zdev
    ec = pdev**2
    e2 = ea - 3.0 * ec
    e3 = eb + 2.0 * pdev * (ea - ec)

    s1 = 1.0 + e2 * (-c1 + 0.75 * c3 * e2 - 1.5 * c4 * e3)
    s2 = eb * (0.5 * c2 + pdev * (-c3 - c3 + pdev * c4))
    s3 = pdev * ea * (c2 - pdev * c3) - c2 * pdev * ec
    drj = 3.0 * sigma + power4 * (s1 + s2 + s3) / (mu * sqrt(mu))

    return drj


@njit(fastmath=False)
def RD(x: float, y: float, z: float, tol: float = 5e-4) -> float:
    r"""
    Computes the reduced Carlson symmetric form of the
    elliptic integral of the second kind, :math:`R_D`.

    It is a requirement that :math:`x+y > 0`.

    Args:
        x: First argument, :math:`x >= 0`.
        y: Second argument, :math:`y >= 0`.
        z: Third argument, :math:`z > 0`.
        tol: Numerical tolerance parameter (defaults to 5e-4)

    Returns:
        The Carlson symmetric form :math:`R_D`.
    """

    # Taylor expansion coefficients
    c1 = 3.0 / 14.0
    c2 = 1.0 / 6.0
    c3 = 9.0 / 22.0
    c4 = 3.0 / 26.0

    sigma = 0.0
    power4 = 1.0

    maxval = 1000
    while maxval > tol:
        xroot = sqrt(x)
        yroot = sqrt(y)
        zroot = sqrt(z)
        lamda = xroot * (yroot + zroot) + yroot * zroot

        sigma += power4 / (zroot * (z + lamda))
        power4 *= 0.25

        x = (x + lamda) * 0.25
        y = (y + lamda) * 0.25
        z = (z + lamda) * 0.25

        mu = (x + y + 3.0 * z) * 0.2
        xdev = (mu - x) / mu
        ydev = (mu - y) / mu
        zdev = (mu - z) / mu
        maxval = max(abs(xdev), abs(ydev), abs(zdev))

    # Taylor expansion
    ea = xdev * ydev
    eb = zdev**2
    ec = ea - eb
    ed = ea - 6.0 * eb
    ef = ed + 2 * ec

    s1 = ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * zdev * ef)
    s2 = zdev * (c2 * ef + zdev * (-c3 * ec + zdev * c4 * ea))

    drd = 3.0 * sigma + power4 * (1.0 + s1 + s2) / (mu * sqrt(mu))

    return drd


@njit(fastmath=False)
def EllipK(k: float, tol: float = 5e-4) -> float:
    r"""
    Computes the complete elliptic integral of the first kind, :math:`K(k)`.

    Switches to a polynomial approximation from Abramowitz & Stegun
    for :math:`k > 1 - 10^{-10}`.

    Args:
        k: The elliptic modulus, where `k \in [0, 1]`.
        tol: Numerical tolerance parameter (defaults to 5e-4)

    Returns:
        The complete elliptic integral :math:`K(k)`.
    """
    if k < 0 or k > 1:
        raise ValueError("Elliptic integral K(k) valid only for k in [0, 1].")
    elif k > 1 - 1e-10:
        y = 1 - k * k
        return (
            1.38629436112
            + y * (0.09666344259 + y * 0.03590092383)
            - log(y) * (0.5 + y * (0.12498593597 + y * 0.06880248576))
        )
    else:
        return RF(0, 1 - k * k, 1, tol=tol)


@njit(fastmath=False)
def EllipE(k: float, tol: float = 5e-4) -> float:
    r"""
    Computes the complete elliptic integral of the second kind, :math:`E(k)`.

    Switches to a polynomial approximation from Abramowitz & Stegun
    for :math:`k > 1 - 10^{-10}`.

    Args:
        k: The elliptic modulus, where `k \in [0, 1]`.
        tol: Numerical tolerance parameter (defaults to 5e-4)

    Returns:
        The complete elliptic integral :math:`E(k)`.
    """
    if k < 0 or k > 1:
        raise ValueError("Elliptic integral E(k) valid only for k in [0, 1].")
    elif k > 1 - 1e-10:
        y = 1 - k * k
        return (
            1
            + y * (0.44325141463 + y * (0.06260601220 + y * 0.04757383546))
            - y * log(y) * (0.24998368310 + y * (0.09200180037 + y * 0.04069697526))
        )
    else:
        y = 1 - k * k
        return RF(0, y, 1, tol=tol) - 1 / 3 * k**2 * RD(0, y, 1, tol=tol)


@njit(fastmath=False)
def EllipPi(n: float, k: float, tol: float = 5e-4) -> float:
    r"""
    Computes the complete elliptic integral of the third kind, :math:`\Pi(n, k)`.

    Args:
        n: The characteristic.
        k: The elliptic modulus, where `k \in [0, 1]`.
        tol: Numerical tolerance parameter (defaults to 5e-4)

    Returns:
        The complete elliptic integral :math:`\Pi(n, k)`.
    """
    if k < 0 or k > 1:
        raise ValueError("Elliptic integral Pi(n, k) valid only for k in [0, 1].")
    y = 1 - k * k
    return RF(0.0, y, 1.0, tol=tol) + 1 / 3 * n * RJ(0, y, 1, 1 - n, tol=tol)


# TODO: add the incomplete elliptic integrals?
