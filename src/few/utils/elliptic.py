r"""
Efficient implementations of complete elliptic integrals, obtained by applying
the duplication theorem to compute Carlson's symmetric forms.

Derived from Carlson's algorithms in the SLATEC library. All functions are numba
``@njit`` scalar kernels (callable from Python and from other numba code; used by the
frequency kernels of :mod:`few.utils.geodesic`):

- Carlson forms :func:`RF`, :func:`RD`, :func:`RJ` (duplication iterations) and
  :func:`RC` (closed form);
- complete Legendre integrals in terms of the modulus :math:`k` (not the parameter
  :math:`m = k^2`): :math:`K(k) = R_F(0, 1 - k^2, 1)` (:func:`EllipK`),
  :math:`E(k) = R_F - \tfrac{k^2}{3} R_D` (:func:`EllipE`) and
  :math:`\Pi(n, k) = R_F + \tfrac{n}{3} R_J(0, 1 - k^2, 1, 1 - n)` (:func:`EllipPi`, with
  :math:`\Pi(n, k) = \int_0^{\pi/2} d\theta / [(1 - n\sin^2\theta)\sqrt{1 - k^2\sin^2\theta}]`).

The ``tol`` argument is the convergence threshold of the duplication loop: iteration stops
when the largest relative deviation of the arguments from their (weighted) mean is at most
``tol``.
"""

from math import acos, acosh, log, sqrt

from numba import njit


@njit(fastmath=False)
def RF(x: float, y: float, z: float, tol: float = 5e-4) -> float:
    r"""
    Computes the Carlson symmetric form of the
    elliptic integral of the first kind, :math:`R_F`.

    At most one of :math:`(x, y, z)` may be zero.

    Args:
        x: First argument, :math:`x \ge 0`.
        y: Second argument, :math:`y \ge 0`.
        z: Third argument, :math:`z \ge 0`.
        tol: Convergence threshold of the duplication loop (see the module docstring).
            Default ``5e-4``.

    Returns:
        float: :math:`R_F(x, y, z)`.
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
    elliptic integral of the first kind, :math:`R_C(x, y) = R_F(x, y, y)`.

    Unlike the other Carlson symmetric forms provided, this one
    can be computed analytically in terms of transcendental functions:
    :math:`\arccos(\sqrt{x/y})/\sqrt{y - x}` for :math:`x < y`,
    :math:`\mathrm{arccosh}(\sqrt{x/y})/\sqrt{x - y}` for :math:`x > y`, and
    :math:`1/\sqrt{y}` when :math:`|x - y| \le 10^{-8}`. Used inside :func:`RJ`.

    Args:
        x: First argument, :math:`x \ge 0`.
        y: Second argument, :math:`y > 0`.

    Returns:
        float: :math:`R_C(x, y)`.
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
    elliptic integral of the third kind, :math:`R_J`.

    No more than one of :math:`(x, y, z)` may be equal to zero.

    Args:
        x: First argument, :math:`x \ge 0`.
        y: Second argument, :math:`y \ge 0`.
        z: Third argument, :math:`z \ge 0`.
        p: Fourth argument, :math:`p > 0` (the Cauchy principal value for :math:`p < 0`
            is not implemented).
        tol: Convergence threshold of the duplication loop (see the module docstring).
            Default ``5e-4``.

    Returns:
        float: :math:`R_J(x, y, z, p)`.
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

    It is a requirement that :math:`x+y > 0`. :math:`R_D(x, y, z) = R_J(x, y, z, z)`.

    Args:
        x: First argument, :math:`x \ge 0`.
        y: Second argument, :math:`y \ge 0`.
        z: Third argument, :math:`z > 0`.
        tol: Convergence threshold of the duplication loop (see the module docstring).
            Default ``5e-4``.

    Returns:
        float: :math:`R_D(x, y, z)`.
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

    :math:`K(k) = R_F(0, 1 - k^2, 1)`, with :math:`k` the modulus. Switches to a
    polynomial approximation from Abramowitz & Stegun (in :math:`1 - k^2`, with its
    logarithmic term) for :math:`k > 1 - 10^{-10}`.

    Args:
        k: The elliptic modulus, :math:`k \in [0, 1]`.
        tol: Convergence threshold of :func:`RF`. Default ``5e-4``.

    Returns:
        float: :math:`K(k)`.

    Raises:
        ValueError: If :math:`k \notin [0, 1]`.

    Example:
        >>> import math
        >>> from few.utils.elliptic import EllipK
        >>> abs(EllipK(0.0) - math.pi / 2) < 1e-10
        True
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

    :math:`E(k) = R_F(0, 1 - k^2, 1) - \tfrac{k^2}{3} R_D(0, 1 - k^2, 1)`. Switches to a
    polynomial approximation from Abramowitz & Stegun for :math:`k > 1 - 10^{-10}`.

    Args:
        k: The elliptic modulus, :math:`k \in [0, 1]`.
        tol: Convergence threshold of :func:`RF` / :func:`RD`. Default ``5e-4``.

    Returns:
        float: :math:`E(k)`.

    Raises:
        ValueError: If :math:`k \notin [0, 1]`.
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

    :math:`\Pi(n, k) = R_F(0, 1 - k^2, 1) + \tfrac{n}{3} R_J(0, 1 - k^2, 1, 1 - n)` (see
    the module docstring for the convention).

    Args:
        n: The characteristic, :math:`n < 1`.
        k: The elliptic modulus, :math:`k \in [0, 1]`.
        tol: Convergence threshold of :func:`RF` / :func:`RJ`. Default ``5e-4``.

    Returns:
        float: :math:`\Pi(n, k)`.

    Raises:
        ValueError: If :math:`k \notin [0, 1]`.
    """
    if k < 0 or k > 1:
        raise ValueError("Elliptic integral Pi(n, k) valid only for k in [0, 1].")
    y = 1 - k * k
    return RF(0.0, y, 1.0, tol=tol) + 1 / 3 * n * RJ(0, y, 1, 1 - n, tol=tol)


# TODO: add the incomplete elliptic integrals?
