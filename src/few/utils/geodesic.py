from math import acos, cos, pow, sqrt
from typing import Union

import numpy as np
from numba import cuda, njit

from ..utils.constants import PI
from ..utils.elliptic import EllipE, EllipK, EllipPi
from ..utils.utility import _brentq_jit, _solveCubic


@njit(fastmath=False)
def _ELQ_to_pex_kernel_inner(a, E, Lz, Q):
    # implements the mapping from orbit integrals
    # (E, Lz, Q) to orbit geometry (p, e, xI).  Also provides the
    # roots r3 and r4 of the Kerr radial geodesic function.

    # Scott A. Hughes (sahughes@mit.edu) code extracted from Gremlin
    # and converted to standalone form 13 Jan 2024.

    if Q < 1.0e-14:  #  equatorial
        E2m1 = E * E - 1.0
        A2 = 2.0 / E2m1
        A1 = a * a - Lz * Lz / E2m1
        A0 = 2.0 * (a * E - Lz) * (a * E - Lz) / E2m1

        rp, ra, r3 = _solveCubic(A2, A1, A0)

        p = 2.0 * ra * rp / (ra + rp)
        e = (ra - rp) / (ra + rp)

        if Lz > 0.0:
            xI = 1.0
        else:
            xI = -1.0

    else:  # non-equatorial
        a2 = a * a
        E2m1 = (E - 1) * (E + 1.0)
        aEmLz = a * E - Lz
        #
        # The quartic: r^4 + A3 r^3 + A2 r^2 + A1 r + A0 == 0.
        # Kerr radial function divided by E^2 - 1.
        #
        A0 = -a2 * Q / E2m1
        A1 = 2.0 * (Q + aEmLz * aEmLz) / E2m1
        A2 = (a2 * E2m1 - Lz * Lz - Q) / E2m1
        A3 = 2.0 / E2m1
        #
        # Definitions following Wolters (https:#quarticequations.com)
        #
        B0 = A0 + A3 * (-0.25 * A1 + A3 * (0.0625 * A2 - 0.01171875 * A3 * A3))
        B1 = A1 + A3 * (-0.5 * A2 + 0.125 * A3 * A3)
        B2 = A2 - 0.375 * A3 * A3
        #
        # Definitions needed for the resolvent cubic: z^3 + C2 z^2 + C1 z + C0 == 0
        #
        C0 = -0.015625 * B1 * B1
        C1 = 0.0625 * B2 * B2 - 0.25 * B0
        C2 = 0.5 * B2
        #
        rtQnr = sqrt(C2 * C2 / 9.0 - C1 / 3.0)
        Rnr = C2 * (C2 * C2 / 27.0 - C1 / 6.0) + C0 / 2.0
        theta = acos(Rnr / (rtQnr * rtQnr * rtQnr))
        #
        # zN = cubic zero N
        #
        rtz1 = sqrt(-2.0 * rtQnr * cos((theta + 2.0 * PI) / 3.0) - C2 / 3.0)
        z2 = -2.0 * rtQnr * cos((theta - 2.0 * PI) / 3.0) - C2 / 3.0
        z3 = -2.0 * rtQnr * cos(theta / 3.0) - C2 / 3.0
        rtz2z3 = sqrt(z2 * z3)
        #
        # Now assemble the roots of the quartic.  Note that M/(2(1 - E^2)) = -0.25*A3.
        #
        if B1 > 0:
            sgnB1 = 1.0
        else:
            sgnB1 = -1.0

        rttermmin = sqrt(z2 + z3 - 2.0 * sgnB1 * rtz2z3)
        # rttermplus = sqrt(z2 + z3 + 2.0 * sgnB1 * rtz2z3)
        ra = -0.25 * A3 + rtz1 + rttermmin
        rp = -0.25 * A3 + rtz1 - rttermmin
        # r3 = -0.25*A3 - rtz1 + rttermplus
        # r4 = -0.25*A3 - rtz1 - rttermplus
        #
        p = 2.0 * ra * rp / (ra + rp)
        e = (ra - rp) / (ra + rp)
        #
        # Note that omE2 = 1 - E^2 = -E2m1 = -(E^2 - 1)
        #
        QpLz2ma2omE2 = Q + Lz * Lz + a2 * E2m1
        denomsqr = QpLz2ma2omE2 + sqrt(
            QpLz2ma2omE2 * QpLz2ma2omE2 - 4.0 * Lz * Lz * a2 * E2m1
        )
        xI = sqrt(2.0) * Lz / sqrt(denomsqr)

    return p, e, xI


@njit(fastmath=False)
def _ELQ_to_pex_kernel(p, e, xI, a, E, Lz, Q):
    for i in range(len(p)):
        p[i], e[i], xI[i] = _ELQ_to_pex_kernel_inner(a[i], E[i], Lz[i], Q[i])


def ELQ_to_pex(
    a: Union[float, np.ndarray],
    E: Union[float, np.ndarray],
    Lz: Union[float, np.ndarray],
    Q: Union[float, np.ndarray],
) -> tuple[Union[float, np.ndarray]]:
    """Convert from Kerr constants of motion to orbital elements.

    arguments:
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        E: Values of energy,
            :math:`E`.
        Lz: Values of angular momentum,
            :math:`L_z`.
        Q: Values of the Carter constant,
            :math:`Q`.

    returns:
        Tuple of (OmegaPhi, OmegaTheta, OmegaR). These are 1D arrays or scalar values depending on inputs.
    """
    # check if inputs are scalar or array
    if not hasattr(E, "__len__"):
        # get frequencies
        p, e, x = _ELQ_to_pex_kernel_inner(a, E, Lz, Q)

    else:
        E_in = np.atleast_1d(E)
        Lz_in = np.atleast_1d(Lz)
        Q_in = np.atleast_1d(Q)

        # cast the spin to the same size array as p
        if not hasattr(a, "__len__"):
            a_in = np.full_like(E_in, a)
        else:
            a_in = np.atleast_1d(a)

        assert len(a_in) == len(E_in)

        p = np.empty_like(E_in)
        e = np.empty_like(E_in)
        x = np.empty_like(E_in)

        # get frequencies
        _ELQ_to_pex_kernel(p, e, x, a_in, E_in, Lz_in, Q_in)

    return (p, e, x)


@njit(fastmath=False)
def _KerrGeoRadialRoots(a, p, e, En, Q):
    r1 = p / (1 - e)
    r2 = p / (1 + e)
    AplusB = (2) / (1 - (En * En)) - (r1 + r2)
    AB = ((a * a) * Q) / ((1 - (En * En)) * r1 * r2)
    r3 = (AplusB + sqrt((AplusB * AplusB) - 4 * AB)) / 2
    r4 = AB / r3

    return r1, r2, r3, r4


@njit(fastmath=False)
def _KerrGeoMinoFrequencies_kernel(a, p, e, x):
    M = 1.0

    En = _KerrGeoEnergy(a, p, e, x)
    L = _KerrGeoAngularMomentum(a, p, e, x, En)
    Q = _KerrGeoCarterConstant(a, p, e, x, En, L)

    r1, r2, r3, r4 = _KerrGeoRadialRoots(a, p, e, En, Q)

    zm = 1 - (x * x)
    a2zp = ((L * L) + (a * a) * (-1 + (En * En)) * (-1 + zm)) / (
        (-1 + (En * En)) * (-1 + zm)
    )

    Epsilon0zp = -(
        ((L * L) + (a * a) * (-1 + (En * En)) * (-1 + zm)) / ((L * L) * (-1 + zm))
    )

    zmOverZp = zm / (
        ((L * L) + (a * a) * (-1 + (En * En)) * (-1 + zm))
        / ((a * a) * (-1 + (En * En)) * (-1 + zm))
    )

    kr = sqrt((r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4))  # (*Eq.(13)*)
    kTheta = sqrt(zmOverZp)  # (*Eq.(13)*)

    EllipK_kr = EllipK(kr)
    EllipK_ktheta = EllipK(kTheta)

    CapitalUpsilonTheta = (PI * L * sqrt(Epsilon0zp)) / (
        2 * EllipK_ktheta
    )  # (*Eq.(15)*)
    CapitalUpsilonR = (PI * sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))) / (
        2 * EllipK_kr
    )  # (*Eq.(15)*)

    rp = M + sqrt(1.0 - (a * a))
    rm = M - sqrt(1.0 - (a * a))

    hr = (r1 - r2) / (r1 - r3)
    hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp))
    hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm))

    # (*Eq. (21)*)
    EllipPi_hr_kr = EllipPi(hr, kr)
    EllipPi_hp_kr = EllipPi(hp, kr)
    EllipPi_hm_kr = EllipPi(hm, kr)

    CapitalUpsilonPhi = (2 * CapitalUpsilonTheta) / (PI * sqrt(Epsilon0zp)) * EllipPi(
        zm, kTheta
    ) + (2 * a * CapitalUpsilonR) / (
        PI * (rp - rm) * sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))
    ) * (
        (2 * M * En * rp - a * L)
        / (r3 - rp)
        * (EllipK_kr - (r2 - r3) / (r2 - rp) * EllipPi_hp_kr)
        - (2 * M * En * rm - a * L)
        / (r3 - rm)
        * (EllipK_kr - (r2 - r3) / (r2 - rm) * EllipPi_hm_kr)
    )
    CapitalGamma = (
        4 * 1.0 * En
        + (2 * a2zp * En * CapitalUpsilonTheta)
        / (PI * L * sqrt(Epsilon0zp))
        * (EllipK_ktheta - EllipE(kTheta))
        + (2 * CapitalUpsilonR)
        / (PI * sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4)))
        * (
            En
            / 2
            * (
                (r3 * (r1 + r2 + r3) - r1 * r2) * EllipK_kr
                + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipPi_hr_kr
                + (r1 - r3) * (r2 - r4) * EllipE(kr)
            )
            + 2 * M * En * (r3 * EllipK_kr + (r2 - r3) * EllipPi_hr_kr)
            + (2 * M)
            / (rp - rm)
            * (
                ((4 * 1.0 * En - a * L) * rp - 2 * M * (a * a) * En)
                / (r3 - rp)
                * (EllipK_kr - (r2 - r3) / (r2 - rp) * EllipPi_hp_kr)
                - ((4 * 1.0 * En - a * L) * rm - 2 * M * (a * a) * En)
                / (r3 - rm)
                * (EllipK_kr - (r2 - r3) / (r2 - rm) * EllipPi_hm_kr)
            )
        )
    )
    return CapitalGamma, CapitalUpsilonPhi, abs(CapitalUpsilonTheta), CapitalUpsilonR


@njit(fastmath=False)
def _KerrCircularMinoFrequencies_kernel(a, p):
    CapitalUpsilonR = sqrt(
        (
            p
            * (
                -2 * (a * a)
                + 6 * a * sqrt(p)
                + (-5 + p) * p
                + (pow(a - sqrt(p), 2) * ((a * a) - 4 * a * sqrt(p) - (-4 + p) * p))
                / abs((a * a) - 4 * a * sqrt(p) - (-4 + p) * p)
            )
        )
        / (2 * a * sqrt(p) + (-3 + p) * p)
    )
    CapitalUpsilonTheta = abs(
        (pow(p, 0.25) * sqrt(3 * (a * a) - 4 * a * sqrt(p) + (p * p)))
        / sqrt(2 * a + (-3 + p) * sqrt(p))
    )
    CapitalUpsilonPhi = pow(p, 1.25) / sqrt(2 * a + (-3 + p) * sqrt(p))
    CapitalGamma = (pow(p, 1.25) * (a + pow(p, 1.5))) / sqrt(2 * a + (-3 + p) * sqrt(p))
    return CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR


@njit(fastmath=False)
def _SchwarzschildGeoCoordinateFrequencies_kernel(p, e):
    qty = sqrt(4 * e / (p - 6.0 + 2 * e))
    EllipE_eval = EllipE(qty)
    EllipK_eval = EllipK(qty)

    EllipPi1 = EllipPi(16 * e / (12.0 + 8 * e - 4 * e * e - 8 * p + p * p), qty)
    EllipPi2 = EllipPi(2 * e * (p - 4) / ((1.0 + e) * (p - 6.0 + 2 * e)), qty)

    OmegaPhi = (2 * pow(p, 1.5)) / (
        sqrt(-4 * (e * e) + pow(-2 + p, 2))
        * (
            8
            + (
                (-2 * EllipPi2 * (6 + 2 * e - p) * (3 + (e * e) - p) * (p * p))
                / ((-1 + e) * ((1.0 + e) * (1.0 + e)))
                - (EllipE_eval * (-4 + p) * (p * p) * (-6 + 2 * e + p)) / (-1 + (e * e))
                + (EllipK_eval * (p * p) * (28 + 4 * (e * e) - 12 * p + (p * p)))
                / (-1 + (e * e))
                + (
                    4
                    * (-4 + p)
                    * p
                    * (2 * (1 + e) * EllipK_eval + EllipPi2 * (-6 - 2 * e + p))
                )
                / (1 + e)
                + 2
                * pow(-4 + p, 2)
                * (
                    EllipK_eval * (-4 + p)
                    + (EllipPi1 * p * (-6 - 2 * e + p)) / (2 + 2 * e - p)
                )
            )
            / (EllipK_eval * pow(-4 + p, 2))
        )
    )

    OmegaR = (p * sqrt((-6 + 2 * e + p) / (-4 * (e * e) + pow(-2 + p, 2))) * PI) / (
        8 * EllipK_eval
        + (
            (-2 * EllipPi2 * (6 + 2 * e - p) * (3 + (e * e) - p) * (p * p))
            / ((-1 + e) * ((1.0 + e) * (1.0 + e)))
            - (EllipE_eval * (-4 + p) * (p * p) * (-6 + 2 * e + p)) / (-1 + (e * e))
            + (EllipK_eval * (p * p) * (28 + 4 * (e * e) - 12 * p + (p * p)))
            / (-1 + (e * e))
            + (
                4
                * (-4 + p)
                * p
                * (2 * (1 + e) * EllipK_eval + EllipPi2 * (-6 - 2 * e + p))
            )
            / (1 + e)
            + 2
            * pow(-4 + p, 2)
            * (
                EllipK_eval * (-4 + p)
                + (EllipPi1 * p * (-6 - 2 * e + p)) / (2 + 2 * e - p)
            )
        )
        / pow(-4 + p, 2)
    )

    return OmegaPhi, OmegaPhi, OmegaR


@njit(fastmath=False)
def _KerrGeoEquatorialMinoFrequencies_kernel(a, p, e, x):
    M = 1.0

    En = _KerrGeoEnergy(a, p, e, x)
    L = _KerrGeoAngularMomentum(a, p, e, x, En)

    r1, r2, r3, r4 = _KerrGeoRadialRoots(a, p, e, En, 0.0)

    # Epsilon0 = (a * a) * (1 - (En * En)) / (L * L)
    # a2zp = ((L * L) + (a * a) * (-1 + (En * En)) * (-1)) / ((-1 + (En * En)) * (-1))
    Epsilon0zp = -(((L * L) + (a * a) * (-1 + (En * En)) * (-1)) / ((L * L) * (-1)))

    zp = (a * a) * (1 - (En * En)) + (L * L)

    kr = sqrt((r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4))  # (*Eq.(13)*)

    EllK = EllipK(kr)
    CapitalUpsilonr = (PI * sqrt((1 - (En * En)) * (r1 - r3) * (r2))) / (
        2 * EllK
    )  # (*Eq.(15)*)
    CapitalUpsilonTheta = x * pow(zp, 0.5)  # (*Eq.(15)*)

    rp = M + sqrt(1.0 - (a * a))
    rm = M - sqrt(1.0 - (a * a))

    hr = (r1 - r2) / (r1 - r3)
    hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp))
    hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm))

    EllipPi_hr_kr = EllipPi(hr, kr)
    EllipPi_hp_kr = EllipPi(hp, kr)
    EllipPi_hm_kr = EllipPi(hm, kr)

    prob1 = (2 * M * En * rp - a * L) * (EllK - (r2 - r3) / (r2 - rp) * EllipPi_hp_kr)

    # This term is zero when r3 - rp == 0.0
    if abs(prob1) != 0.0:
        prob1 = prob1 / (r3 - rp)
    CapitalUpsilonPhi = (CapitalUpsilonTheta) / (sqrt(Epsilon0zp)) + (
        2 * a * CapitalUpsilonr
    ) / (PI * (rp - rm) * sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))) * (
        prob1
        - (2 * M * En * rm - a * L)
        / (r3 - rm)
        * (EllK - (r2 - r3) / (r2 - rm) * EllipPi_hm_kr)
    )

    # This term is zero when r3 - rp == 0.0
    prob2 = ((4 * 1.0 * En - a * L) * rp - 2 * M * (a * a) * En) * (
        EllK - (r2 - r3) / (r2 - rp) * EllipPi_hp_kr
    )
    if abs(prob2) != 0.0:
        prob2 = prob2 / (r3 - rp)

    CapitalGamma = 4 * 1.0 * En + (2 * CapitalUpsilonr) / (
        PI * sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))
    ) * (
        En
        / 2
        * (
            (r3 * (r1 + r2 + r3) - r1 * r2) * EllK
            + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipPi_hr_kr
            + (r1 - r3) * (r2 - r4) * EllipE(kr)
        )
        + 2 * M * En * (r3 * EllK + (r2 - r3) * EllipPi_hr_kr)
        + (2 * M)
        / (rp - rm)
        * (
            prob2
            - ((4 * 1.0 * En - a * L) * rm - 2 * M * (a * a) * En)
            / (r3 - rm)
            * (EllK - (r2 - r3) / (r2 - rm) * EllipPi_hm_kr)
        )
    )

    return CapitalGamma, CapitalUpsilonPhi, abs(CapitalUpsilonTheta), CapitalUpsilonr


@njit(fastmath=False)
def _KerrGeoCoordinateFrequencies_kernel_inner(a, p, e, x):
    if a != 0:
        if e > 0 or abs(x) < 1:
            if abs(x) < 1:
                Gamma, UpsilonPhi, UpsilonTheta, UpsilonR = (
                    _KerrGeoMinoFrequencies_kernel(a, p, e, x)
                )
            else:
                Gamma, UpsilonPhi, UpsilonTheta, UpsilonR = (
                    _KerrGeoEquatorialMinoFrequencies_kernel(a, p, e, x)
                )
        else:
            Gamma, UpsilonPhi, UpsilonTheta, UpsilonR = (
                _KerrCircularMinoFrequencies_kernel(a * x, p)
            )
            UpsilonPhi = x * UpsilonPhi
        return UpsilonPhi / Gamma, UpsilonTheta / Gamma, UpsilonR / Gamma
    else:
        sgnx = 1 if x > 0 else -1
        OmegaPhi, OmegaPhi, OmegaR = _SchwarzschildGeoCoordinateFrequencies_kernel(p, e)
        return sgnx * OmegaPhi, OmegaPhi, OmegaR


@njit(fastmath=False)
def _KerrGeoCoordinateFrequencies_kernel_cpu(OmegaPhi, OmegaTheta, OmegaR, a, p, e, x):
    for i in range(len(OmegaPhi)):
        OmegaPhi[i], OmegaTheta[i], OmegaR[i] = (
            _KerrGeoCoordinateFrequencies_kernel_inner(a[i], p[i], e[i], x[i])
        )


@cuda.jit
def _KerrGeoCoordinateFrequencies_kernel_gpu(OmegaPhi, OmegaTheta, OmegaR, a, p, e, x):
    i = cuda.grid(1)
    if i < OmegaPhi.size:
        OmegaPhi[i], OmegaTheta[i], OmegaR[i] = (
            _KerrGeoCoordinateFrequencies_kernel_inner(a[i], p[i], e[i], x[i])
        )


def get_fundamental_frequencies(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    use_gpu: bool = False,
) -> tuple[Union[float, np.ndarray]]:
    r"""Get dimensionless fundamental frequencies.

    Determines fundamental frequencies in generic Kerr from
    `Schmidt 2002 <https://arxiv.org/abs/gr-qc/0202090>`_.

    arguments:
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p: Values of separation,
            :math:`p`.
        e: Values of eccentricity,
            :math:`e`.
        x: Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.

    returns:
        Tuple of (OmegaPhi, OmegaTheta, OmegaR). These are 1D arrays or scalar values depending on inputs.

    """
    if use_gpu:
        import cupy as xp
    else:
        import numpy as xp

    # check if inputs are scalar or array
    if not hasattr(p, "__len__"):
        OmegaPhi, OmegaTheta, OmegaR = _KerrGeoCoordinateFrequencies_kernel_inner(
            a, p, e, x
        )
    else:
        p_in = xp.atleast_1d(p)
        e_in = xp.atleast_1d(e)
        x_in = xp.atleast_1d(x)

        # cast the spin to the same size array as p
        if not hasattr(a, "__len__"):
            a_in = xp.full_like(p_in, a)
        else:
            a_in = xp.atleast_1d(a)

        assert len(a_in) == len(p_in)

        OmegaPhi = xp.empty_like(p_in)
        OmegaTheta = xp.empty_like(p_in)
        OmegaR = xp.empty_like(p_in)
        if use_gpu:
            threadsperblock = 256
            blockspergrid = (p_in.size + (threadsperblock - 1)) // threadsperblock
            _KerrGeoCoordinateFrequencies_kernel_gpu[blockspergrid, threadsperblock](
                OmegaPhi, OmegaTheta, OmegaR, a_in, p_in, e_in, x_in
            )
        else:
            _KerrGeoCoordinateFrequencies_kernel_cpu(
                OmegaPhi, OmegaTheta, OmegaR, a_in, p_in, e_in, x_in
            )

    return (OmegaPhi, OmegaTheta, OmegaR)


@njit(fastmath=False)
def _P(r, a, En, xi):
    return En * r * r - a * xi


@njit(fastmath=False)
def _deltaP(r, a, En, xi, deltaEn, deltaxi):
    return deltaEn * r * r - xi / r - a * deltaxi


@njit(fastmath=False)
def _deltaRt(r, am1, a0, a1, a2):
    return am1 / r + a0 + r * (a1 + r * a2)


@njit(fastmath=False)
def _KerrEqSpinFrequenciesCorrections_kernel_inner(a, p, e, x):
    M = 1.0
    En = _KerrGeoEnergy(a, p, e, x)
    xi = _KerrGeoAngularMomentum(a, p, e, x, En) - a * En

    # get radial roots
    r1, r2, r3, r4 = _KerrGeoRadialRoots(a, p, e, En, 0.0)

    deltaEn = (
        xi
        * (
            -(a * (En * En) * (r1 * r1) * (r2 * r2))
            - En * (r1 * r1) * (r2 * r2) * xi
            + (a * a) * En * ((r1 * r1) + r1 * r2 + (r2 * r2)) * xi
            + a * ((r1 * r1) + r1 * (-2 + r2) + (-2 + r2) * r2) * (xi * xi)
        )
    ) / (
        (r1 * r1)
        * (r2 * r2)
        * (
            a * (En * En) * r1 * r2 * (r1 + r2)
            + En * ((r1 * r1) * (-2 + r2) + r1 * (-2 + r2) * r2 - 2 * (r2 * r2)) * xi
            + 2 * a * (xi * xi)
        )
    )

    deltaxi = (
        ((r1 * r1) + r1 * r2 + (r2 * r2))
        * xi
        * (En * (r2 * r2) - a * xi)
        * (-(En * (r1 * r1)) + a * xi)
    ) / (
        (r1 * r1)
        * (r2 * r2)
        * (
            a * (En * En) * r1 * r2 * (r1 + r2)
            + En * ((r1 * r1) * (-2 + r2) + r1 * (-2 + r2) * r2 - 2 * (r2 * r2)) * xi
            + 2 * a * (xi * xi)
        )
    )

    am1 = (-2 * a * (xi * xi)) / (r1 * r2)
    a0 = (
        -2
        * En
        * (
            -(a * deltaxi)
            + deltaEn * (r1 * r1)
            + deltaEn * r1 * r2
            + deltaEn * (r2 * r2)
        )
        + 2 * (a * deltaEn + deltaxi) * xi
    )
    a1 = -2 * deltaEn * En * (r1 + r2)
    a2 = -2 * deltaEn * En

    kr = sqrt((r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4))
    hr = (r1 - r2) / (r1 - r3)

    rp = M + sqrt(1.0 - (a * a))
    rm = M - sqrt(1.0 - (a * a))

    hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp))
    hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm))

    Kkr = EllipK(kr)  # (* Elliptic integral of the first kind *)
    Ekr = EllipE(kr)  # (* Elliptic integral of the second kind *)
    Pihrkr = EllipPi(hr, kr)  # (* Elliptic integral of the third kind *)
    Pihmkr = EllipPi(hm, kr)
    Pihpkr = EllipPi(hp, kr)

    Vtr3 = a * xi + (((a * a) + (r3 * r3)) * _P(r3, a, En, xi)) / _CapitalDelta(r3, a)
    deltaVtr3 = a * deltaxi + (r3 * r3 + a * a) / _CapitalDelta(r3, a) * _deltaP(
        r3, a, En, xi, deltaEn, deltaxi
    )

    deltaIt1 = (
        2
        * (
            (deltaEn * Pihrkr * (r2 - r3) * (4 + r1 + r2 + r3)) / 2.0
            + (Ekr * (r1 - r3) * (deltaEn * r1 * r2 * r3 + 2 * xi)) / (2.0 * r1 * r3)
            + (
                (r2 - r3)
                * (
                    (
                        Pihmkr
                        * ((a * a) + (rm * rm))
                        * _deltaP(rm, a, En, xi, deltaEn, deltaxi)
                    )
                    / ((r2 - rm) * (r3 - rm))
                    - (
                        Pihpkr
                        * ((a * a) + (rp * rp))
                        * _deltaP(rp, a, En, xi, deltaEn, deltaxi)
                    )
                    / ((r2 - rp) * (r3 - rp))
                )
            )
            / (-rm + rp)
            + Kkr * (-0.5 * (deltaEn * (r1 - r3) * (r2 - r3)) + deltaVtr3)
        )
    ) / sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))

    cK = Kkr * (
        -0.5 * (a2 * En * (r1 - r3) * (r2 - r3))
        + (
            pow(a, 4) * En * r3 * (-am1 + (r3 * r3) * (a1 + 2 * a2 * r3))
            + 2
            * (a * a)
            * En
            * (r3 * r3)
            * (-(am1 * (-2 + r3)) + a0 * r3 + pow(r3, 3) * (a1 - a2 + 2 * a2 * r3))
            + En
            * pow(r3, 5)
            * (-2 * a0 - am1 + r3 * (a1 * (-4 + r3) + 2 * a2 * (-3 + r3) * r3))
            + 2 * pow(a, 3) * (2 * am1 + a0 * r3 - a2 * pow(r3, 3)) * xi
            + 2
            * a
            * r3
            * (
                am1 * (-6 + 4 * r3)
                + r3 * (2 * a1 * (-1 + r3) * r3 + a2 * pow(r3, 3) + a0 * (-4 + 3 * r3))
            )
            * xi
        )
        / ((r3 * r3) * pow(r3 - rm, 2) * pow(r3 - rp, 2))
    )
    cEPi = (
        En
        * (
            a2 * Ekr * r2 * (r1 - r3)
            + Pihrkr * (r2 - r3) * (2 * a1 + a2 * (4 + r1 + r2 + 3 * r3))
        )
    ) / 2.0
    cPi = (
        (-r2 + r3)
        * (
            (
                Pihmkr
                * ((a * a) + (rm * rm))
                * _P(rm, a, En, xi)
                * _deltaRt(rm, am1, a0, a1, a2)
            )
            / ((r2 - rm) * pow(r3 - rm, 2) * rm)
            - (
                Pihpkr
                * ((a * a) + (rp * rp))
                * _P(rp, a, En, xi)
                * _deltaRt(rp, am1, a0, a1, a2)
            )
            / ((r2 - rp) * pow(r3 - rp, 2) * rp)
        )
    ) / (-rm + rp)

    cE = (
        Ekr
        * (
            (2 * am1 * (-r1 + r3) * xi) / (a * r1)
            + (r2 * Vtr3 * _deltaRt(r3, am1, a0, a1, a2)) / (r2 - r3)
        )
    ) / (r3 * r3)

    deltaIt2 = -(
        (cE + cEPi + cK + cPi) / (pow(1 - (En * En), 1.5) * sqrt((r1 - r3) * (r2 - r4)))
    )
    deltaIt = deltaIt1 + deltaIt2

    It = (
        2
        * (
            (En * (Ekr * r2 * (r1 - r3) + Pihrkr * (r2 - r3) * (4 + r1 + r2 + r3)))
            / 2.0
            + (
                (r2 - r3)
                * (
                    (Pihmkr * ((a * a) + (rm * rm)) * _P(rm, a, En, xi))
                    / ((r2 - rm) * (r3 - rm))
                    - (Pihpkr * ((a * a) + (rp * rp)) * _P(rp, a, En, xi))
                    / ((r2 - rp) * (r3 - rp))
                )
            )
            / (-rm + rp)
            + Kkr * (-0.5 * (En * (r1 - r3) * (r2 - r3)) + Vtr3)
        )
    ) / sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))

    VPhir3 = xi + a / _CapitalDelta(r3, a) * _P(r3, a, En, xi)
    deltaVPhir3 = deltaxi + a / _CapitalDelta(r3, a) * _deltaP(
        r3, a, En, xi, deltaEn, deltaxi
    )

    deltaIPhi1 = (
        2
        * (
            (Ekr * (r1 - r3) * xi) / (a * r1 * r3)
            + (
                a
                * (r2 - r3)
                * (
                    (Pihmkr * _deltaP(rm, a, En, xi, deltaEn, deltaxi))
                    / ((r2 - rm) * (r3 - rm))
                    - (Pihpkr * _deltaP(rp, a, En, xi, deltaEn, deltaxi))
                    / ((r2 - rp) * (r3 - rp))
                )
            )
            / (-rm + rp)
            + Kkr * deltaVPhir3
        )
    ) / sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))

    dK = (
        Kkr
        * (
            -(
                a
                * En
                * (r3 * r3)
                * (
                    2 * a0 * (-1 + r3) * r3
                    + (a1 + 2 * a2) * pow(r3, 3)
                    + am1 * (-4 + 3 * r3)
                )
            )
            - pow(a, 3) * En * r3 * (am1 - (r3 * r3) * (a1 + 2 * a2 * r3))
            - (a * a)
            * (am1 * (-4 + r3) - 2 * a0 * r3 - (a1 + 2 * a2 * (-1 + r3)) * pow(r3, 3))
            * xi
            - pow(-2 + r3, 2) * r3 * (3 * am1 + r3 * (2 * a0 + a1 * r3)) * xi
        )
    ) / ((r3 * r3) * pow(r3 - rm, 2) * pow(r3 - rp, 2))

    dPi = -(
        (
            a
            * (r2 - r3)
            * (
                (Pihmkr * _P(rm, a, En, xi) * _deltaRt(rm, am1, a0, a1, a2))
                / ((r2 - rm) * pow(r3 - rm, 2) * rm)
                - (Pihpkr * _P(rp, a, En, xi) * _deltaRt(rp, am1, a0, a1, a2))
                / ((r2 - rp) * pow(r3 - rp, 2) * rp)
            )
        )
        / (-rm + rp)
    )
    dE = (
        Ekr
        * (
            (-2 * am1 * (r1 - r3) * xi) / ((a * a) * r1)
            + (r2 * VPhir3 * _deltaRt(r3, am1, a0, a1, a2)) / (r2 - r3)
        )
    ) / (r3 * r3)

    deltaIPhi2 = -(
        (dE + dK + dPi) / (pow(1 - (En * En), 1.5) * sqrt((r1 - r3) * (r2 - r4)))
    )
    deltaIPhi = deltaIPhi1 + deltaIPhi2

    IPhi = (
        2
        * (
            (
                a
                * (r2 - r3)
                * (
                    (Pihmkr * _P(rm, a, En, xi)) / ((r2 - rm) * (r3 - rm))
                    - (Pihpkr * _P(rp, a, En, xi)) / ((r2 - rp) * (r3 - rp))
                )
            )
            / (-rm + rp)
            + Kkr * VPhir3
        )
    ) / sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))

    deltaOmegaR = -PI / (It * It) * deltaIt
    deltaOmegaPhi = deltaIPhi / It - IPhi / (It * It) * deltaIt

    return deltaOmegaPhi, deltaOmegaR


@njit(fastmath=False)
def _KerrEqSpinFrequenciesCorrections_kernel(OmegaPhi, OmegaTheta, OmegaR, a, p, e, x):
    for i in range(len(OmegaPhi)):
        OmegaPhi[i], OmegaR[i] = _KerrEqSpinFrequenciesCorrections_kernel_inner(
            a[i], p[i], e[i], x[i]
        )


def get_fundamental_frequencies_spin_corrections(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
) -> tuple[Union[float, np.ndarray]]:
    r"""Get the leading-order correction term to the fundamental frequencies due to the spin of the secondary compact object.

    Currently only supported for equatorial orbits.

    arguments:
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p: Values of separation,
            :math:`p`.
        e: Values of eccentricity,
            :math:`e`.
        x: Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.

    returns:
        Tuple of (OmegaPhi, OmegaTheta, OmegaR). These are 1D arrays or scalar values depending on inputs.

    """

    assert np.all(np.abs(x) == 1.0), "Currently only supported for equatorial orbits."

    # check if inputs are scalar or array
    if not hasattr(p, "__len__"):
        OmegaPhi, OmegaTheta, OmegaR = _KerrEqSpinFrequenciesCorrections_kernel_inner(
            a, p, e, x
        )
    else:
        p_in = np.atleast_1d(p)
        e_in = np.atleast_1d(e)
        x_in = np.atleast_1d(x)

        # cast the spin to the same size array as p
        if not hasattr(a, "__len__"):
            a_in = np.full_like(p_in, a)
        else:
            a_in = np.atleast_1d(a)

        assert len(a_in) == len(p_in)

        OmegaPhi = np.empty_like(p_in)
        OmegaTheta = np.zeros_like(p_in)
        OmegaR = np.empty_like(p_in)
        # get frequencies
        _KerrEqSpinFrequenciesCorrections_kernel(
            OmegaPhi, OmegaTheta, OmegaR, a_in, p_in, e_in, x_in
        )

    return (OmegaPhi, OmegaTheta, OmegaR)


@njit(fastmath=False)
def _CapitalDelta(r, a):
    return (r * r) - 2.0 * r + (a * a)


@njit(fastmath=False)
def _f(r, a, zm):
    return (r * r * r * r) + (a * a) * (r * (r + 2.0) + (zm * zm) * _CapitalDelta(r, a))


@njit(fastmath=False)
def _g(r, a, zm):
    return 2.0 * a * r


@njit(fastmath=False)
def _h(r, a, zm):
    return r * (r - 2.0) + (zm * zm) / (1.0 - (zm * zm)) * _CapitalDelta(r, a)


@njit(fastmath=False)
def _d(r, a, zm):
    return ((r * r) + (a * a) * (zm * zm)) * _CapitalDelta(r, a)


@njit(fastmath=False)
def _fdot(r, a, zm):
    zm2 = zm * zm
    return 4.0 * (r * r * r) + (a * a) * (2.0 * r * (1.0 + zm2) + 2.0 * (1 - zm2))


@njit(fastmath=False)
def _gdot(r, a, zm):
    return 2.0 * a


@njit(fastmath=False)
def _hdot(r, a, zm):
    zm2 = zm * zm
    return 2.0 * (r - 1.0) * (1.0 + zm2 / (1.0 - zm2))


@njit(fastmath=False)
def _ddot(r, a, zm):
    a2 = a * a
    zm2 = zm * zm
    return (
        4.0 * (r * r * r) - 6.0 * (r * r) + 2.0 * a2 * r * (1.0 + zm2) - 2.0 * a2 * zm2
    )


@njit(fastmath=False)
def _KerrGeoEnergy(a, p, e, x):
    zm = sqrt(1.0 - x * x)
    sgnax = 1 if a * x > 0 else -1
    if (
        e < 1e-10
    ):  # switch to spherical formulas A13-A17 (2102.02713) to avoid instability
        r = p

        Kappa = _d(r, a, zm) * _hdot(r, a, zm) - _h(r, a, zm) * _ddot(r, a, zm)
        Epsilon = _d(r, a, zm) * _gdot(r, a, zm) - _g(r, a, zm) * _ddot(r, a, zm)
        Rho = _f(r, a, zm) * _hdot(r, a, zm) - _h(r, a, zm) * _fdot(r, a, zm)
        Eta = _f(r, a, zm) * _gdot(r, a, zm) - _g(r, a, zm) * _fdot(r, a, zm)
        Sigma = _g(r, a, zm) * _hdot(r, a, zm) - _h(r, a, zm) * _gdot(r, a, zm)

    elif abs(x) == 1.0:
        denom = (
            -4.0 * (a * a) * ((-1 + (e * e)) * (-1 + (e * e)))
            + ((3 + (e * e) - p) * (3 + (e * e) - p)) * p
        )
        numer = (-1 + (e * e)) * (
            (a * a) * (1 + 3 * (e * e) + p)
            + p
            * (
                -3
                - (e * e)
                + p
                - sgnax
                * 2
                * sqrt(
                    (
                        (a * a * a * a * a * a) * ((-1 + (e * e)) * (-1 + (e * e)))
                        + (a * a) * (-4 * (e * e) + ((-2 + p) * (-2 + p))) * (p * p)
                        + 2 * (a * a * a * a) * p * (-2 + p + (e * e) * (2 + p))
                    )
                    / (p * p * p)
                )
            )
        )

        if abs(denom) < 1e-14 or abs(numer) < 1e-14:
            ratio = 0.0
        else:
            ratio = numer / denom

        return sqrt(1.0 - ((1.0 - (e * e)) * (1.0 + ratio)) / p)

    else:
        r1 = p / (1.0 - e)
        r2 = p / (1.0 + e)

        Kappa = _d(r1, a, zm) * _h(r2, a, zm) - _h(r1, a, zm) * _d(r2, a, zm)
        Epsilon = _d(r1, a, zm) * _g(r2, a, zm) - _g(r1, a, zm) * _d(r2, a, zm)
        Rho = _f(r1, a, zm) * _h(r2, a, zm) - _h(r1, a, zm) * _f(r2, a, zm)
        Eta = _f(r1, a, zm) * _g(r2, a, zm) - _g(r1, a, zm) * _f(r2, a, zm)
        Sigma = _g(r1, a, zm) * _h(r2, a, zm) - _h(r1, a, zm) * _g(r2, a, zm)

    return sqrt(
        (
            Kappa * Rho
            + 2.0 * Epsilon * Sigma
            - sgnax
            * 2.0
            * sqrt(
                Sigma
                * (
                    Sigma * Epsilon * Epsilon
                    + Rho * Epsilon * Kappa
                    - Eta * Kappa * Kappa
                )
            )
        )
        / (Rho * Rho + 4.0 * Eta * Sigma)
    )


@njit(fastmath=False)
def _KerrGeoAngularMomentum(a, p, e, x, En):
    r1 = p / (1 - e)

    zm = sqrt(1 - (x * x))

    sgnx = 1 if x > 0 else -1

    return (
        -En * _g(r1, a, zm)
        + sgnx
        * sqrt(
            (
                -_d(r1, a, zm) * _h(r1, a, zm)
                + (En * En) * (pow(_g(r1, a, zm), 2) + _f(r1, a, zm) * _h(r1, a, zm))
            )
        )
    ) / _h(r1, a, zm)


@njit(fastmath=False)
def _KerrGeoCarterConstant(a, p, e, x, En, L):
    zm = sqrt(1 - (x * x))

    return (zm * zm) * ((a * a) * (1 - (En * En)) + (L * L) / (1 - (zm * zm)))


@njit(fastmath=False)
def _KerrGeoConstantsOfMotion_kernel_inner(a, p, e, x):
    E_out = _KerrGeoEnergy(a, p, e, x)
    L_out = _KerrGeoAngularMomentum(a, p, e, x, E_out)
    Q_out = _KerrGeoCarterConstant(a, p, e, x, E_out, L_out)
    return E_out, L_out, Q_out


@njit(fastmath=False)
def _KerrGeoConstantsOfMotion_kernel(E_out, L_out, Q_out, a, p, e, x):
    for i in range(len(p)):
        E_out[i], L_out[i], Q_out[i] = _KerrGeoConstantsOfMotion_kernel_inner(
            a[i], p[i], e[i], x[i]
        )


def get_kerr_geo_constants_of_motion(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
) -> tuple[Union[float, np.ndarray]]:
    r"""Get Kerr constants of motion.

    Determines the constants of motion: :math:`(E, L, Q)` associated with a
    geodesic orbit in the generic Kerr spacetime.

    arguments:
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p: Values of separation,
            :math:`p`.
        e: Values of eccentricity,
            :math:`e`.
        x: Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.

    returns:
        tuple: Tuple of (E, L, Q). These are 1D arrays or scalar values depending on inputs.
    """

    # check if inputs are scalar or array
    if not hasattr(p, "__len__"):
        E, L, Q = _KerrGeoConstantsOfMotion_kernel_inner(a, p, e, x)
    else:
        p_in = np.atleast_1d(p)
        e_in = np.atleast_1d(e)
        x_in = np.atleast_1d(x)

        # cast the spin to the same size array as p
        if not hasattr(a, "__len__"):
            a_in = np.full_like(p_in, a)
        else:
            a_in = np.atleast_1d(a)

        assert len(a_in) == len(p_in)

        E = np.empty_like(p_in)
        L = np.empty_like(p_in)
        Q = np.empty_like(p_in)

        # get constants of motion
        _KerrGeoConstantsOfMotion_kernel(E, L, Q, a_in, p_in, e_in, x_in)

    return (E, L, Q)


@njit(fastmath=False)
def _separatrix_polynomial_full(p, args):
    a = args[0]
    e = args[1]
    x = args[2]
    return (
        -4 * (3 + e) * pow(p, 11)
        + pow(p, 12)
        + pow(a, 12) * pow(-1 + e, 4) * pow(1 + e, 8) * pow(-1 + x, 4) * pow(1 + x, 4)
        - 4
        * pow(a, 10)
        * (-3 + e)
        * pow(-1 + e, 3)
        * pow(1 + e, 7)
        * p
        * pow(-1 + pow(x, 2), 4)
        - 4
        * pow(a, 8)
        * (-1 + e)
        * pow(1 + e, 5)
        * pow(p, 3)
        * pow(-1 + x, 3)
        * pow(1 + x, 3)
        * (
            7
            - 7 * pow(x, 2)
            - pow(e, 2) * (-13 + pow(x, 2))
            + pow(e, 3) * (-5 + pow(x, 2))
            + 7 * e * (-1 + pow(x, 2))
        )
        + 8
        * pow(a, 6)
        * (-1 + e)
        * pow(1 + e, 3)
        * pow(p, 5)
        * pow(-1 + pow(x, 2), 2)
        * (
            3
            + e
            + 12 * pow(x, 2)
            + 4 * e * pow(x, 2)
            + pow(e, 3) * (-5 + 2 * pow(x, 2))
            + pow(e, 2) * (1 + 2 * pow(x, 2))
        )
        - 8
        * pow(a, 4)
        * pow(1 + e, 2)
        * pow(p, 7)
        * (-1 + x)
        * (1 + x)
        * (
            -3
            + e
            + 15 * pow(x, 2)
            - 5 * e * pow(x, 2)
            + pow(e, 3) * (-5 + 3 * pow(x, 2))
            + pow(e, 2) * (-1 + 3 * pow(x, 2))
        )
        + 4
        * pow(a, 2)
        * pow(p, 9)
        * (
            -7
            - 7 * e
            + pow(e, 3) * (-5 + 4 * pow(x, 2))
            + pow(e, 2) * (-13 + 12 * pow(x, 2))
        )
        + 2
        * pow(a, 8)
        * pow(-1 + e, 2)
        * pow(1 + e, 6)
        * pow(p, 2)
        * pow(-1 + pow(x, 2), 3)
        * (
            2 * pow(-3 + e, 2) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                pow(e, 2) * (-3 + pow(x, 2))
                - 3 * (1 + pow(x, 2))
                + 2 * e * (1 + pow(x, 2))
            )
        )
        - 2
        * pow(p, 10)
        * (
            -2 * pow(3 + e, 2)
            + pow(a, 2)
            * (
                -3
                + 6 * pow(x, 2)
                + pow(e, 2) * (-3 + 2 * pow(x, 2))
                + e * (-2 + 4 * pow(x, 2))
            )
        )
        + pow(a, 6)
        * pow(1 + e, 4)
        * pow(p, 4)
        * pow(-1 + pow(x, 2), 2)
        * (
            -16 * pow(-1 + e, 2) * (-3 - 2 * e + pow(e, 2)) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                15
                + 6 * pow(x, 2)
                + 9 * pow(x, 4)
                + pow(e, 2) * (26 + 20 * pow(x, 2) - 2 * pow(x, 4))
                + pow(e, 4) * (15 - 10 * pow(x, 2) + pow(x, 4))
                + 4 * pow(e, 3) * (-5 - 2 * pow(x, 2) + pow(x, 4))
                - 4 * e * (5 + 2 * pow(x, 2) + 3 * pow(x, 4))
            )
        )
        - 4
        * pow(a, 4)
        * pow(1 + e, 2)
        * pow(p, 6)
        * (-1 + x)
        * (1 + x)
        * (
            -2 * (11 - 14 * pow(e, 2) + 3 * pow(e, 4)) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                5
                - 5 * pow(x, 2)
                - 9 * pow(x, 4)
                + 4 * pow(e, 3) * pow(x, 2) * (-2 + pow(x, 2))
                + pow(e, 4) * (5 - 5 * pow(x, 2) + pow(x, 4))
                + pow(e, 2) * (6 - 6 * pow(x, 2) + 4 * pow(x, 4))
            )
        )
        + pow(a, 2)
        * pow(p, 8)
        * (
            -16 * pow(1 + e, 2) * (-3 + 2 * e + pow(e, 2)) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                15
                - 36 * pow(x, 2)
                + 30 * pow(x, 4)
                + pow(e, 4) * (15 - 20 * pow(x, 2) + 6 * pow(x, 4))
                + 4 * pow(e, 3) * (5 - 12 * pow(x, 2) + 6 * pow(x, 4))
                + 4 * e * (5 - 12 * pow(x, 2) + 10 * pow(x, 4))
                + pow(e, 2) * (26 - 72 * pow(x, 2) + 44 * pow(x, 4))
            )
        )
    )


@njit(fastmath=False)
def _separatrix_polynomial_polar(p, args):
    a = args[0]
    e = args[1]
    return (
        pow(a, 6) * pow(-1 + e, 2) * pow(1 + e, 4)
        + pow(p, 5) * (-6 - 2 * e + p)
        + pow(a, 2)
        * pow(p, 3)
        * (-4 * (-1 + e) * pow(1 + e, 2) + (3 + e * (2 + 3 * e)) * p)
        - pow(a, 4)
        * pow(1 + e, 2)
        * p
        * (6 + 2 * pow(e, 3) + 2 * e * (-1 + p) - 3 * p - 3 * pow(e, 2) * (2 + p))
    )


@njit(fastmath=False)
def _separatrix_polynomial_equat(p, args):
    a = args[0]
    e = args[1]
    return (
        pow(a, 4) * pow(-3 - 2 * e + pow(e, 2), 2)
        + pow(p, 2) * pow(-6 - 2 * e + p, 2)
        - 2 * pow(a, 2) * (1 + e) * p * (14 + 2 * pow(e, 2) + 3 * p - e * p)
    )


_POLAR_PSEP_X_LO = 1.0 + sqrt(3.0) + sqrt(3.0 + 2.0 * sqrt(3.0))


@njit(fastmath=False)
def _get_separatrix_kernel_inner(a: float, e: float, x: float, tol: float = 1e-13):
    if a == 0:
        # Schwarzschild
        return 6 + 2 * e

    elif np.abs(x) == 1.0 and a * x > 0:  # Eccentric Prograde Equatorial
        x_lo = 1.0 + e
        x_hi = 6.0 + 2.0 * e

        p_sep = _brentq_jit(_separatrix_polynomial_equat, x_lo, x_hi, (a, e), tol)
        return p_sep

    elif np.abs(x) == 1.0 and a * x < 0:  # Eccentric Retrograde Equatorial
        x_lo = 6 + 2.0 * e
        x_hi = 5 + e + 4 * sqrt(1 + e)

        p_sep = _brentq_jit(_separatrix_polynomial_equat, x_lo, x_hi, (a, e), tol)
        return p_sep

    else:
        # solve for polar p_sep
        x_lo = _POLAR_PSEP_X_LO
        x_hi = 8.0

        polar_p_sep = _brentq_jit(_separatrix_polynomial_polar, x_lo, x_hi, (a, e), tol)

        if x == 0.0:
            return polar_p_sep

        elif x > 0.0:
            x_lo = 1.0 + e
            x_hi = 6 + 2.0 * e

            equat_p_sep = _brentq_jit(
                _separatrix_polynomial_equat, x_lo, x_hi, (a, e), tol
            )

            x_lo = equat_p_sep
            x_hi = polar_p_sep

        else:
            x_lo = polar_p_sep
            x_hi = 12.0

        p_sep = _brentq_jit(_separatrix_polynomial_full, x_lo, x_hi, (a, e, x), tol)
        return p_sep


@njit(fastmath=False)
def _get_separatrix_kernel_cpu(
    p_sep: np.ndarray, a: np.ndarray, e: np.ndarray, x: np.ndarray, tol: float = 1e-13
):
    for i in range(len(a)):
        p_sep[i] = _get_separatrix_kernel_inner(a[i], e[i], x[i], tol=tol)


@cuda.jit
def _get_separatrix_kernel_gpu(
    p_sep: np.ndarray, a: np.ndarray, e: np.ndarray, x: np.ndarray, tol: float = 1e-13
):
    i = cuda.grid(1)
    if i < len(a):
        p_sep[i] = _get_separatrix_kernel_inner(a[i], e[i], x[i], tol=tol)


def get_separatrix(
    a: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    tol: float = 1e-13,
    use_gpu: bool = False,
) -> Union[float, np.ndarray]:
    r"""Get separatrix in generic Kerr.

    Determines separatrix in generic Kerr from
    `Stein & Warburton 2020 <https://arxiv.org/abs/1912.07609>`_.

    arguments:
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        e: Values of eccentricity,
            :math:`e`.
        x: Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.
        tol: Tolerance for root-finding. Default is 1e-13.

    returns:
        Separatrix value with shape based on input shapes.

    """
    # determines shape of input
    if not hasattr(e, "__len__"):
        return _get_separatrix_kernel_inner(a, e, x, tol=tol)

    if use_gpu:
        import cupy as xp
    else:
        import numpy as xp

    e_in = xp.atleast_1d(e)

    if not hasattr(x, "__len__"):
        x_in = xp.full_like(e_in, x)
    else:
        x_in = xp.atleast_1d(x)

    # cast spin values if necessary
    if not hasattr(a, "__len__"):
        a_in = xp.full_like(e_in, a)
    else:
        a_in = xp.atleast_1d(a)

    assert len(a_in) == len(e_in) == len(x_in)

    separatrix = xp.empty_like(e_in, dtype=float)
    if use_gpu:
        threadsperblock = 256
        blockspergrid = (len(a_in) + (threadsperblock - 1)) // threadsperblock
        _get_separatrix_kernel_gpu[blockspergrid, threadsperblock](
            separatrix, a_in, e_in, x_in, tol
        )
    else:
        _get_separatrix_kernel_cpu(separatrix, a_in, e_in, x_in, tol=tol)

    return separatrix
