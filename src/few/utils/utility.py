# Utilities to aid in FastEMRIWaveforms Packages

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import warnings

import numpy as np
from scipy.optimize import brentq
from multispline.spline import BicubicSpline
from .elliptic import EllipK, EllipE, EllipPi

from numba import njit, cuda
from math import sqrt, pow, cos, acos
import few

# check to see if cupy is available for gpus
if few.cutils.fast.is_gpu:
    import cupy as cp
    from cupy.cuda.runtime import setDevice

    gpu = True
else:
    setDevice = None
    gpu = False

from .constants import YRSID_SI, PI

from typing import Union, Optional

# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_overlap(
    time_series_1: np.ndarray, time_series_2: np.ndarray, use_gpu: bool = False
) -> float:
    r"""Calculate the overlap.

    Takes two time series and finds which one is shorter in length. It then
    shortens the longer time series if necessary. Then it performs a
    normalized correlation calulation on the two time series to give the
    overlap. The overlap of :math:`a(t)` and
    :math:`b(t)`, :math:`\gamma_{a,b}`, is given by,

    .. math:: \gamma_{a,b} = <a,b>/(<a,a><b,b>)^{(1/2)},

    where :math:`<a,b>` is the inner product of the two time series.

    args:
        time_series_1: Strain time series 1.
        time_series_2: Strain time series 2.
        use_gpu: If True use cupy. If False, use numpy. Default
            is False.

    """

    # adjust arrays based on GPU usage
    if use_gpu:
        xp = cp

        if isinstance(time_series_1, np.ndarray):
            time_series_1 = xp.asarray(time_series_1)
        if isinstance(time_series_2, np.ndarray):
            time_series_2 = xp.asarray(time_series_2)

    else:
        xp = np

        try:
            if isinstance(time_series_1, cp.ndarray):
                time_series_1 = xp.asarray(time_series_1)

        except NameError:
            pass

        try:
            if isinstance(time_series_2, cp.ndarray):
                time_series_2 = xp.asarray(time_series_2)

        except NameError:
            pass

    # get the lesser of the two lengths
    min_len = int(np.min([len(time_series_1), len(time_series_2)]))

    if len(time_series_1) != len(time_series_2):
        warnings.warn(
            "The two time series are not the same length ({} vs {}). The calculation will run with length {} starting at index 0 for both arrays.".format(
                len(time_series_1), len(time_series_2), min_len
            )
        )

    # chop off excess length on a longer array
    # take fft
    time_series_1_fft = xp.fft.fft(time_series_1[:min_len])
    time_series_2_fft = xp.fft.fft(time_series_2[:min_len])

    # autocorrelation
    ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
        xp.dot(time_series_1_fft.conj(), time_series_1_fft)
        * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
    )

    # if using cupy, it will return a dimensionless array
    if use_gpu:
        return ac.item().real
    return ac.real


def get_mismatch(
    time_series_1: np.ndarray, time_series_2: np.ndarray, use_gpu: bool = False
) -> float:
    """Calculate the mismatch.

    The mismatch is 1 - overlap. Therefore, see documentation for
    :func:`few.utils.utility.overlap` for information on the overlap
    calculation.

    args:
        time_series_1: Strain time series 1.
        time_series_2: Strain time series 2.
        use_gpu: If True use cupy. If False, use numpy. Default
            is False.

    """
    overlap = get_overlap(time_series_1, time_series_2, use_gpu=use_gpu)
    return 1.0 - overlap

@njit(fastmath=False)
def _solveCubic(A2, A1, A0):
    # Coefficients
    a = 1.0  # coefficient of r^3
    b = A2  # coefficient of r^2
    c = A1  # coefficient of r^1
    d = A0  # coefficient of r^0

    # Calculate p and q
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)

    # Calculate discriminant
    discriminant = q * q / 4.0 + p * p * p / 27.0

    if discriminant >= 0:
        # One real root and two complex conjugate roots
        u = (-q / 2.0 + sqrt(discriminant)) ** (1 / 3)
        v = (-q / 2.0 - sqrt(discriminant)) ** (1 / 3)
        root = u + v - b / (3.0 * a)
        # cout << "Real Root: " << root << endl

        # imaginaryPart(-sqrt(3.0) / 2.0 * (u - v), 0.5 * (u + v))
        imaginaryPart = 0.5 * (u + v)
        root2 = -0.5 * (u + v) - b / (3.0 * a) + imaginaryPart
        root3 = -0.5 * (u + v) - b / (3.0 * a) - imaginaryPart
        # cout << "Complex Root 1: " << root2 << endl
        # cout << "Complex Root 2: " << root3 << endl
        ra = -0.5 * (u + v) - b / (3.0 * a)
        rp = -0.5 * (u + v) - b / (3.0 * a)
        r3 = root
    # } else if (discriminant == 0) {
    #     # All roots are real and at least two are equal
    #     u = cbrt(-q/2.)
    #     v = cbrt(-q/2.)
    #     root = u + v - b/(3.*a)
    #     # cout << "Real Root: " << root << endl
    #     # cout << "Real Root (equal to above): " << root << endl
    #     # complex<double> root2 = -0.5 * (u + v) - b / (3 * a)
    #     # cout << "Complex Root: " << root2 << endl
    #     *ra = -0.5 * (u + v) - b / (3. * a)
    #     *rp = -0.5 * (u + v) - b / (3. * a)
    #     *r3 = root
    else:
        # All three roots are real and different
        r = sqrt(-p / 3.0)
        theta = acos(-q / (2.0 * r * r * r))
        root1 = 2.0 * r * cos(theta / 3.0) - b / (3.0 * a)
        root2 = 2.0 * r * cos((theta + 2.0 * PI) / 3.0) - b / (3.0 * a)
        root3 = 2.0 * r * cos((theta - 2.0 * PI) / 3.0) - b / (3.0 * a)
        # ra = -2.*rtQnr*cos((theta + 2.*M_PI)/3.) - A2/3.
        # rp = -2.*rtQnr*cos((theta - 2.*M_PI)/3.) - A2/3.
        ra = root1
        rp = root3
        r3 = root2

    return rp, ra, r3
    # cout << "ra: " << *ra << endl
    # cout << "rp: " << *rp << endl
    # cout << "r3: " << *r3 << endl


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
    if isinstance(E, float):
        # get frequencies
        p, e, x = _ELQ_to_pex_kernel_inner(a, E, Lz, Q)

    else:
        E_in = np.atleast_1d(E)
        Lz_in = np.atleast_1d(Lz)
        Q_in = np.atleast_1d(Q)

        # cast the spin to the same size array as p
        if isinstance(a, float):
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
    return CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR


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
    if a > 0:
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
                _KerrCircularMinoFrequencies_kernel(a, p)
            )
        return UpsilonPhi / Gamma, UpsilonTheta / Gamma, UpsilonR / Gamma
    else:
        return _SchwarzschildGeoCoordinateFrequencies_kernel(p, e)


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
        OmegaPhi[i], OmegaTheta[i], OmegaR[i] = _KerrGeoCoordinateFrequencies_kernel_inner(
            a[i], p[i], e[i], x[i]
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
        xp = cp
    else:
        xp = np

    # check if inputs are scalar or array
    if isinstance(p, float):
        OmegaPhi, OmegaTheta, OmegaR = _KerrGeoCoordinateFrequencies_kernel_inner(
            a, p, e, x
        )
    else:
        p_in = xp.atleast_1d(p)
        e_in = xp.atleast_1d(e)
        x_in = xp.atleast_1d(x)

        # cast the spin to the same size array as p
        if isinstance(a, float):
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
    if isinstance(p, float):
        OmegaPhi, OmegaTheta, OmegaR = _KerrEqSpinFrequenciesCorrections_kernel_inner(
            a, p, e, x
        )
    else:
        p_in = np.atleast_1d(p)
        e_in = np.atleast_1d(e)
        x_in = np.atleast_1d(x)

        # cast the spin to the same size array as p
        if isinstance(a, float):
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
                - x
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
            - x
            * 2.0
            * sqrt(
                Sigma
                * (
                    Sigma * Epsilon * Epsilon
                    + Rho * Epsilon * Kappa
                    - Eta * Kappa * Kappa
                )
                / (x * x)
            )
        )
        / (Rho * Rho + 4.0 * Eta * Sigma)
    )


@njit(fastmath=False)
def _KerrGeoAngularMomentum(a, p, e, x, En):
    r1 = p / (1 - e)

    zm = sqrt(1 - (x * x))

    return (
        -En * _g(r1, a, zm)
        + x
        * sqrt(
            (
                -_d(r1, a, zm) * _h(r1, a, zm)
                + (En * En) * (pow(_g(r1, a, zm), 2) + _f(r1, a, zm) * _h(r1, a, zm))
            )
            / (x * x)
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
    if isinstance(p, float):
        E, L, Q = _KerrGeoConstantsOfMotion_kernel_inner(a, p, e, x)
    else:
        p_in = np.atleast_1d(p)
        e_in = np.atleast_1d(e)
        x_in = np.atleast_1d(x)

        # cast the spin to the same size array as p
        if isinstance(a, float):
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
def _brentq_jit(f, a, b, args, tol):
    # Machine epsilon for double precision
    eps = 2.220446049250313e-16

    fa = f(a, args)
    fb = f(b, args)

    # Check that f(a) and f(b) have different signs
    if fa == 0.0 or fb == 0.0:
        return a if fa == 0.0 else b

    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs.")

    c = a
    fc = fa
    d = b - a
    e = d

    while True:
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            # within tolerance -> return root
            return b

        # Check if bisection is forced
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Linear interpolation
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            if (2.0 * p < 3.0 * xm * q - abs(tol1 * q)) and (p < abs(0.5 * e * q)):
                # Accept interpolation
                d = p / q
            else:
                # Bisection step
                d = xm
                e = d
        else:
            # Bisection step
            d = xm
            e = d

        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0 else -tol1

        fb = f(b, args)
        if fb * fc > 0.0:
            c = a
            fc = fa
            d = b - a
            e = d


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

    elif x == 1.0:  # Eccentric Prograde Equatorial
        x_lo = 1.0 + e
        x_hi = 6.0 + 2.0 * e

        p_sep = _brentq_jit(_separatrix_polynomial_equat, x_lo, x_hi, (a, e), tol)
        return p_sep

    elif x == -1.0:  # Eccentric Retrograde Equatorial
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
def _get_separatrix_kernel_gpu(p_sep: np.ndarray, a: np.ndarray, e: np.ndarray, x: np.ndarray, tol: float=1e-13):
    i = cuda.grid(1)
    if i < len(a):
        p_sep[i] = _get_separatrix_kernel_inner(a[i], e[i], x[i], tol=tol)

def get_separatrix(
    a: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    tol: float = 1e-13,
    use_gpu:bool=False
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
    if use_gpu:
        xp = cp
    else:
        xp = np
    
    # determines shape of input
    if isinstance(e, float):
        separatrix = _get_separatrix_kernel_inner(a, e, x, tol=tol)

    else:
        e_in = xp.atleast_1d(e)

        if isinstance(x, float):
            x_in = xp.full_like(e_in, x)
        else:
            x_in = xp.atleast_1d(x)

        # cast spin values if necessary
        if isinstance(a, float):
            a_in = xp.full_like(e_in, a)
        else:
            a_in = xp.atleast_1d(a)

        if isinstance(x, float):
            x_in = xp.full_like(e_in, x)
        else:
            x_in = xp.atleast_1d(x)

        assert len(a_in) == len(e_in) == len(x_in)

        separatrix = xp.empty_like(e_in)
        if use_gpu:
            threadsperblock = 256
            blockspergrid = (len(a_in) + (threadsperblock - 1)) // threadsperblock
            _get_separatrix_kernel_gpu[blockspergrid, threadsperblock](separatrix, a_in, e_in, x_in, tol)
        else:
            _get_separatrix_kernel_cpu(separatrix, a_in, e_in, x_in, tol=tol)

    return separatrix


# TODO: initialise this properly from the coefficients files, rather than all this stuff getting run every time

CHI2_SCALE = 3
CHI2_AMAX = 0.99998
CHI2_EMIN, CHI2_EMAX = 0.0, 0.9


def chi2_to_a(chi2):
    ymin = (1 - CHI2_AMAX) ** (1 / CHI2_SCALE)
    ymax = (1 + CHI2_AMAX) ** (1 / CHI2_SCALE)
    return 1 - (chi2 * (ymax - ymin) + ymin) ** CHI2_SCALE


@njit(fastmath=False)
def a_to_chi2(a):
    y = (1 - a) ** (1 / CHI2_SCALE)
    ymin = (1 - CHI2_AMAX) ** (1 / CHI2_SCALE)
    ymax = (1 + CHI2_AMAX) ** (1 / CHI2_SCALE)
    return (y - ymin) / (ymax - ymin)


Nx1 = 256
Nx2 = 256

x1 = np.linspace(0, 1, num=Nx1)
x2 = np.linspace(CHI2_EMIN**0.5, CHI2_EMAX**0.5, num=Nx2)
chi2, sqrtecc = np.meshgrid(x1, x2, indexing="ij")

spin = chi2_to_a(chi2.flatten())
e = sqrtecc.flatten() ** 2
to_interp = get_separatrix(np.abs(spin), e, np.sign(spin) * 1.0) / (6.0 + 2.0 * e)

reshapedF = np.asarray(to_interp).reshape((Nx1, Nx2))

PSEP_INTERPOLANT = BicubicSpline(x1, x2, reshapedF, bc="E(3)")


def get_separatrix_interpolant(a, e, x):
    a_sign = a * x
    w = e**0.5
    chi2 = a_to_chi2(a_sign)

    return PSEP_INTERPOLANT(chi2, w) * (6 + 2 * e)


def get_at_t(
    traj_module: object,
    traj_args: list[float],
    bounds: list[float],
    t_out: float,
    index_of_interest: int,
    traj_kwargs: Optional[dict] = None,
    xtol: float = 2e-12,
    rtol: float = 8.881784197001252e-16,
) -> float:
    """Root finding wrapper using Brent's method.

    This function uses scipy's brentq routine to find root.

    arguments:
        traj_module: Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        traj_args: List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        bounds: Minimum and maximum values over which brentq will search for a root.
        t_out: The desired length of time for the waveform.
        index_of_interest: Index where to insert the new values in
            the :code:`traj_args` list.
        traj_kwargs: Keyword arguments for :code:`traj_module`.
            Default is an empty dict.
        xtol: Absolute tolerance of the brentq root-finding - see :code: `np.allclose()` for details.
            Defaults to 2e-12 (scipy default).
        rtol: Relative tolerance of the brentq root-finding - see :code: `np.allclose()` for details.
            Defaults to ~8.8e-16 (scipy default).

    returns:
        Root value.

    """
    if traj_kwargs is None:
        traj_kwargs = {}

    def get_time_root(val, traj, inj_args, traj_kwargs, t_out, ind_interest):
        """
        Function with one p root at T = t_outp, for brentq input.
        """
        inputs = inj_args.copy()
        inputs.insert(ind_interest, val)
        traj_kwargs["T"] = t_out * 2.0
        out = traj(*inputs, **traj_kwargs)
        try:
            return out[0][-1] - t_out * YRSID_SI
        except IndexError:  # trajectory must have started at p_sep
            return -t_out * YRSID_SI

    root = brentq(
        get_time_root,
        bounds[0],
        bounds[1],
        xtol=xtol,
        rtol=rtol,
        args=(traj_module, traj_args, traj_kwargs, t_out, index_of_interest),
    )
    return root


def get_p_at_t(
    traj_module: object,
    t_out: float,
    traj_args: list[float],
    index_of_p: int = 3,
    index_of_a: int = 2,
    index_of_e: int = 4,
    index_of_x: int = 5,
    bounds: list[Optional[float]] = None,
    **kwargs,
) -> float:
    """Find the value of p that will give a specific length inspiral using Brent's method.

    If you want to generate an inspiral that is a specific length, you
    can adjust p accordingly. This function tells you what that value of p
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    This function uses scipy's brentq routine to find the (presumed only)
    value of p that gives a trajectory of duration t_out.

    arguments:
        traj_module: Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out: The desired length of time for the waveform.
        traj_args: List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        index_of_p: Index where to insert the new p values in
            the :code:`traj_args` list. Default is 3.
        index_of_a: Index of a in provided :code:`traj_module` arguments. Default is 2.
        index_of_e: Index of e0 in provided :code:`traj_module` arguments. Default is 4.
        index_of_x: Index of x0 in provided :code:`traj_module` arguments. Default is 5.
        bounds: Minimum and maximum values of p over which brentq will search for a root.
            If not given, will be set to [separatrix + 0.101, 50]. To supply only one of these two limits, set the
            other limit to None.
        **kwargs: Keyword arguments for :func:`get_at_t`.

    returns:
        Value of p that creates the proper length trajectory.

    """

    # fix indexes for p
    if index_of_a > index_of_p:
        index_of_a -= 1
    if index_of_e > index_of_p:
        index_of_e -= 1
    if index_of_x > index_of_p:
        index_of_x -= 1

    if "traj_kwargs" in kwargs and "enforce_schwarz_sep" in kwargs["traj_kwargs"]:
        enforce_schwarz_sep = kwargs["traj_kwargs"]["enforce_schwarz_sep"]

    else:
        enforce_schwarz_sep = False

    # fix bounds
    if bounds is None:
        if not enforce_schwarz_sep:
            p_sep = get_separatrix(
                traj_args[index_of_a], traj_args[index_of_e], traj_args[index_of_x]
            )  # should be fairly close.
        else:
            p_sep = 6 + 2 * traj_args[index_of_e]
        bounds = [p_sep + 0.2, 16.0 + 2 * traj_args[index_of_e]]

    elif bounds[0] is None:
        if not enforce_schwarz_sep:
            p_sep = get_separatrix(
                traj_args[index_of_a], traj_args[index_of_e], traj_args[index_of_x]
            )  # should be fairly close.
        else:
            p_sep = 6 + 2 * traj_args[index_of_e]
        bounds[0] = p_sep + 0.2

    elif bounds[1] is None:
        bounds[1] = 16.0 + 2 * traj_args[index_of_e]

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_p, **kwargs)
    return root


def get_mu_at_t(
    traj_module: object,
    t_out: float,
    traj_args: list[float],
    index_of_mu: int = 1,
    bounds: list[Optional[float]] = None,
    **kwargs,
) -> float:
    """Find the value of mu that will give a specific length inspiral using Brent's method.

    If you want to generate an inspiral that is a specific length, you
    can adjust mu accordingly. This function tells you what that value of mu
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    This function uses scipy's brentq routine to find the (presumed only)
    value of mu that gives a trajectory of duration t_out.

    arguments:
        traj_module: Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out: The desired length of time for the waveform.
        traj_args: List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        index_of_mu: Index where to insert the new p values in
            the :code:`traj_args` list. Default is 1.
        bounds: Minimum and maximum values of p over which brentq will search for a root.
            If not given, will be set to [1e-1, 1e3]. To supply only one of these two limits, set the
            other limit to None.
        **kwargs: Keyword arguments for :func:`get_at_t`.

    returns:
        Value of mu that creates the proper length trajectory.

    """

    # fix bounds
    if bounds is None:
        bounds = [1e-1, 1e3]

    elif bounds[0] is None:
        bounds[0] = 1e-1

    elif bounds[1] is None:
        bounds[1] = 1e3

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_mu, **kwargs)
    return root


# data history is saved here nased on version nunber
# record_by_version = {
#     "1.0.0": 3981654,
#     "1.1.0": 3981654,
#     "1.1.1": 3981654,
#     "1.1.2": 3981654,
#     "1.1.3": 3981654,
#     "1.1.4": 3981654,
#     "1.1.5": 3981654,
#     "1.2.0": 3981654,
#     "1.2.1": 3981654,
#     "1.2.2": 3981654,
#     "1.3.0": 3981654,
#     "1.3.1": 3981654,
#     "1.3.2": 3981654,
#     "1.3.3": 3981654,
#     "1.3.4": 3981654,
#     "1.3.5": 3981654,
#     "1.3.6": 3981654,
#     "1.3.7": 3981654,
#     "1.4.0": 3981654,
#     "1.4.1": 3981654,
#     "1.4.2": 3981654,
#     "1.4.3": 3981654,
#     "1.4.4": 3981654,
#     "1.4.5": 3981654,
#     "1.4.6": 3981654,
#     "1.4.7": 3981654,
#     "1.4.8": 3981654,
#     "1.4.9": 3981654,
#     "1.4.10": 3981654,
#     "1.4.11": 3981654,
#     "1.5.0": 3981654,
#     "1.5.1": 3981654,
# }


def wrapper(*args, **kwargs):
    """Function to convert array and C/C++ class arguments to ptrs

    This function checks the object type. If it is a cupy or numpy array,
    it will determine its pointer by calling the proper attributes. If you design
    a Cython class to be passed through python, it must have a :code:`ptr`
    attribute.

    If you use this function, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    args:
        *args (list): list of the arguments for a function.
        **kwargs (dict): dictionary of keyword arguments to be converted.

    returns:
        Tuple: (targs, tkwargs) where t indicates target (with pointer values
            rather than python objects).

    """
    # declare target containers
    targs = []
    tkwargs = {}

    # args first
    for arg in args:
        if gpu:
            # cupy arrays
            if isinstance(arg, cp.ndarray):
                targs.append(arg.data.mem.ptr)
                continue

        # numpy arrays
        if isinstance(arg, np.ndarray):
            targs.append(arg.__array_interface__["data"][0])
            continue

        try:
            # cython classes
            targs.append(arg.ptr)
            continue
        except AttributeError:
            # regular argument
            targs.append(arg)

    # kwargs next
    for key, arg in kwargs.items():
        if gpu:
            # cupy arrays
            if isinstance(arg, cp.ndarray):
                tkwargs[key] = arg.data.mem.ptr
                continue

        if isinstance(arg, np.ndarray):
            # numpy arrays
            tkwargs[key] = arg.__array_interface__["data"][0]
            continue

        try:
            # cython classes
            tkwargs[key] = arg.ptr
            continue
        except AttributeError:
            # other arguments
            tkwargs[key] = arg

    return (targs, tkwargs)


def pointer_adjust(func):
    """Decorator function for cupy/numpy agnostic cython

    This decorator applies :func:`few.utils.utility.wrapper` to functions
    via the decorator construction.

    If you use this decorator, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper


def cuda_set_device(dev: int):
    """Globally sets CUDA device

    Args:
        dev: CUDA device number.

    """
    if setDevice is not None:
        setDevice(dev)
    else:
        warnings.warn("Setting cuda device, but cupy/cuda not detected.")
