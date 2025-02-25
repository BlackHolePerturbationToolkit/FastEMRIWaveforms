import numpy as np
from numba import njit
from math import sqrt, log
from .utility import (
    get_separatrix,
    _KerrGeoEnergy,
    _KerrGeoAngularMomentum,
)
from typing import Optional, Union


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


@njit
def _schwarz_jac_kernel(p, e, Edot, Ldot):
    pdot = (
        -2
        * (
            Edot
            * sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p))
            * (3 + pow(e, 2) - p)
            * pow(p, 1.5)
            + Ldot * pow(-4 + p, 2) * sqrt(-3 - pow(e, 2) + p)
        )
    ) / (4 * pow(e, 2) - pow(-6 + p, 2))
    if e > 0:
        edot = -(
            (
                Edot
                * sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p))
                * pow(p, 1.5)
                * (18 + 2 * pow(e, 4) - 3 * pow(e, 2) * (-4 + p) - 9 * p + pow(p, 2))
                + (-1 + pow(e, 2))
                * Ldot
                * sqrt(-3 - pow(e, 2) + p)
                * (12 + 4 * pow(e, 2) - 8 * p + pow(p, 2))
            )
            / (e * (4 * pow(e, 2) - pow(-6 + p, 2)) * p)
        )
    else:
        edot = 0.0
    return pdot, edot


@njit
def ELdot_to_PEdot_Jacobian(
    a: float, p: float, ecc: float, xI: float, Edot: float, Lzdot: float
):
    """
    Jacobian transformation of fluxes from (E, Lz) to (p, ecc) coordinates.
    As this function is a numba kernel, float inputs are required.
    Arguments:
        a (float): Spin parameter of the massive black hole.
        p (float): Semi-latus rectum of inspiral.
        ecc (float): Eccentricity of inspiral.
        xI (float): Cosine of inclination of inspiral. Must have a magnitude of 1.
        Edot (float): Time derivative of orbital energy.
        Lzdot (float): Time derivative of orbital angular momentum.
    """

    if a == 0:
        return _schwarz_jac_kernel(p, ecc, Edot, Lzdot)

    E = _KerrGeoEnergy(a, p, ecc, xI)
    Lz = _KerrGeoAngularMomentum(a, p, ecc, xI, E)

    Lz2pQ = Lz * Lz
    LzmaE = Lz - a * E
    omE2 = (1.0 - E) * (1.0 + E)

    if ecc == 0.0:
        u2 = 1.0 / p
        denom = u2 * (-6 * omE2 + u2 * (6.0 + u2 * (-a * a * omE2 - Lz2pQ)))
        numerEdot = -4.0 * E + u2 * u2 * (-2.0 * a * a * E + u2 * (2.0 * a * LzmaE))
        numerLzdot = u2 * u2 * (2.0 * Lz + u2 * (-2 * LzmaE))

        pdot = (numerEdot * Edot + numerLzdot * Lzdot) / denom
        eccdot = 0.0

    else:
        ra = p / (1.0 - ecc)
        rp = p / (1.0 + ecc)
        ua2 = 1.0 / ra
        up2 = 1.0 / rp

        denoma = ua2 * (
            -4.0 * omE2
            + ua2
            * (
                6
                + ua2
                * (
                    -2 * Lz2pQ
                    - 2 * a * a * omE2
                    + ua2 * (2.0 * Lz2pQ - 4.0 * a * E * Lz + 2.0 * a * a * E * E)
                )
            )
        )
        denomp = up2 * (
            -4.0 * omE2
            + up2
            * (
                6
                + up2
                * (
                    -2 * Lz2pQ
                    - 2 * a * a * omE2
                    + up2 * (2.0 * Lz2pQ - 4.0 * a * E * Lz + 2.0 * a * a * E * E)
                )
            )
        )

        numeraEdot = -2.0 * (E + ua2 * ua2 * (a * a * E + ua2 * (-2.0 * a * (LzmaE))))
        numeraLzdot = 2.0 * ua2 * ua2 * (Lz - 2.0 * ua2 * (LzmaE))

        numerpEdot = -2.0 * (E + up2 * up2 * (a * a * E + up2 * (-2.0 * a * (LzmaE))))
        numerpLzdot = 2.0 * up2 * up2 * (Lz - 2.0 * up2 * (LzmaE))

        radot = (numeraEdot * Edot + numeraLzdot * Lzdot) / denoma
        rpdot = (numerpEdot * Edot + numerpLzdot * Lzdot) / denomp

        pdot = (
            0.5 * (1 + ecc * (-2.0 + ecc)) * radot
            + 0.5 * (1.0 + ecc * (2.0 + ecc)) * rpdot
        )
        eccdot = (
            (1.0 + ecc)
            * ((1.0 + ecc * (-2.0 + ecc)) * radot + (ecc * ecc - 1.0) * rpdot)
            / (2.0 * p)
        )
        if abs(eccdot) < 3.0e-14:
            eccdot = 0.0

    return pdot, eccdot


def schwarzecc_p_to_y(p, e, use_gpu=False):
    """Convert from separation :math:`p` to :math:`y` coordinate

    Conversion from the semilatus rectum or separation :math:`p` to :math:`y`.

    arguments:
        p (double scalar or 1D xp.ndarray): Values of separation,
            :math:`p`, to convert.
        e (double scalar or 1D xp.ndarray): Associated eccentricity values
            of :math:`p` necessary for conversion.
        use_gpu (bool, optional): If True, use Cupy/GPUs. Default is False.

    """
    if use_gpu:
        import cupy as cp

        e_cp = cp.asarray(e)
        p_cp = cp.asarray(p)
        return cp.log(-(21 / 10) - 2 * e_cp + p_cp)

    else:
        return np.log(-(21 / 10) - 2 * e + p)


def kerrecceq_legacy_p_to_u(a, p, e, xI, use_gpu=False):
    """Convert from separation :math:`p` to :math:`y` coordinate

    Conversion from the semilatus rectum or separation :math:`p` to :math:`y`.

    arguments:
        p (double scalar or 1D xp.ndarray): Values of separation,
            :math:`p`, to convert.
        e (double scalar or 1D xp.ndarray): Associated eccentricity values
            of :math:`p` necessary for conversion.
        use_gpu (bool, optional): If True, use Cupy/GPUs. Default is False.

    """
    if use_gpu:
        import cupy as xp
    else:
        import numpy as xp

    scalar = False
    if isinstance(a, float):
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


def kerrecceq_forward_map(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    xI: Union[float, np.ndarray],
    pLSO: Optional[Union[float, np.ndarray]] = None,
    return_mask: bool = False,
    kind: str = "flux",
):
    """
    Map from apex coordinates to interpolation coordinates for the KerrEccEq model.
    This mapping returns the interpolation coordinates (u, w, y, z) corresponding to the apex coordinates (p, e, xI, a).

    The interpolation coordinates are defined on the range [0, 1] and refer to either a "close" region A or a "far" region B.
    To also output an array indicating whether each point is in region A or B, set return_mask=True.

    For either flux or amplitude interpolation with the KerrEccEq model, the same formulae are applied with different tunable
    parameters. The default is flux interpolation, but amplitude interpolation can be selected by setting kind="amplitude".

    Arguments:
        a (float or np.ndarray): Spin parameter of the massive black hole.
        p (float or np.ndarray): Semi-latus rectum of inspiral.
        e (float or np.ndarray): Eccentricity of inspiral.
        xI (float or np.ndarray): Cosine of inclination of inspiral. Only a value of 1 is supported.
        pLSO (float or np.ndarray, optional): Separatrix value for the given parameters. If not provided, it will be computed.
        return_mask (bool, optional): If True, return a mask indicating whether the point is in region A or B.
            A value of True corresponds to region A. For kind="flux", a mask is always returned for efficiency.
            Default is False.
        kind (str, optional): Type of mapping to perform. Default is "flux".
    """
    xp = np  # TODO: gpu

    if np.any(xI != 1):
        raise ValueError("Only xI = 1 is supported.")

    if kind == "flux":
        alpha = ALPHA_FLUX
        beta = BETA_FLUX
    elif kind == "amplitude":
        alpha = ALPHA_AMP
        beta = BETA_AMP
    else:
        raise ValueError

    # if scalar directly evaluate the kernel for speed
    if isinstance(p, float):
        if pLSO is None:
            a_sep = abs(a)
            xI_sep = -1 if a < 0 else 1
            pLSO = get_separatrix(a_sep, e, xI_sep)

        if p <= pLSO + DELTAPMAX:
            return *_uwyz_of_apex_kernel(
                a, p, e, xI, pLSO, alpha=alpha, beta=beta
            ), True
        else:
            return *_UWYZ_of_apex_kernel(a, p, e, xI, pLSO, True), False

    # else, we have multiple points
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
    near = p < pLSO + DELTAPMAX
    if xp.any(near):
        out = _uwyz_of_apex_kernel(
            a[near], p[near], e[near], xI[near], pLSO[near], alpha=alpha, beta=beta
        )
        u[near] = out[0]
        w[near] = out[1]
        y[near] = out[2]
        z[near] = out[3]

    far = ~near
    if xp.any(far):
        out = _UWYZ_of_apex_kernel(a[far], p[far], e[far], xI[far], pLSO[far], False)
        u[far] = out[0]
        w[far] = out[1]
        y[far] = out[2]
        z[far] = out[3]

    if return_mask:
        return u, w, y, z, near
    else:
        return u, w, y, z


@njit
def u_of_p(p, pLSO, dpmin, dpmax, alpha):
    return np.abs(
        (np.log(p - pLSO + dpmax - 2 * dpmin) - log(dpmax - dpmin)) / log(2)
    ) ** (alpha)


@njit
def y_of_x(x, xmin=XMIN):
    return (x - xmin) / (1 - xmin)


@njit
def chi_of_a(a):
    return (1 - a) ** (1 / 3)


@njit
def chi2_of_a(a):
    return (1 - a) ** (2 / 3)


@njit
def z_of_a(a, amin=AMIN, amax=AMAX):
    chimax = chi_of_a(amin)
    chimin = chi_of_a(amax)
    return (chi_of_a(a) - chimin) / (chimax - chimin)


@njit
def z2_of_a(a, amin=AMIN, amax=AMAX):
    chimax = chi2_of_a(amin)
    chimin = chi2_of_a(amax)
    return (chi2_of_a(a) - chimin) / (chimax - chimin)


@njit
def Secc_of_uz(
    u,
    z,
    beta,
    esep=ESEP,
    emax=EMAX,
):
    return esep + (emax - esep) * np.sqrt(np.abs(z + u**beta * (1 - z)))


@njit
def w_of_euz(e, u, z, beta, esep=ESEP, emax=EMAX):
    return e / Secc_of_uz(u, z, beta, esep, emax)


@njit
def p_of_u(u, pLSO, dpmin, dpmax, alpha):
    return (pLSO + dpmin) + (dpmax - dpmin) * (np.exp(u ** (1 / alpha) * log(2)) - 1)


@njit
def x_of_y(y, xmin=XMIN):
    return y * (1 - xmin) + xmin


@njit
def a_of_chi2(chi):
    return 1 - chi ** (1.5)


@njit
def a_of_z2(z, amin=AMIN, amax=AMAX):
    chimax = chi2_of_a(amin)
    chimin = chi2_of_a(amax)
    return a_of_chi2(chimin + z * (chimax - chimin))


@njit
def a_of_chi(chi):
    return 1 - chi**3


@njit
def a_of_z(z, amin=AMIN, amax=AMAX):
    chimax = chi_of_a(amin)
    chimin = chi_of_a(amax)
    return a_of_chi(chimin + z * (chimax - chimin))


@njit
def e_of_uwz(u, w, z, beta, esep=ESEP, emax=EMAX):
    return Secc_of_uz(u, z, beta, esep, emax) * w


@njit
def _uwyz_of_apex_kernel(
    a,
    p,
    e,
    x,
    pLSO,
    amin=AMIN,
    amax=AMAX,
    dpmin=DELTAPMIN,
    dpmax=DELTAPMAX,
    xmin=XMIN,
    esep=ESEP,
    emax=EMAX,
    alpha=ALPHA_FLUX,
    beta=BETA_FLUX,
):
    u = u_of_p(p, pLSO, dpmin, dpmax, alpha)
    y = y_of_x(x, xmin)
    z = z_of_a(a, amin, amax)
    w = w_of_euz(e, u, z, beta, esep, emax)
    return u, w, y, z


def apex_of_uwyz(
    u,
    w,
    y,
    z,
    amin=AMIN,
    amax=AMAX,
    dpmin=DELTAPMIN,
    dpmax=DELTAPMAX,
    xmin=XMIN,
    esep=ESEP,
    emax=EMAX,
    alpha=ALPHA_FLUX,
    beta=BETA_FLUX,
):
    a = a_of_z(z, amin, amax)
    x = x_of_y(y, xmin)
    e = e_of_uwz(u, w, z, beta, esep, emax)
    a = np.asarray(a)
    a_in = np.abs(a)
    x_in = np.sign(a)
    x_in[x_in == 0] = 1

    pLSO = get_separatrix(a_in, e, x_in)
    p = p_of_u(u, pLSO, dpmin, dpmax, alpha)
    return a, p, e, x


# # Region B


@njit
def U_of_p_flux(p, pLSO, delta_pmin=DELTAPMIN_REGIONB, pmax=PMAX_REGIONB):
    pmin = pLSO + delta_pmin
    return ((pmin - pLSO) ** (-0.5) - (p - pLSO) ** (-0.5)) / (
        (pmin - pLSO) ** (-0.5) - (pmax - pLSO) ** (-0.5)
    )


@njit
def p_of_U_flux(U, pLSO, delta_pmin=DELTAPMIN_REGIONB, pmax=PMAX_REGIONB):
    pmin = pLSO + delta_pmin
    return (
        (pmin - pLSO) ** (-0.5)
        - U * ((pmin - pLSO) ** (-0.5) - (pmax - pLSO) ** (-0.5))
    ) ** (-2) + pLSO


@njit
def U_of_p_amplitude(p, pmin, pmax=PMAX_REGIONB):
    pc = pmin
    pmax = pmax + pc
    return (pc ** (-0.5) - p ** (-0.5)) / (pc ** (-0.5) - pmax ** (-0.5))


@njit
def p_of_U_amplitude(U, pmin, pmax=PMAX_REGIONB):
    pc = pmin
    pmax += pc
    return (pc ** (-0.5) - U * (pc ** (-0.5) - pmax ** (-0.5))) ** (-2)


@njit
def W_of_e(e, emax=EMAX_REGIONB):
    return e / emax


@njit
def e_of_W(y, emax=EMAX_REGIONB):
    return y * emax


@njit
def Y_of_x(x, xmin):
    return y_of_x(x, xmin)


@njit
def x_of_Y(y, xmin):
    return x_of_y(y, xmin)


@njit
def Z_of_a(a, amin=AMIN_REGIONB, amax=AMAX_REGIONB):
    chimax = chi_of_a(amin)
    chimin = chi_of_a(amax)
    return (chi_of_a(a) - chimin) / (chimax - chimin)


@njit
def a_of_Z(z, amin=AMIN_REGIONB, amax=AMAX_REGIONB):
    chimax = chi_of_a(amin)
    chimin = chi_of_a(amax)
    return a_of_chi(chimin + z * (chimax - chimin))


@njit
def _UWYZ_of_apex_kernel(
    a,
    p,
    e,
    x,
    pLSO,
    is_flux,
    amin=AMIN,
    amax=AMAX,
    dpc=DPC_REGIONB,
    pmax=PMAX_REGIONB,
    xmin=XMIN,
    emax=EMAX,
    delta_pmin=DELTAPMIN_REGIONB,
):
    if is_flux:
        u = U_of_p_flux(p, pLSO, delta_pmin, pmax)
    else:
        pmin = pLSO + dpc
        u = U_of_p_amplitude(p, pmin, pmax)
    y = Y_of_x(x, xmin)
    z = Z_of_a(a, amin, amax)
    w = W_of_e(e, emax)
    return u, w, y, z


def apex_of_UWYZ(
    u,
    w,
    y,
    z,
    is_flux,
    amin=AMIN,
    amax=AMAX,
    dpc=DPC_REGIONB,
    pmax=PMAX_REGIONB,
    xmin=XMIN,
    emax=EMAX,
    delta_pmin=DELTAPMIN_REGIONB,
):
    a = a_of_Z(z, amin, amax)
    x = x_of_Y(y, xmin)
    e = e_of_W(w, emax)

    a = np.asarray(a)
    a_in = np.abs(a)
    x_in = np.sign(a)
    x_in[x_in == 0] = 1

    pLSO = get_separatrix(a_in, e, x_in)
    if is_flux:
        p = p_of_U_flux(u, pLSO, delta_pmin, pmax)
    else:
        pmin = pLSO + dpc
        p = p_of_U_amplitude(u, pmin, pmax)
    return a, p, e, x
