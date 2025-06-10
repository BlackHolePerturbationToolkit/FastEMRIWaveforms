from math import sqrt

from numba import njit

from ..geodesic import (
    _KerrGeoAngularMomentum,
    _KerrGeoEnergy,
)


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
