r"""Jacobian of the flux conversion :math:`(\dot E, \dot L_z) \to (\dot p, \dot e)`.

:func:`ELdot_to_PEdot_Jacobian` converts energy and angular-momentum fluxes of equatorial
Kerr orbits into rates of change of the semi-latus rectum and eccentricity (via the radial
turning points :math:`r_a = p/(1 - e)`, :math:`r_p = p/(1 + e)`). On the production path
:class:`~few.trajectory.ode.flux.KerrEccEqFlux` applies it once per flux-grid node when it
builds its ``(pdot, edot)`` splines (with ``a >= 0`` and ``x = +-1``);
:class:`~few.trajectory.ode.base.ODEBase` also uses it for models that output
``(Edot, Ldot)`` while integrating ``(p, e, x)``, and :class:`~few.trajectory.ode.pn5.PN5`
for equatorial orbits.
"""

from math import sqrt

from numba import njit

from ..geodesic import (
    _KerrGeoAngularMomentum,
    _KerrGeoEnergy,
)


@njit
def _schwarz_jac_kernel(p, e, Edot, Ldot):
    """Closed-form Schwarzschild (``a = 0``) Jacobian: returns ``(pdot, edot)``.

    ``edot = 0`` for ``e = 0``; the orbit is taken as prograde (no ``xI`` argument).
    """
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
    """Convert the fluxes ``(Edot, Lzdot)`` of an equatorial orbit to ``(pdot, eccdot)``.

    Differentiates the turning-point conditions at :math:`r_a = p/(1 - e)` and
    :math:`r_p = p/(1 + e)` (a single radius for ``ecc == 0``) with the Kerr energy and
    angular momentum of the orbit; ``a == 0`` uses the closed Schwarzschild form
    (:func:`_schwarz_jac_kernel`). A numba kernel: scalar float inputs only. The output
    rates are per unit of the time variable of the input fluxes.

    Args:
        a: Dimensionless spin; the flux-grid and PN5 callers pass ``a >= 0`` with the
            orbital direction in ``xI``.
        p: Semi-latus rectum in units of :math:`M`.
        ecc: Eccentricity.
        xI: Cosine of the inclination, ``+1`` or ``-1``.
        Edot: Rate of change of the energy per unit mass.
        Lzdot: Rate of change of the axial angular momentum per unit mass.

    Returns:
        tuple[float, float]: ``(pdot, eccdot)``; ``eccdot`` is 0 for ``ecc == 0`` and (for
        ``a != 0``) set to 0 when ``|eccdot| < 3e-14``.
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
