"""Transformations from the orbital parameters to the parameters used by models and splines.

- :mod:`few.utils.mappings.kerrecceq`: apex coordinates ``(a, p, e, xI)`` to the
  ``(u, w, y, z)`` grid coordinates of the Kerr eccentric equatorial flux and amplitude data
  (production ``FastKerrEccentricEquatorialFlux`` path).
- :mod:`few.utils.mappings.jacobian`: flux Jacobian ``(Edot, Lzdot) -> (pdot, edot)``.
- :mod:`few.utils.mappings.pn`: inclination conversions between ``Y = cos(iota)`` and
  ``xI = cos(I)``.
- :mod:`few.utils.mappings.schwarzecc`: ``p -> y`` grid coordinate of the Schwarzschild
  eccentric amplitudes.
- :mod:`few.utils.mappings.common`: binary mass conversions.
"""
