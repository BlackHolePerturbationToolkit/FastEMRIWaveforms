"""Right-hand sides of the trajectory ODE: the stock models and the base for custom ones.

Stock models (re-exported here):

- :class:`~few.trajectory.ode.flux.KerrEccEqFlux`: Kerr eccentric equatorial, tricubic
  splines of the flux data ``KerrEccEqFluxData.h5`` (the ODE of
  :class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux`).
- :class:`~few.trajectory.ode.flux.SchwarzEccFlux`: Schwarzschild eccentric, bicubic
  splines of ``FluxNewMinusPNScaled_fixed_y_order.dat``.
- :class:`~few.trajectory.ode.pn5.PN5`: 5PN, :math:`e^{10}` post-Newtonian fluxes for
  generic Kerr orbits (the ODE of the AAK waveform).

A custom model subclasses :class:`~few.trajectory.ode.base.ODEBase` and implements
``evaluate_rhs``; pass the class (or, for a stock model, its name) as ``func`` to
:class:`~few.trajectory.inspiral.EMRIInspiral`.
"""

from .flux import KerrEccEqFlux, SchwarzEccFlux
from .pn5 import PN5

# name -> class for a string ``func`` (EMRIInspiral(func="KerrEccEqFlux")); looked up by
# few.trajectory.integrate.digest_func
_STOCK_TRAJECTORY_OPTIONS = {
    "SchwarzEccFlux": SchwarzEccFlux,
    "KerrEccEqFlux": KerrEccEqFlux,
    "PN5": PN5,
}
