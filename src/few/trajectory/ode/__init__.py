"""
The ode module of the trajectory package implements the stock trajectory models supported in FEW and provides
the tools necessary for users to implement their own trajectory models.
"""

from .flux import SchwarzEccFlux, KerrEccEqFlux, KerrEccEqFluxAPEX
from .pn5 import PN5

_STOCK_TRAJECTORY_OPTIONS = {
    "SchwarzEccFlux": SchwarzEccFlux,
    "KerrEccEqFlux": KerrEccEqFlux,
    "PN5": PN5,
}
