"""
The ode module of the trajectory package implements the stock trajectory models supported in FEW and provides
the tools necessary for users to implement their own trajectory models.
"""

from .base import get_ode_function_options
from .flux import SchwarzEccFlux, KerrEccEqFlux
from .pn5 import PN5

_STOCK_TRAJECTORY_OPTIONS = {
    "SchwarzEccFlux" : SchwarzEccFlux,
    "KerrEccEqFlux" : KerrEccEqFlux,
    "PN5" : PN5,
}