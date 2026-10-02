"""
The ode module of the trajectory package implements the stock trajectory models supported in FEW and provides
the tools necessary for users to implement their own trajectory models.
"""

from .flux import KerrEccEqFlux, SchwarzEccFlux
from .pn5 import PN5
from .circ1pat1r import TrajectoryCirc1PAT1R

_STOCK_TRAJECTORY_OPTIONS = {
    "SchwarzEccFlux": SchwarzEccFlux,
    "KerrEccEqFlux": KerrEccEqFlux,
    "PN5": PN5,
    "TrajectoryCirc1PAT1R": TrajectoryCirc1PAT1R,
}
