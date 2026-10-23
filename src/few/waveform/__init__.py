"""Waveform generation: the stock FEW models and the generic interface.

Re-exports, from :mod:`few.waveform.waveform`, the generic interface
:class:`~few.waveform.waveform.GenerateEMRIWaveform` and the stock models
:class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux`,
:class:`~few.waveform.waveform.FastSchwarzschildEccentricFlux`,
:class:`~few.waveform.waveform.FastSchwarzschildEccentricFluxBicubic`,
:class:`~few.waveform.waveform.SlowSchwarzschildEccentricFlux` and
:class:`~few.waveform.waveform.Pn5AAKWaveform`. The base classes and
:class:`~few.waveform.base.SparseInfoHolder` live in :mod:`few.waveform.base`.

Example:
    .. code-block:: python

        from few.waveform import GenerateEMRIWaveform

        gen = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux")
"""

from .waveform import (
    FastKerrEccentricEquatorialFlux,
    FastSchwarzschildEccentricFlux,
    FastSchwarzschildEccentricFluxBicubic,
    GenerateEMRIWaveform,
    Pn5AAKWaveform,
    SlowSchwarzschildEccentricFlux,
)

__all__ = [
    "GenerateEMRIWaveform",
    "FastKerrEccentricEquatorialFlux",
    "FastSchwarzschildEccentricFlux",
    "FastSchwarzschildEccentricFluxBicubic",
    "SlowSchwarzschildEccentricFlux",
    "Pn5AAKWaveform",
]
