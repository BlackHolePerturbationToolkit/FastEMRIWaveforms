import importlib

if importlib.import_module("few_backend_cpu") is None:
    from ...utils.exceptions import BackendNotInstalled
    raise BackendNotInstalled("CPU backend")

from few_backend_cpu.pyAAK import pyWaveform
from few_backend_cpu.pyAmpInterp2D import interp2D
from few_backend_cpu.pyinterp import (
    interpolate_arrays_wrap,
    get_waveform_wrap,
    get_waveform_generic_fd_wrap,
)
from few_backend_cpu.pymatmul import neural_layer_wrap, transform_output_wrap

__backend__ = "cpu"

__all__ = [
    "pyWaveform",
    "interp2D",
    "interpolate_arrays_wrap",
    "get_waveform_wrap",
    "get_waveform_generic_fd_wrap",
    "neural_layer_wrap",
    "transform_output_wrap",
    "__backend__",
]
