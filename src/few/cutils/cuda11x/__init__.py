import importlib

if importlib.import_module("few_backend_cuda11x") is None:
    from ...utils.exceptions import BackendNotInstalled
    raise BackendNotInstalled("CUDA 11.x backend")

try:
    from few_backend_cuda11x.pyAAK import pyWaveform
    from few_backend_cuda11x.pyAmpInterp2D import interp2D
    from few_backend_cuda11x.pyinterp import (
        interpolate_arrays_wrap,
        get_waveform_wrap,
        get_waveform_generic_fd_wrap,
    )
    from few_backend_cuda11x.pymatmul import neural_layer_wrap, transform_output_wrap
except ImportError as e:
    from few.utils.exceptions import BackendImportFailed
    raise BackendImportFailed("CUDA 11.x backend installed but not importable") from e

__backend__ = "cuda11x"

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
