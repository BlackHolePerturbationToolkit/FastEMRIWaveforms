try:
    from .pyAAK import pyWaveform
    from .pyAmpInterp2D import interp2D
    from .pyinterp import (
        interpolate_arrays_wrap,
        get_waveform_wrap,
        get_waveform_generic_fd_wrap,
    )
    from .pymatmul import neural_layer_wrap, transform_output_wrap
except ImportError as e:
    from few.utils.exceptions import BackendUnavailable
    raise BackendUnavailable("CUDA 11.x") from e

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
