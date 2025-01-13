from ..fast_selector import BackendSelectionMode

def load_backend(mode: BackendSelectionMode):
    from ..fast_selector import import_fast_backend

    global pyWaveform, interp2D, interpolate_arrays_wrap, get_waveform_wrap, get_waveform_generic_fd_wrap, neural_layer_wrap, transform_output_wrap, __backend__, xp, is_gpu

    backend = import_fast_backend(mode)
    pyWaveform = backend.pyWaveform
    interp2D = backend.interp2D
    interpolate_arrays_wrap = backend.interpolate_arrays_wrap
    get_waveform_wrap = backend.get_waveform_wrap
    get_waveform_generic_fd_wrap = backend.get_waveform_generic_fd_wrap
    neural_layer_wrap = backend.neural_layer_wrap
    transform_output_wrap = backend.transform_output_wrap
    __backend__ = backend.__backend__
    xp = backend.xp
    is_gpu = backend.is_gpu

load_backend(BackendSelectionMode.LAZY)

__all__ = [
    "BackendSelectionMode"
    "pyWaveform",
    "interp2D",
    "interpolate_arrays_wrap",
    "get_waveform_wrap",
    "get_waveform_generic_fd_wrap",
    "neural_layer_wrap",
    "transform_output_wrap",
    "__backend__",
    "is_gpu",
    "xp",
]
