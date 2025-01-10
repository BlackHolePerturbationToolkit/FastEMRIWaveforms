from ..fast_selector import import_fast_backend, BackendSelectionMode

def load_backend(backend):
    global pyWaveform, interp2D, interpolate_arrays_wrap, get_waveform_wrap, get_waveform_generic_fd_wrap, neural_layer_wrap, transform_output_wrap, __backend__
    pyWaveform = backend.pyWaveform
    interp2D = backend.interp2D
    interpolate_arrays_wrap = backend.interpolate_arrays_wrap
    get_waveform_wrap = backend.get_waveform_wrap
    get_waveform_generic_fd_wrap = backend.get_waveform_generic_fd_wrap
    neural_layer_wrap = backend.neural_layer_wrap
    transform_output_wrap = backend.transform_output_wrap
    __backend__ = backend.__backend__

def select_best_backend():
    load_backend(import_fast_backend(BackendSelectionMode.BEST))

def select_lazy_backend():
    load_backend(import_fast_backend(BackendSelectionMode.LAZY))

def force_cpu_backend():
    load_backend(import_fast_backend(BackendSelectionMode.CPU))

def force_cuda11x_backend():
    load_backend(import_fast_backend(BackendSelectionMode.CUDA11X))

def force_cuda12x_backend():
    load_backend(import_fast_backend(BackendSelectionMode.CUDA12X))

load_backend(import_fast_backend(BackendSelectionMode.LAZY))

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
