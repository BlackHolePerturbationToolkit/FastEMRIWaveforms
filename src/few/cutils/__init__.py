from . import cpu

from .fast_selector import import_fast_backend, BackendSelectionMode

fast = import_fast_backend(BackendSelectionMode.BEST)

def select_best_backend():
    global fast
    fast = import_fast_backend(BackendSelectionMode.BEST)

def select_lazy_backend():
    global fast
    fast = import_fast_backend(BackendSelectionMode.LAZY)

def force_cpu_backend():
    global fast
    fast = import_fast_backend(BackendSelectionMode.CPU)

def force_cuda11x_backend():
    global fast
    fast = import_fast_backend(BackendSelectionMode.CUDA11X)

def force_cuda12x_backend():
    global fast
    fast = import_fast_backend(BackendSelectionMode.CUDA12X)

__all__ = (
    "cpu",
    "fast",
    "select_best_backend",
    "select_lazy_backend",
    "force_cpu_backend",
    "force_cuda11x_backend",
    "force_cuda12x_backend",
)
