from . import cpu
from . import fast

def guide_best_backend():
    """Select the fastest backend. Fails if missing dependencies to use it."""
    fast.load_backend(fast.BackendSelectionMode.BEST)

def load_best_backend():
    """Select the fastest backend available without installing any dependency."""
    fast.load_backend(fast.BackendSelectionMode.LAZY)

def force_cpu_backend():
    """Force usage of CPU backend for fast operations."""
    fast.load_backend(fast.BackendSelectionMode.CPU)

def force_cuda11x_backend():
    """Force usage of CUDA 11.x backend for fast operations."""
    fast.load_backend(fast.BackendSelectionMode.CUDA11X)

def force_cuda12x_backend():
    """Force usage of CUDA 12.x backend for fast operations."""
    fast.load_backend(fast.BackendSelectionMode.CUDA12X)

__all__ = (
    "cpu",
    "fast",
    "guide_best_backend",
    "load_best_backend",
    "force_cpu_backend",
    "force_cuda11x_backend",
    "force_cuda12x_backend",
)
