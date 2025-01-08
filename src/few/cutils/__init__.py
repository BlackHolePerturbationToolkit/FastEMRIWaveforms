from . import cpu

from .fast_selector import import_fast_backend, BackendSelectionMode

fast = cpu

__all__ = (
    "cpu",
    "fast",
)
