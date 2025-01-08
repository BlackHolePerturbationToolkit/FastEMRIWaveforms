from few.utils.exceptions import BackendUnavailable
from . import cpu

try:
    from . import cuda12x as fast
except BackendUnavailable:
    try:
        from . import cuda11x as fast
    except BackendUnavailable:
        from . import cpu as fast

__all__ = (
    "cpu",
    "fast",
)
