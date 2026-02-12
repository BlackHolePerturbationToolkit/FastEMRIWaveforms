"""Fast and accurate EMRI Waveforms."""

# ruff: noqa: E402
try:
    from few._version import (  # pylint: disable=E0401,E0611
        __version__,
        __version_tuple__,
    )

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False

from . import amplitude, cutils, files, summation, trajectory, utils, waveform


from gpubackendtools.globals import Globals


from .cutils import FEWCpuBackend, FEWCuda11xBackend, FEWCuda12xBackend

add_backends = {
    "few_cpu": FEWCpuBackend,
    "few_cuda11x": FEWCuda11xBackend,
    "few_cuda12x": FEWCuda12xBackend,
}

Globals().backends_manager.add_backends(add_backends)


__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
    "amplitude",
    "cutils",
    "files",
    "summation",
    "trajectory",
    "utils",
    "waveform",
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_file_manager",
    "has_backend",
]
