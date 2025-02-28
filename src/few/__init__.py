"""Fast and accurate EMRI Waveforms."""

try:
    from few._version import __version__, __version_tuple__  # pylint: disable=E0401,E0611

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = __version__.split(".")
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

from .utils.globals import (
    get_logger,
    get_backend,
    get_config,
    get_file_manager,
    get_config_setter,
    has_backend,
)

from . import amplitude, cutils, files, summation, trajectory, utils, waveform

try:
    from . import _editable

    _is_editable: bool = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable: bool = False

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
    "log",
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_file_manager",
    "has_backend",
]
