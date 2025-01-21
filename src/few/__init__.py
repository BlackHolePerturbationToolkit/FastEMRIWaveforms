"""Fast and accurate EMRI Waveforms."""

try:
    from few._version import __version__, __version_tuple__ # pylint: disable=E0401,E0611

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
    try:
        __version__ = version(__name__)
        __version_tuple__ = __version__.split('.')
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, 'unknown')
    finally:
        del version, PackageNotFoundError

from . import amplitude, cutils, files, summation, trajectory, utils, waveform

from .utils.config import CONFIG as cfg
from .utils.logging import LOGGER as log, postconfig_install_handlers

# Post init tasks
cutils.fast.load_backend(cfg.fast_backend)
postconfig_install_handlers()
del postconfig_install_handlers

__all__ = [
    "__version__",
    "__version_tuple__",
    "amplitude",
    "cutils",
    "files",
    "summation",
    "trajectory",
    "utils",
    "waveform",
    "cfg",
    "log",
]
