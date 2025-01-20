"""Fast and accurate EMRI Waveforms."""

try:
    from few._version import __version__ # pylint: disable=E0401,E0611

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
    try:
        __version__ = version(__name__)
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
    finally:
        del version, PackageNotFoundError

from . import amplitude, cutils, files, summation, trajectory, utils, waveform

from .utils.config import CONFIG as cfg

__all__ = [
    "__version__",
    "amplitude",
    "cutils",
    "files",
    "summation",
    "trajectory",
    "utils",
    "waveform",
    "cfg"
]
