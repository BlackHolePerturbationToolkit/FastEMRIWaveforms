"""FastEMRIWaveforms (``few``): fast and accurate extreme-mass-ratio-inspiral (EMRI) waveforms.

Importing the package loads its subpackages (:mod:`few.amplitude`, :mod:`few.cutils`,
:mod:`few.files`, :mod:`few.summation`, :mod:`few.trajectory`, :mod:`few.utils`,
:mod:`few.waveform`) and registers the four FEW backends with the ``gpubackendtools``
registry (``Globals().backends_manager.add_backends``) under the names ``few_cpu``,
``few_cuda11x``, ``few_cuda12x`` and ``few_cuda13x`` (classes in :mod:`few.cutils`).
Registration does not build a backend: the registry builds one on its first lookup, so a
missing compiled plugin (``few_backend_cpu``, ``few_backend_cuda13x``, ...) only fails when
that backend is requested.

Main entry points: the waveform generators of :mod:`few.waveform` (for example
:class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux` and
:class:`~few.waveform.waveform.GenerateEMRIWaveform`), the trajectory module
:class:`~few.trajectory.inspiral.EMRIInspiral`, and the package globals re-exported from
:mod:`few.utils.globals`: :func:`get_logger`, :func:`get_config`,
:func:`get_config_setter` and :func:`get_file_manager`.

Attributes:
    __version__ (str): Package version (from the generated ``few._version``, else the
        installed distribution metadata, else ``"unknown"``).
    __version_tuple__ (tuple): Version split into its components.
    _is_editable (bool): ``True`` when the package is an editable install (the generated
        ``few._editable`` module is importable). In that case the default file storage and
        download directories are ``src/few/data`` (see :mod:`few.utils.config`) and the
        citation file is read from the repository root.

Example:
    .. code-block:: python

        import few
        from few.waveform import FastKerrEccentricEquatorialFlux

        print(few.__version__)
        few.get_logger().setLevel("INFO")
"""

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


from .cutils import FEWCpuBackend, FEWCuda11xBackend, FEWCuda12xBackend, FEWCuda13xBackend

# registry name -> backend class; registered once at import, each backend is built by the
# gpubackendtools registry on its first lookup (e.g. force_backend="cuda13x" on a FEW module)
add_backends = {
    "few_cpu": FEWCpuBackend,
    "few_cuda11x": FEWCuda11xBackend,
    "few_cuda12x": FEWCuda12xBackend,
    "few_cuda13x": FEWCuda13xBackend,
}

Globals().backends_manager.add_backends(add_backends)

from .utils.globals import (
    get_config,
    get_config_setter,
    get_file_manager,
    get_logger,
)

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
    "get_file_manager",
]
