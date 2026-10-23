"""FEW compute backends for the ``gpubackendtools`` registry.

Each concrete backend imports the four compiled extension modules of its plugin package
(``few_backend_cpu``, ``few_backend_cuda11x``, ``few_backend_cuda12x`` or
``few_backend_cuda13x``: ``pyAAK``, ``pyAmpInterp2D``, ``pyinterp``, ``pymatmul``) and
exposes their functions through :class:`FewBackendMethods` / :class:`FEWBackend`, next to
``xp`` (``numpy`` or ``cupy``). :mod:`few` registers the classes at import under the
registry names ``few_cpu``, ``few_cuda11x``, ``few_cuda12x`` and ``few_cuda13x``; a FEW
module built with ``force_backend="cuda13x"`` (as the LISA global-fit EMRI path does)
resolves to :class:`FEWCuda13xBackend`. The trajectory code (:mod:`few.trajectory`) is pure
Python / numba and uses no backend.

To add a native function: bind it in a ``.pyx`` file, add a field to
:class:`FewBackendMethods`, copy it in :meth:`FEWBackend.__init__` and fill it in every
loader.

Note:
    The loaders raise :class:`gpubackendtools.exceptions.BackendUnavailableException`
    (star-imported from :mod:`gpubackendtools.exceptions`) when one of the extension
    modules cannot be imported. That is not the class the ``gpubackendtools`` registry
    catches (:class:`gpubackendtools.gpubackendtools.BackendUnavailableException`), so
    such a failure propagates out of ``get_backend`` / ``has_backend`` instead of marking
    the backend unavailable; a plugin package that is not installed at all is caught
    earlier, by the ``gpubackendtools`` base-class check. The follow-up numpy / cupy check
    references ``MissingDependencies``, which that star import does not provide
    (``gpubackendtools.exceptions`` defines ``MissingDependency``), so a missing numpy /
    cupy surfaces as a ``NameError``.
"""

import dataclasses
import enum
import types
import typing

from ..utils.exceptions import FewException

from gpubackendtools.gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend, Cuda13xBackend
from gpubackendtools.exceptions import *

@dataclasses.dataclass
class FewBackendMethods(BackendMethods):
    """Method table of a FEW backend, built by each backend's loader.

    Attributes:
        xp (types.ModuleType): Array module (``numpy`` or ``cupy``), inherited from
            :class:`gpubackendtools.gpubackendtools.BackendMethods`.
        pyWaveform (Callable): AAK waveform kernel (``pyAAK``), used by
            :class:`few.summation.aakwave.AAKSummation`.
        interp2D (Callable): Bicubic amplitude-spline evaluation (``pyAmpInterp2D``,
            wrapped with :func:`few.cutils.wrappers.pointer_adjust`), used by
            :class:`few.amplitude.ampinterp2d.AmpInterp2D` and
            :class:`few.amplitude.ampinterp2d.AmpInterpSchwarzEcc`.
        interpolate_arrays_wrap (Callable): Batched cubic-spline fit (``pyinterp``), used
            by :class:`few.summation.interpolatedmodesum.CubicSplineInterpolant`.
        get_waveform_wrap (Callable): Time-domain interpolated mode sum (``pyinterp``),
            used by :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.
        get_waveform_generic_fd_wrap (Callable): Frequency-domain mode sum (``pyinterp``),
            used by :class:`few.summation.fdinterp.FDInterpolatedModeSum`.
        neural_layer_wrap (Callable): Dense neural-network layer (``pymatmul``), used by
            :class:`few.amplitude.romannet.RomanAmplitude`.
        transform_output_wrap (Callable): Output transform of the ROMAN network
            (``pymatmul``), used by :class:`few.amplitude.romannet.RomanAmplitude`.
    """

    pyWaveform: typing.Callable[(...), None]
    interp2D: typing.Callable[(...), None]
    interpolate_arrays_wrap: typing.Callable[(...), None]
    get_waveform_wrap: typing.Callable[(...), None]
    get_waveform_generic_fd_wrap: typing.Callable[(...), None]
    neural_layer_wrap: typing.Callable[(...), None]
    transform_output_wrap: typing.Callable[(...), None]


class FEWBackend:
    """Mixin copying the :class:`FewBackendMethods` fields onto the backend instance.

    Concrete backends inherit from a ``gpubackendtools`` device base (``CpuBackend``,
    ``Cuda13xBackend``, ...) and from this class, so ``backend.interp2D``,
    ``backend.get_waveform_wrap``, ``backend.xp``, ... are plain attributes.

    Args:
        few_backend_methods (FewBackendMethods): The loaded method table.

    Raises:
        AssertionError: If ``few_backend_methods`` is not a :class:`FewBackendMethods`.
    """

    pyWaveform: typing.Callable[(...), None]
    interp2D: typing.Callable[(...), None]
    interpolate_arrays_wrap: typing.Callable[(...), None]
    get_waveform_wrap: typing.Callable[(...), None]
    get_waveform_generic_fd_wrap: typing.Callable[(...), None]
    neural_layer_wrap: typing.Callable[(...), None]
    transform_output_wrap: typing.Callable[(...), None]

    def __init__(self, few_backend_methods):

        # set direct gbgpu methods
        # pass rest to general backend
        assert isinstance(few_backend_methods, FewBackendMethods)

        self.pyWaveform = few_backend_methods.pyWaveform
        self.interp2D = few_backend_methods.interp2D
        self.interpolate_arrays_wrap = few_backend_methods.interpolate_arrays_wrap
        self.get_waveform_wrap = few_backend_methods.get_waveform_wrap
        self.get_waveform_generic_fd_wrap = few_backend_methods.get_waveform_generic_fd_wrap
        self.neural_layer_wrap = few_backend_methods.neural_layer_wrap
        self.transform_output_wrap = few_backend_methods.transform_output_wrap
        self.xp = few_backend_methods.xp

        # TODO: what to do with this
        # self.features = features

class FEWCpuBackend(CpuBackend, FEWBackend):
    """CPU backend (registry name ``few_cpu``, plugin package ``few_backend_cpu``).

    Loads the CPU builds of ``pyAAK``, ``pyAmpInterp2D``, ``pyinterp`` and ``pymatmul``
    with ``xp=numpy``.

    Args:
        *args: Forwarded to :class:`gpubackendtools.gpubackendtools.CpuBackend`.
        **kwargs: Forwarded to :class:`gpubackendtools.gpubackendtools.CpuBackend`.

    Example:
        .. code-block:: python

            import few
            from gpubackendtools import get_backend

            backend = get_backend("few_cpu")
            backend.xp  # numpy
    """
    
    _backend_name = "few_backend_cpu"
    _name = "few_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        FEWBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> FewBackendMethods:
        """Import the ``few_backend_cpu`` extension modules and build the method table.

        Returns:
            FewBackendMethods: Table with ``xp=numpy``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If a ``few_backend_cpu``
                module cannot be imported.
            NameError: If numpy cannot be imported (the intended ``MissingDependencies``
                is not defined in this module).
        """
        try:
            import few_backend_cpu.pyAAK
            import few_backend_cpu.pyAmpInterp2D
            import few_backend_cpu.pyinterp
            import few_backend_cpu.pymatmul
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        try:
            import numpy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cpu' backend requires numpy", pip_deps=["numpy"], conda_deps=["numpy"]
            ) from e

        return FewBackendMethods(
            pyWaveform=few_backend_cpu.pyAAK.pyWaveform,
            interp2D=few_backend_cpu.pyAmpInterp2D.interp2D,
            interpolate_arrays_wrap=few_backend_cpu.pyinterp.interpolate_arrays_wrap,
            get_waveform_wrap=few_backend_cpu.pyinterp.get_waveform_wrap,
            get_waveform_generic_fd_wrap=few_backend_cpu.pyinterp.get_waveform_generic_fd_wrap,
            neural_layer_wrap=few_backend_cpu.pymatmul.neural_layer_wrap,
            transform_output_wrap=few_backend_cpu.pymatmul.transform_output_wrap,
            xp=numpy,
        )


class FEWCuda13xBackend(Cuda13xBackend, FEWBackend):
    """CUDA 13.x backend (registry name ``few_cuda13x``, plugin ``few_backend_cuda13x``).

    Loads the CUDA builds of ``pyAAK``, ``pyAmpInterp2D``, ``pyinterp`` and ``pymatmul``
    with ``xp=cupy``. This is the backend of the LISA global-fit EMRI path
    (``force_backend="cuda13x"``): its ``interp2D`` evaluates the Kerr eccentric-equatorial
    amplitudes on the GPU.

    Args:
        *args: Forwarded to :class:`gpubackendtools.gpubackendtools.Cuda13xBackend`.
        **kwargs: Forwarded to :class:`gpubackendtools.gpubackendtools.Cuda13xBackend`.
    """
    _backend_name : str = "few_backend_cuda13x"
    _name = "few_cuda13x"

    def __init__(self, *args, **kwargs):
        Cuda13xBackend.__init__(self, *args, **kwargs)
        FEWBackend.__init__(self, self.cuda13x_module_loader())

    @staticmethod
    def cuda13x_module_loader():
        """Import the ``few_backend_cuda13x`` extension modules and build the method table.

        Returns:
            FewBackendMethods: Table with ``xp=cupy``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If a
                ``few_backend_cuda13x`` module cannot be imported.
            NameError: If cupy cannot be imported (the intended ``MissingDependencies``
                is not defined in this module).
        """
        try:
            import few_backend_cuda13x.pyAAK
            import few_backend_cuda13x.pyAmpInterp2D
            import few_backend_cuda13x.pyinterp
            import few_backend_cuda13x.pymatmul
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda13x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda13x' backend requires cupy", pip_deps=["cupy-cuda13x"]
            ) from e

        return FewBackendMethods(
            pyWaveform=few_backend_cuda13x.pyAAK.pyWaveform,
            interp2D=few_backend_cuda13x.pyAmpInterp2D.interp2D,
            interpolate_arrays_wrap=few_backend_cuda13x.pyinterp.interpolate_arrays_wrap,
            get_waveform_wrap=few_backend_cuda13x.pyinterp.get_waveform_wrap,
            get_waveform_generic_fd_wrap=few_backend_cuda13x.pyinterp.get_waveform_generic_fd_wrap,
            neural_layer_wrap=few_backend_cuda13x.pymatmul.neural_layer_wrap,
            transform_output_wrap=few_backend_cuda13x.pymatmul.transform_output_wrap,
            xp=cupy,
        )


class FEWCuda12xBackend(Cuda12xBackend, FEWBackend):
    """CUDA 12.x backend (registry name ``few_cuda12x``, plugin ``few_backend_cuda12x``).

    Loads the CUDA builds of ``pyAAK``, ``pyAmpInterp2D``, ``pyinterp`` and ``pymatmul``
    with ``xp=cupy``.

    Args:
        *args: Forwarded to :class:`gpubackendtools.gpubackendtools.Cuda12xBackend`.
        **kwargs: Forwarded to :class:`gpubackendtools.gpubackendtools.Cuda12xBackend`.
    """
    _backend_name : str = "few_backend_cuda12x"
    _name = "few_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        FEWBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        """Import the ``few_backend_cuda12x`` extension modules and build the method table.

        Returns:
            FewBackendMethods: Table with ``xp=cupy``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If a
                ``few_backend_cuda12x`` module cannot be imported.
            NameError: If cupy cannot be imported (the intended ``MissingDependencies``
                is not defined in this module).
        """
        try:
            import few_backend_cuda12x.pyAAK
            import few_backend_cuda12x.pyAmpInterp2D
            import few_backend_cuda12x.pyinterp
            import few_backend_cuda12x.pymatmul
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return FewBackendMethods(
            pyWaveform=few_backend_cuda12x.pyAAK.pyWaveform,
            interp2D=few_backend_cuda12x.pyAmpInterp2D.interp2D,
            interpolate_arrays_wrap=few_backend_cuda12x.pyinterp.interpolate_arrays_wrap,
            get_waveform_wrap=few_backend_cuda12x.pyinterp.get_waveform_wrap,
            get_waveform_generic_fd_wrap=few_backend_cuda12x.pyinterp.get_waveform_generic_fd_wrap,
            neural_layer_wrap=few_backend_cuda12x.pymatmul.neural_layer_wrap,
            transform_output_wrap=few_backend_cuda12x.pymatmul.transform_output_wrap,
            xp=cupy,
        )
    

class FEWCuda11xBackend(Cuda11xBackend, FEWBackend):
    """CUDA 11.x backend (registry name ``few_cuda11x``, plugin ``few_backend_cuda11x``).

    Loads the CUDA builds of ``pyAAK``, ``pyAmpInterp2D``, ``pyinterp`` and ``pymatmul``
    with ``xp=cupy``.

    Args:
        *args: Forwarded to :class:`gpubackendtools.gpubackendtools.Cuda11xBackend`.
        **kwargs: Forwarded to :class:`gpubackendtools.gpubackendtools.Cuda11xBackend`.
    """
    _backend_name : str = "few_backend_cuda11x"
    _name = "few_cuda11x"
    
    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        FEWBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        """Import the ``few_backend_cuda11x`` extension modules and build the method table.

        Returns:
            FewBackendMethods: Table with ``xp=cupy``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If a
                ``few_backend_cuda11x`` module cannot be imported.
            NameError: If cupy cannot be imported (the intended ``MissingDependencies``
                is not defined in this module).
        """
        try:
            import few_backend_cuda11x.pyAAK
            import few_backend_cuda11x.pyAmpInterp2D
            import few_backend_cuda11x.pyinterp
            import few_backend_cuda11x.pymatmul
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return FewBackendMethods(
            pyWaveform=few_backend_cuda11x.pyAAK.pyWaveform,
            interp2D=few_backend_cuda11x.pyAmpInterp2D.interp2D,
            interpolate_arrays_wrap=few_backend_cuda11x.pyinterp.interpolate_arrays_wrap,
            get_waveform_wrap=few_backend_cuda11x.pyinterp.get_waveform_wrap,
            get_waveform_generic_fd_wrap=few_backend_cuda11x.pyinterp.get_waveform_generic_fd_wrap,
            neural_layer_wrap=few_backend_cuda11x.pymatmul.neural_layer_wrap,
            transform_output_wrap=few_backend_cuda11x.pymatmul.transform_output_wrap,
            xp=cupy,
        )

# dict short name -> backend class; its keys validate the ``enabled_backends`` entry of
# few.utils.config (backend selection itself goes through the gpubackendtools registry)
KNOWN_BACKENDS = {
    "cuda13x": FEWCuda13xBackend,
    "cuda12x": FEWCuda12xBackend,
    "cuda11x": FEWCuda11xBackend,
    "cpu": FEWCpuBackend,
}
"""Dict of the existing backends: short name -> backend class, in default order of preference."""

