import dataclasses
import enum
import types
import typing

from ..utils.exceptions import FewException

from gpubackendtools.gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend
from gpubackendtools.utils.exceptions import *

@dataclasses.dataclass
class FewBackendMethods(BackendMethods):
    pyWaveform: typing.Callable[(...), None]
    interp2D: typing.Callable[(...), None]
    interpolate_arrays_wrap: typing.Callable[(...), None]
    get_waveform_wrap: typing.Callable[(...), None]
    get_waveform_generic_fd_wrap: typing.Callable[(...), None]
    neural_layer_wrap: typing.Callable[(...), None]
    transform_output_wrap: typing.Callable[(...), None]


class FEWBackend:
    """Abstract definition of a backend"""

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
    """Implementation of the CPU backend"""
    
    _backend_name = "few_backend_cpu"
    _name = "few_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        FEWBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> FewBackendMethods:
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


class FEWCuda12xBackend(Cuda12xBackend, FEWBackend):
    """Implementation of CUDA 12.x backend"""
    _backend_name : str = "few_backend_cuda12x"
    _name = "few_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        FEWBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
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
    """Implementation of CUDA 11.x backend"""
    _backend_name : str = "few_backend_cuda11x"
    _name = "few_cuda11x"
    
    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        FEWBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
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

KNOWN_BACKENDS = {
    "cuda12x": FEWCuda12xBackend,
    "cuda11x": FEWCuda11xBackend,
    "cpu": FEWCpuBackend,
}
"""List of existing backends, per default order of preference."""

