"""Module which aliases either cuda12x, cuda11x or cpu for accelerated capabilities"""

from enum import Enum
from types import ModuleType
from typing import Sequence
from ...utils.exceptions import BackendUnavailable, CudaException, CuPyException, MissingDependency

class BackendSelectionMode(Enum):
    """
    Enum for selecting the backend mode.
    """
    BEST = "best"  # Determine the best backend best on available drivers and devices. Fails if it cannot be loaded.
    LAZY = "lazy"  # Automatically selects the best available backend in current environment.

def get_cuda_version() -> tuple[int, int]:
    """Get current CUDA version."""
    try:
        import pynvml
        pynvml.nvmlInit()
        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
    except pynvml.NVMLError_DriverNotLoaded as e:
        raise CudaException("CUDA driver not loaded: could not detect a CUDA version.") from e
    except pynvml.NVMLError as e:
        raise CudaException("CUDA exception: could not detect a CUDA version.") from e

    cuda_major = cuda_version // 1000
    cuda_minor = (cuda_version % 1000) // 10
    return (cuda_major, cuda_minor)

def check_cupy_works():
    """Check if cupy works."""

    # Check that CUDA version is detectable
    try:
        cuda_version = get_cuda_version()
    except CudaException as e:
        raise CuPyException("CuPy does not work due to CUDA error.") from e

    # Try to import CuPy
    try:
        import cupy
    except ImportError as e:
        raise MissingDependency(f"CuPy is not installed. Run `pip install cupy-cuda{cuda_version[0]}x` to install it.") from e

    # Try basic CuPy function to check real-time compilation is working
    try:
        _ = cupy.arange(1)
    except cupy.cuda.compiler.CompileException as e:
        raise MissingDependency(f"CuPy fails to run due to missing dependencies. Run `pip install nvidia-cuda-runtime-cu{cuda_version[0]}=='{cuda_version[0]}.{cuda_version[0]}{cuda_version[1]}.*'` to install them.") from e


def try_import_cuda11x() -> ModuleType:
    """Try to import the cuda11x backend."""
    try:
        cuda_version = get_cuda_version()
    except CudaException as e:
        raise BackendUnavailable("CUDA 11.x backend is not available: check CUDA drivers are properly installed on your system.") from e

    if cuda_version[0] != 11:
        raise BackendUnavailable(f"CUDA 11.x backend is not available: CUDA driver version should be 11.x but is {cuda_version[0]}.{cuda_version[1]}")

    try:
        from .. import cuda11x
    except ImportError as e:
        raise BackendUnavailable("CUDA 11.x backend is not importable.") from e
    return cuda11x

def try_import_cuda12x() -> ModuleType:
    """Try to import the cuda12x backend."""
    try:
        cuda_version = get_cuda_version()
    except CudaException as e:
        raise BackendUnavailable("CUDA 12.x backend is not available: check CUDA drivers are properly installed on your system.") from e

    if cuda_version[0] != 12:
        raise BackendUnavailable(f"CUDA 12.x backend is not available: CUDA driver version should be 12.x but is {cuda_version[0]}.{cuda_version[1]}")

    try:
        from .. import cuda12x
    except ImportError as e:
        raise BackendUnavailable("CUDA 12.x backend is not importable.") from e
    return cuda12x

def import_cpu_backend() -> ModuleType:
    """Import the CPU backend."""
    from .. import cpu
    return cpu

def import_so_libs(libs: Sequence[tuple[str, str, str]]) -> None:
    """Import a list of libraries."""
    raise NotImplementedError("Importing shared libraries is not implemented yet.")
    solibs = [item for item in libs.items()]

    import ctypes



def import_lazy_backend() -> ModuleType:
    """Determine the best available backend in current environment."""
    try:
        return try_import_cuda12x()
    except BackendUnavailable:
        pass

    try:
        return try_import_cuda11x()
    except BackendUnavailable:
        pass

    return import_cpu_backend()

def check_cuda12x_backend_installed():
    """Check that the cuda12x backend is installed."""
    import importlib.util
    import sys

    cuda12x_spec = importlib.util.find_spec("few.cutils.cuda12x")

def try_preload_cuda12x_dynlibs():
    """Preload dynamic libraries required for CUDA 12x backend"""
    import sys
    if sys.platform == "linux":
        cuda12x_solibs = [
            ("nvidia-cuda-runtime-cu12", "cuda_runtime", "libcudart.so.12"),
            ("nvidia-cublas-cu12", "cublas", "libcublas.so.12"),
            ("nvidia-cusparse-cu12", "cusparse", "libcusparse.so.12"),
            ("nvidia-nvjitlink-cu12", "nvjitlink", "libnvJitLink.so.12"),
            ("nvidia-cuda-nvrtc-cu12", "cuda_nvrtc", "libnvrtc.so.12"),
            ("nvidia-cufft-cu12", "cufft", "libcufftw.so.11"),
        ]
        try_import_so_libs(cuda12x_solibs)


def force_import_cuda12x_backend() -> ModuleType:
    """Force import the cuda12x backend."""
    # If this method is called, the CUDA driver is installed with version 12.x.

    # First check that CuPy works
    try:
        check_cupy_works()
    except CuPyException as e:
        raise BackendUnavailable("Your system has CUDA 12.x installed but CuPy does not work. Follow previous instructions.") from e

    # Then check whether the cuda12x backend is installed (either from source or as a plugin)
    try:
        check_cuda12x_backend_installed()
    except BackendUnavailable as e:
        raise BackendUnavailable("Your system has CUDA 12.x installed but the corresponding backend is not installed. "
                                 "If you installed FEW from pip, run `pip install fastemriwaveforms-cuda12x`. "
                                 "If you installed from source, ensure that you enabled GPU support (refer to the documentation).") from e

    # Then try to import the cuda12x backend
    try:
        return try_import_cuda12x()
    except BackendUnavailable as e:
        pass # Failure could occur if some dynamic dependencies are missing, let's try to import them

    # If we reach this point, the cuda12x backend is not importable. This could be due to missing dynamic dependencies.
    try:
        try_preload_cuda12x_dynlibs()
    except MissingDependency as e:
        raise BackendUnavailable("Some CUDA libraries required by FEW are not available. Please follow previous instructions.") from e


def import_best_backend() -> ModuleType:
    """Determine the best backend based on available drivers and devices."""
    try:
        cuda_version = get_cuda_version()
    except CudaException:
        return import_cpu_backend()

    if cuda_version[0] == 12:
        return force_import_cuda12x_backend()

    if cuda_version[0] == 11:
        return force_import_cuda11x_backend()

    return import_cpu_backend()

def import_fast_backend(mode: BackendSelectionMode) -> ModuleType:
    """
    Selects the appropriate backend module based on the provided mode.
    """

    if mode == BackendSelectionMode.AUTO_LAZY:
        return import_lazy_backend()

    if mode == BackendSelectionMode.AUTO_BEST:
        return import_best_backend()

    raise ValueError(f"Invalid backend selection mode: {mode}")
