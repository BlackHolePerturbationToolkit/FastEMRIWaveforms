"""Module which aliases either cuda12x, cuda11x or cpu for accelerated capabilities"""

from enum import Enum
from types import ModuleType
from typing import Sequence
from ...utils.exceptions import BackendUnavailable, BackendNotInstalled, CudaException, CuPyException, FewException, MissingDependency

class BackendSelectionMode(Enum):
    """
    Enum for selecting the backend mode.
    """
    BEST = "best"  # Determine the best backend best on available drivers and devices. Fails if it cannot be loaded.
    LAZY = "lazy"  # Automatically selects the best available backend in current environment.
    CPU = "cpu"  # Forces the CPU backend.
    CUDA11X = "cuda11x"  # Forces the CUDA 11.x backend.
    CUDA12X = "cuda12x"  # Forces the CUDA 12.x backend.

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
        import cupy_backends.cuda
    except ImportError as e:
        raise MissingDependency(f"CuPy is not installed. Run `pip install cupy-cuda{cuda_version[0]}x` to install it.") from e

    # Try basic CuPy function to check real-time compilation is working
    try:
        _ = cupy.arange(1)
    except cupy.cuda.compiler.CompileException as e:
        raise MissingDependency(f"CuPy fails to run due to missing dependencies. Run `pip install nvidia-cuda-runtime-cu{cuda_version[0]}=='{cuda_version[0]}.{cuda_version[1]}.*'` to install them.") from e
    except (cupy_backends.cuda.api.runtime.CUDARuntimeError, RuntimeError) as e:
        raise CuPyException("CuPy could not execute properly.") from e

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

def try_import_nvidia_so_libs(libs: Sequence[tuple[str, str, str]], cuda_version: tuple[int, int]) -> None:
    """Try to load a set of NVidia dynamic libraries."""
    import ctypes
    import importlib
    import pathlib

    try:
        nvidia_root = pathlib.Path(importlib.import_module("nvidia").__file__).parent
    except ModuleNotFoundError:
        nvidia_root = None

    failed_idx = []
    exceptions = []
    for idx, (_, module_name, soname) in enumerate(libs):
        try:
            ctypes.cdll.LoadLibrary(soname)
            continue
        except OSError as e:
            exceptions.append(e)

        try:
            if nvidia_root is not None:
                ctypes.cdll.LoadLibrary(nvidia_root / module_name / "lib" / soname)
                continue
        except OSError as e:
            exceptions.append(e)

        failed_idx.append(idx)

    if failed_idx:
        failed_libs = [libs[idx][2] for idx in failed_idx]
        packages = " ".join([f"{lib[0]}=='{cuda_version[0]}.{cuda_version[1]}.*'" for lib in libs])

        raise MissingDependency(f"Could not load the following NVidia libraries: {failed_libs}. "
                                f"If you installed FEW using pip, you may install them by running"
                                f"`pip install {packages}`. ") from ExceptionGroup("Following exceptions were raised when trying to load these libraries", exceptions)


def try_preload_cuda12x_dynlibs(cuda_version: tuple[int, int]):
    """Preload dynamic libraries required for CUDA 12x backend"""
    import sys
    if sys.platform == "linux":
        cuda12x_solibs = [
            ("nvidia-cuda-runtime-cu12", "cuda_runtime", "libcudart.so.12"),
            ("nvidia-cublas-cu12", "cublas", "libcublas.so.12"),
            ("nvidia-nvjitlink-cu12", "nvjitlink", "libnvJitLink.so.12"),
            ("nvidia-cusparse-cu12", "cusparse", "libcusparse.so.12"),
            ("nvidia-cuda-nvrtc-cu12", "cuda_nvrtc", "libnvrtc.so.12"),
            ("nvidia-cufft-cu12", "cufft", "libcufftw.so.11"),
        ]
        try_import_nvidia_so_libs(cuda12x_solibs, cuda_version)

def try_preload_cuda11x_dynlibs(cuda_version: tuple[int, int]):
    """Preload dynamic libraries required for CUDA 11x backend"""
    import sys
    if sys.platform == "linux":
        cuda11x_solibs = [
            ("nvidia-cuda-runtime-cu11", "cuda_runtime", "libcudart.so.11"),
            ("nvidia-cublas-cu11", "cublas", "libcublas.so.11"),
            ("nvidia-nvjitlink-cu11", "nvjitlink", "libnvJitLink.so.11"),
            ("nvidia-cusparse-cu11", "cusparse", "libcusparse.so.11"),
            ("nvidia-cuda-nvrtc-cu11", "cuda_nvrtc", "libnvrtc.so.11"),
            ("nvidia-cufft-cu11", "cufft", "libcufftw.so.11"),
        ]
        try_import_nvidia_so_libs(cuda11x_solibs, cuda_version)


def force_import_cuda11x_backend(cuda_version: tuple[int, int]) -> ModuleType:
    """Force import the cuda11x backend."""
    # If this method is called, the CUDA driver is installed with version 12.x.

    try:
        try_import_cuda11x()
    except BackendNotInstalled as e:
        raise BackendUnavailable("Backend CUDA 11.x is not installed. "
                                 "Run `pip install fastemriwaveforms-cuda11x` to install it.") from e

    # Check that nvidia dynamic libraries are available
    try:
        try_preload_cuda11x_dynlibs(cuda_version)
    except MissingDependency as e:
        raise BackendUnavailable("Some CUDA libraries required by FEW are not available. Please follow previous instructions.") from e

    # Then check that CuPy works
    try:
        check_cupy_works()
    except CuPyException as e:
        raise BackendUnavailable("Your system has CUDA 11.x installed but CuPy does not work. Follow previous instructions.") from e

    # Then try to import the cuda12x backend
    return try_import_cuda11x()

def force_import_cuda12x_backend(cuda_version: tuple[int, int]) -> ModuleType:
    """Force import the cuda12x backend."""
    # If this method is called, the CUDA driver is installed with version 12.x.

    try:
        try_import_cuda12x()
    except BackendNotInstalled as e:
        raise BackendUnavailable("Backend CUDA 12.x is not installed. "
                                 "Run `pip install fastemriwaveforms-cuda12x` to install it.") from e

    # Check that nvidia dynamic libraries are available
    try:
        try_preload_cuda12x_dynlibs(cuda_version)
    except MissingDependency as e:
        raise BackendUnavailable("Some CUDA libraries required by FEW are not available. Please follow previous instructions.") from e

    # Then check that CuPy works
    try:
        check_cupy_works()
    except CuPyException as e:
        raise BackendUnavailable("Your system has CUDA 12.x installed but CuPy does not work. Follow previous instructions.") from e

    # Then try to import the cuda12x backend
    return try_import_cuda12x()

def import_lazy_backend() -> ModuleType:
    """Determine the best available backend in current environment."""
    try:
        cuda_version = get_cuda_version()
    except CudaException:
        return import_cpu_backend()

    try:
        try_preload_cuda12x_dynlibs(cuda_version)
        check_cupy_works()
        return try_import_cuda12x()
    except FewException:
        pass

    try:
        try_preload_cuda11x_dynlibs(cuda_version)
        check_cupy_works()
        return try_import_cuda11x()
    except FewException:
        pass

    return import_cpu_backend()


def import_best_backend() -> ModuleType:
    """Determine the best backend based on available drivers and devices."""
    try:
        cuda_version = get_cuda_version()
    except CudaException:
        return import_cpu_backend()

    if cuda_version[0] == 12:
        return force_import_cuda12x_backend(cuda_version)

    if cuda_version[0] == 11:
        return force_import_cuda11x_backend(cuda_version)

    return import_cpu_backend()

def import_fast_backend(mode: BackendSelectionMode) -> ModuleType:
    """
    Selects the appropriate backend module based on the provided mode.
    """

    if mode == BackendSelectionMode.LAZY:
        return import_lazy_backend()

    if mode == BackendSelectionMode.BEST:
        return import_best_backend()

    if mode == BackendSelectionMode.CPU:
        return import_cpu_backend()

    if mode == BackendSelectionMode.CUDA11X:
        return force_import_cuda11x_backend(get_cuda_version())

    if mode == BackendSelectionMode.CUDA12X:
        return force_import_cuda12x_backend(get_cuda_version())

    raise ValueError(f"Invalid backend selection mode: {mode}")
