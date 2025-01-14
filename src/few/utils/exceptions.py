"""Definition of FEW package common exceptions"""

class FewException(Exception):
    """Base class for FEW package exceptions."""
    pass

class CudaException(FewException):
    """Base class for CUDA-related exceptions."""
    pass

class CuPyException(FewException):
    """Base class for CuPy-related exceptions."""
    pass

class MissingDependency(FewException):
    """Exception raised when a required dependency is missing."""
    pass

class BackendUnavailable(FewException):
    """Exception raised when the backend is not available."""
    pass

class BackendNotInstalled(BackendUnavailable):
    """Exception raised when the backend is not installed."""
    pass

class BackendImportFailed(BackendUnavailable):
    """Exception raised when the backend import fails."""
    pass

class InvalidInputFile(FewException):
    """Exception raised when the content of an input file does not match expectations."""
