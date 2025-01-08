"""Definition of FEW package common exceptions"""

class FewException(Exception):
    """Base class for FEW package exceptions."""
    pass

class BackendUnavailable(FewException):
    """Exception raised when the backend is not available."""
    pass
