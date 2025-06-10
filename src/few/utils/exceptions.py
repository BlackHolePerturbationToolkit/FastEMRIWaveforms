"""Definition of FEW package common exceptions"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    ExceptionGroup = ExceptionGroup


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


class InvalidInputFile(FewException):
    """Exception raised when the content of an input file does not match expectations."""


class ConfigurationError(FewException):
    """Exception raised when configuration setup fails."""


class ConfigurationMissing(ConfigurationError):
    """Exception raised when an expected configuration entry is missing."""


class ConfigurationValidationError(ConfigurationError):
    """Exception raised when a configuration entry is invalid."""


class FileManagerException(FewException):
    """Exception raised by the FileManager."""


class FileNotInRegistry(FileManagerException):
    """Exception raised when a requested file is not in file registry."""


class FileNotFoundLocally(FileManagerException):
    """Exception raised when file not found locally but expected to be."""


class FileInvalidChecksum(FileManagerException):
    """Exception raised when file has invalid checksum."""


class FileDownloadException(FileManagerException):
    """Exception raised if file download failed."""


class FileDownloadNotFound(FileDownloadException):
    """Exception raised if file is not found at expected URL."""


class FileDownloadConnectionError(FileDownloadException):
    """Exception raised in case of connection error during download."""


class FileDownloadIntegrityError(FileDownloadException):
    """Exception raised in case of integrity error after download."""


class FileManagerDisabledAccess(FileManagerException):
    """Exception raised when trying to access a file whose tags are disabled"""

    disabled_tag: str
    file_name: str

    def __init__(self, /, *args, disabled_tag: str, file_name: str, **kwargs):
        self.disabled_tag = disabled_tag
        self.file_name = file_name
        super().__init__(*args, **kwargs)


### Trajectory-related exceptions
class TrajectoryOffGridException(Exception):
    """Exception raised when a trajectory goes off-grid (except for the lower boundary in p)."""
