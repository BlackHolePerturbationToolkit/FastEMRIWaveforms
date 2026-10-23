"""Definition of FEW package common exceptions.

- :class:`FewException` and its subclasses: configuration errors (raised by
  :mod:`few.utils.config`), file-manager and download errors (:mod:`few.files`), invalid
  input files (citations, file registry).
- :class:`TrajectoryOffGridException`: raised by the KerrEccEq flux trajectory when it
  leaves the flux grid; it is not a ``ValueError``, so the integrator does not retry the
  step and it reaches the caller (LISA Analysis Tools catches it in its EMRI domain
  guard).
- ``ExceptionGroup``: the ``exceptiongroup`` backport if installed, else the builtin
  (Python >= 3.11); raised for several configuration errors at once.

Backend errors come from ``gpubackendtools.exceptions``.
"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    ExceptionGroup = ExceptionGroup


class FewException(Exception):
    """Base class for FEW package exceptions."""

    pass


# TODO check: keep/remove/document -- never raised or caught in FEW
class CudaException(FewException):
    """Base class for CUDA-related exceptions."""

    pass


# TODO check: keep/remove/document -- never raised or caught in FEW
class CuPyException(FewException):
    """Base class for CuPy-related exceptions."""

    pass


# TODO check: keep/remove/document -- never raised or caught in FEW
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
    """Exception raised when trying to access a file whose tags are disabled.

    Args:
        *args: Forwarded to :class:`Exception`.
        disabled_tag: The disabled tag (keyword-only).
        file_name: The requested file (keyword-only).
        **kwargs: Forwarded to :class:`Exception`.
    """

    disabled_tag: str
    file_name: str

    def __init__(self, /, *args, disabled_tag: str, file_name: str, **kwargs):
        self.disabled_tag = disabled_tag
        self.file_name = file_name
        super().__init__(*args, **kwargs)


### Trajectory-related exceptions
class TrajectoryOffGridException(Exception):
    """Exception raised when a trajectory goes off the flux grid.

    Raised by :meth:`~few.trajectory.ode.flux.KerrEccEqFlux.interpolate_flux_grids` when the
    eccentricity coordinate leaves the grid (``w < 0``, or ``w > 1`` in a forward
    integration) or the spin coordinate ``z`` leaves ``[0, 1]``; a ``p`` outside the grid
    raises ``ValueError`` instead. Derives from :class:`Exception`, not
    :class:`FewException` or ``ValueError``.
    """
