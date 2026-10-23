"""Data-file management: locate, verify and download FEW's precomputed data files.

:class:`~few.files.manager.FileManager` resolves a file name against the
:class:`~few.files.registry.FileRegistry` (the packaged ``registry.yml``: repositories, file
checksums and tags), looks for it in the local search paths, checks its checksums and, if
allowed, downloads it. The global instance, configured from the FEW configuration
(``file_*`` options, see the configuration guide), is returned by
:func:`few.get_file_manager`; the ``few_files`` command (:mod:`few.cmd.files`) prefetches
or lists the files.

Example:
    .. code-block:: python

        from few import get_file_manager

        path = get_file_manager().get_file("KerrEccEqFluxData.h5")  # pathlib.Path
"""

from .manager import FileManager
from .registry import FileRegistry

__all__ = ["FileManager", "FileRegistry"]
