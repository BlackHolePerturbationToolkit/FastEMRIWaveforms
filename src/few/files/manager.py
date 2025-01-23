"""Class implementing the file manager."""

from __future__ import annotations

import enum
import os
import pathlib
import typing

import pydantic
import rich.progress

from .registry import FileRegistry
from ..utils import exceptions


class FileIntegrityCheckMode(enum.Enum):
    """Mode for file-integrity verification."""

    ALWAYS = "always"  # Check file integrity at each request
    ONCE = "once"  # Check file integrity once and keep result in cache
    NEVER = "never"  # Do not check file integrity


class FileMissingAction(enum.Enum):
    """Action to take when file is not in storage path (or its integrity check fails)."""

    DOWNLOAD = "download"  # Download the file from one of its repos
    FAIL = "fail"  # Fail the program (to be used in offline mode)


class FileUnknownAction(enum.Enum):
    """Action to take when a requested file is not known in the file registry."""

    ATTEMPT_ALL_REPOS = "attempt_all_repos"  # Try to locate file in known repositories
    FAIL = "fail"  # Fail the program


class FileManagerOptions(pydantic.BaseModel):
    """Options controliing the file manager behaviour"""

    download_path: pathlib.Path
    """Where to install downloaded files"""

    extra_paths: typing.List[pathlib.Path] = []
    """Additional read-only paths where to look for files"""

    integrity_check: FileIntegrityCheckMode = FileIntegrityCheckMode.ONCE
    """How to handle integrity checks"""

    on_missing_file: FileMissingAction = FileMissingAction.DOWNLOAD
    """What action to take on missing file"""

    on_unknown_file: FileUnknownAction = FileUnknownAction.FAIL
    """What action to take on request of file absent from registry"""

    download_max_attempts: int = 3
    """Maximum download attempts per repository for a file"""

    @property
    def search_paths(self) -> typing.List[pathlib.Path]:
        """List of file search paths."""
        return [self.download_path] + self.extra_paths

    @staticmethod
    def from_config() -> FileManagerOptions:
        """Build an option instance from FEW config manager entries."""


class FileDownloadMetadata(pydantic.BaseModel):
    """Metadata associated to a file cache entry on download"""

    repository: str
    """Name of the repository from which file was downloaded"""

    url: str
    """URL used for download"""


class FileCacheEntry(pydantic.BaseModel):
    name: str
    """File name"""

    path: pathlib.Path
    """Path to the file"""

    download_metadata: typing.Optional[FileDownloadMetadata]
    """Optional metadata associated to file download."""


class FileManager:
    """File manager: handles file download and integrity checks."""

    _registry: FileRegistry
    _options: FileManagerOptions
    _cache: typing.Dict[str, FileCacheEntry]

    def __init__(self):
        from few import cfg

        self._registry = FileRegistry.load_and_validate(cfg.file_registry_path)
        self._options = FileManagerOptions.from_config()
        self._cache = list()

    def _try_add_local_file_to_cache(
        self, file_name: str
    ) -> typing.Optional[FileCacheEntry]:
        """Try to locate file locally and add it to cache if success"""

        from ..utils.exceptions import FileNotInRegistry

        if (file_entry := self._registry.file_mapping.get(file_name, None)) is None:
            raise FileNotInRegistry(
                "File '{}' is not defined in file registry.".format(file_name)
            )

        for search_path in self._options.search_paths:
            file_path = search_path / file_name
            if not file_path.is_file():
                continue
            if self._options.integrity_check != FileIntegrityCheckMode.NEVER:
                try:
                    file_entry.check_file_matches_checksums(file_path)
                except exceptions.FileInvalidChecksum:
                    continue
            cache_entry = FileCacheEntry(name=file_name, path=file_path)
            # Add or replace cache entry
            self._cache[file_name] = cache_entry
            return cache_entry

        # File was not found (or found with invalid checksum)
        return None

    def _try_get_file_from_local_cache(
        self, file_name: str
    ) -> typing.Optional[FileCacheEntry]:
        """Get a file local cache entry if defined."""
        if file_name in self._cache:
            return self._cache[file_name]
        return None

    def _is_file_locally_present(self, file_name: str) -> bool:
        """Detect if a file is locally present."""
        if self._try_get_file_from_local_cache(file_name=file_name) is not None:
            return True
        if self._try_add_local_file_to_cache(file_name=file_name) is not None:
            return True
        return False

    def build_local_cache(self):
        """Add to cache all local files from the registry."""
        for file in self._registry.files:
            self._try_add_local_file_to_cache(file.name)

    def _download_file_from_repos(
        self, file_name: str, repository_names: typing.List[str]
    ) -> FileCacheEntry:
        """
        Try to download a file from list of repositories.
        """

        errors: typing.List[Exception] = []
        for attempt in range(self._options.download_max_attempts):
            for repository_name in repository_names:
                try:
                    return self._download_file_from_repo(file_name, repository_name)
                except exceptions.FileDownloadException as e:
                    new_exception = exceptions.FileDownloadException(
                        "Error while downloading {} from {} (attempt #{})".format(
                            file_name, repository_name, attempt
                        )
                    )
                    new_exception.__cause__ = e
                    errors.append(new_exception)
        raise exceptions.ExceptionGroup(
            "File {} could not be downloaded from repositories {}".format(
                file_name, repository_names
            ),
            errors,
        )

    def _download_file_from_repo(
        self, file_name: str, repository_name: str
    ) -> FileCacheEntry:
        """
        Try to download a file from a specific repository.
        """
        repo_entry = self._registry.get_repository(repository_name)
        assert repo_entry is not None

        url = repo_entry.build_url(file_name=file_name)
        output_path = self._options.download_path / file_name

        self._download_file(
            url=url,
            output_path=output_path,
            description="Downloading '{}'...".format(file_name),
        )  # Will raise a detailed FileDownloadException in case of errors

        if (
            (file_entry := self._registry.get_file(file_name)) is not None
            and self._options.integrity_check is not FileIntegrityCheckMode.NEVER
        ):
            try:
                file_entry.check_file_matches_checksums(output_path)
            except exceptions.FileInvalidChecksum as e:
                raise exceptions.FileDownloadIntegrityError(
                    "File '{}' downloaded from '{}' failed integrity checks."
                ) from e

        return FileCacheEntry(
            name=file_name,
            path=output_path,
            download_metadata=FileDownloadMetadata(repository=repository_name, url=url),
        )

    def _download_file(
        self,
        url: str,
        output_path: os.PathLike,
        description: typing.Optional[str] = None,
        chunk_size: int = 2**15,
    ):
        """
        Download file from url to given output path

        raise a FileDownloadException in case of error:
          FileDownloadNotFound:
        """
        import requests

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise exceptions.FileDownloadConnectionError() from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise exceptions.FileDownloadNotFound() from e
            raise exceptions.FileDownloadException() from e

        file_size = int(response.headers.get("content-length"))
        with open(output_path, "wb") as file:
            for chunk in rich.progress.track(
                response.iter_content(chunk_size=chunk_size),
                description=description
                if description is not None
                else "Downloading...",
                total=file_size / chunk_size,
            ):
                file.write(chunk)
