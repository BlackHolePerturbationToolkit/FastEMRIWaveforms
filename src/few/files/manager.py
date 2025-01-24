"""Class implementing the file manager."""

from __future__ import annotations

import enum
import os
import pathlib
import typing

import pydantic
import rich.progress

from .registry import FileRegistry, File
from ..utils import exceptions
from ..utils.config import CompleteConfigConsumer as Configuration


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

    SEARCH_LOCALLY = "search_locally"  # Tyy to locate file locally
    ATTEMPT_ALL_REPOS = "attempt_all_repos"  # Try to locate file in known repositories
    FAIL = "fail"  # Fail the program


class FileManagerOptions(pydantic.BaseModel):
    """Options controling the file manager behavior"""

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
    def from_config(few_cfg: Configuration) -> FileManagerOptions:
        """Build an option instance from FEW config manager entries."""
        import platformdirs

        from few import __version__

        download_path = (
            few_cfg.file_download_dir
            if few_cfg.file_download_dir is not None
            else platformdirs.user_data_path(
                appname="few", version="v{}".format(__version__), ensure_exists=True
            )
        )
        return FileManagerOptions(download_path=download_path)


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

    download_metadata: typing.Optional[FileDownloadMetadata] = None
    """Optional metadata associated to file download."""


class FileManager:
    """File manager: handles file download and integrity checks."""

    _registry: FileRegistry
    _options: FileManagerOptions
    _cache: typing.Dict[str, FileCacheEntry]

    def __init__(self, config: typing.Optional[Configuration] = None):
        if config is None:
            from few import globals

            config = globals.config

        self._registry = FileRegistry.load_and_validate(config.file_registry_path)
        self._options = FileManagerOptions.from_config(config)
        self._cache = dict()

    @property
    def storage_dir(self) -> pathlib.Path:
        """Directory in which files can be written."""
        return self._options.download_path

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
            self._cache.update({file_name: cache_entry})
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

    def _get_file_if_locally_present(
        self, file_name: str
    ) -> typing.Optional[FileCacheEntry]:
        """Detect if a file is locally present."""
        if (
            entry := self._try_get_file_from_local_cache(file_name=file_name)
        ) is not None:
            return entry
        if (
            entry := self._try_add_local_file_to_cache(file_name=file_name)
        ) is not None:
            return entry
        return None

    def build_local_cache(self):
        """Add to cache all local files from the registry."""
        for file in self._registry.files:
            self._try_add_local_file_to_cache(file.name)

    def _download_file_from_repos(
        self, file_name: str, repository_names: typing.Iterable[str]
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
        raise exceptions.FileDownloadException(
            "Failed downloading '{}' after {} attempts per repository.".format(
                file_name, self._options.download_max_attempts
            )
        ) from exceptions.ExceptionGroup(
            "File {} could not be downloaded from repositories {}".format(
                file_name, [name for name in repository_names]
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
                    "File '{}' downloaded from '{}' failed integrity checks.".format(
                        file_name, repository_name
                    )
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

    def _ensure_known_file(self, file_entry: File) -> FileCacheEntry:
        """Applies allowed strategies to obtain a file cache entry of declared file"""

        # 1 - Try to get file from existing cache
        if (
            cache_entry := self._try_get_file_from_local_cache(
                file_name=file_entry.name
            )
        ) is not None:
            if self._options.integrity_check == FileIntegrityCheckMode.ALWAYS:
                if file_entry.check_file_matches_checksums(cache_entry.path):
                    return cache_entry
            else:
                return cache_entry

        # 2 - Try to find file locally
        if (
            cache_entry := self._try_add_local_file_to_cache(file_name=file_entry.name)
        ) is not None:
            return cache_entry

        # 3 - Raise error if download is disabled
        if self._options.on_missing_file == FileMissingAction.FAIL:
            raise exceptions.FileNotFoundLocally(
                "File '{}' is not found locally and download is disabled.".format(
                    file_entry.name
                )
            )

        # 4 - Try to download file
        return self._download_file_from_repos(file_entry.name, file_entry.repositories)

    def _ensure_unknown_file(self, file_name: str) -> FileCacheEntry:
        """Applies allowed strategies to obtain a file cache entry of unknown file"""
        if self._options.on_unknown_file == FileUnknownAction.FAIL:
            raise exceptions.FileNotInRegistry(
                "File '{}' is not defined in registry.".format(file_name)
            )

        local_entry = self._get_file_if_locally_present(file_name=file_name)
        if local_entry is not None:
            return local_entry

        if self._options.on_unknown_file == FileUnknownAction.SEARCH_LOCALLY:
            raise exceptions.FileNotInRegistry(
                "File '{}' is not defined in registry and not found locally.".format(
                    file_name
                )
            )

        assert self._options.on_unknown_file == FileUnknownAction.ATTEMPT_ALL_REPOS
        if self._options.on_missing_file == FileMissingAction.FAIL:
            raise exceptions.FileNotFoundLocally(
                "File '{}' is not found locally and download is disabled.".format(
                    file_name
                )
            )

        try:
            return self._download_file_from_repos(
                file_name=file_name, repository_names=self._registry.repository_names
            )
        except exceptions.FileDownloadException as e:
            raise exceptions.FileNotInRegistry(
                "File '{}' is not defined in registry and not found locally or in known repositories.".format(
                    file_name
                )
            ) from e

    def _ensure_file(self, file_name: str) -> FileCacheEntry:
        """Try all allowed strategies to obtain a file cache entry."""
        # 1 - Check file is in registry
        file_entry = self._registry.get_file(file_name)

        if file_entry is not None:
            return self._ensure_known_file(file_entry)

        return self._ensure_unknown_file(file_name)

    def get_file(self, file_name: str) -> pathlib.Path:
        """Get file locally and return its path"""
        return self._ensure_file(file_name=file_name).path

    def prefetch_files_by_list(self, file_names: typing.Iterable[str]):
        """Ensure all files in the given list are present (or raise errors for missing files)"""
        errors = []
        for file_name in file_names:
            try:
                self._ensure_file(file_name=file_name)
            except exceptions.FileManagerException as e:
                errors.append(e)
        if errors:
            raise exceptions.FileManagerException(
                "Some files could not be prefetch."
            ) from exceptions.ExceptionGroup(
                "The following exceptions were raised while prefetching files.", errors
            )

    def prefetch_files_by_tag(self, tag: str):
        """Ensure all files matching a given tag are locally present (or raise errors)"""
        self.prefetch_files_by_list(
            (file.name for file in self._registry.get_files_by_tag(tag))
        )

    def prefetch_all_files(self):
        """Ensure all files defined in registry are locally present (or raise errors)"""
        self.prefetch_files_by_list((file.name for file in self._registry.files))
