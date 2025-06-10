"""Class implementing the file manager."""

from __future__ import annotations

import enum
import os
import pathlib
import typing

import pydantic
import rich.progress

from ..utils import exceptions
from ..utils.config import Configuration
from .registry import File, FileRegistry


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

    storage_path: pathlib.Path
    """Base directory for FEW local storage."""

    download_path: pathlib.Path
    """Where to install downloaded files (absolute path, or relative to storage path)"""

    extra_paths: typing.List[pathlib.Path] = []
    """Additional read-only paths where to look for files (absolute or relative to storage path)"""

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
        return [self.download_path, self.storage_path] + [
            extra_path if extra_path.is_absolute() else self.storage_path / extra_path
            for extra_path in self.extra_paths
        ]

    @staticmethod
    def from_config(few_cfg: Configuration) -> FileManagerOptions:
        """Build an option instance from FEW config manager entries."""
        import platformdirs

        from few import __version__

        storage_path = (
            few_cfg.file_storage_path
            if few_cfg.file_storage_path is not None
            else platformdirs.user_data_path(
                appname="few", version="v{}".format(__version__), ensure_exists=True
            )
        )

        download_path = (
            few_cfg.file_download_path
            if few_cfg.file_download_path is not None
            else storage_path / "download"
        )
        download_path.mkdir(parents=True, exist_ok=True)

        on_missing_file = (
            FileMissingAction.DOWNLOAD
            if few_cfg.file_allow_download
            else FileMissingAction.FAIL
        )
        integrity_check = FileIntegrityCheckMode(few_cfg.file_integrity_check)
        extra_paths = (
            [] if few_cfg.file_extra_paths is None else few_cfg.file_extra_paths
        )

        return FileManagerOptions(
            storage_path=storage_path,
            download_path=download_path,
            on_missing_file=on_missing_file,
            integrity_check=integrity_check,
            extra_paths=extra_paths,
        )


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
            from few import get_config

            config = get_config()

        self._registry = FileRegistry.load_and_validate(config.file_registry_path)
        self._options = FileManagerOptions.from_config(config)
        self._cache = dict()

    @property
    def storage_dir(self) -> pathlib.Path:
        """Directory in which files can be read or written."""
        return self._options.storage_path

    @property
    def download_dir(self) -> pathlib.Path:
        """Directory in which downloaded files are written."""
        return self._options.download_path

    @property
    def options(self) -> FileManagerOptions:
        """Get options of this file manager"""
        return self._options

    @property
    def registry(self) -> FileRegistry:
        """Get the registry of this file manager"""
        return self._registry

    def _try_add_local_file_to_cache(
        self, file_name: str
    ) -> typing.Optional[FileCacheEntry]:
        """Try to locate file locally and add it to cache if success"""
        from ..utils.globals import get_logger

        logger = get_logger()

        logger.debug(f"Trying to locate file '{file_name}' locally.")
        from ..utils.exceptions import FileNotInRegistry

        if (file_entry := self._registry.file_mapping.get(file_name, None)) is None:
            raise FileNotInRegistry(
                "File '{}' is not defined in file registry.".format(file_name)
            )

        for search_path in self._options.search_paths:
            logger.debug(f" Trying in path '{search_path}'...")
            file_path = search_path / file_name
            if not file_path.is_file():
                logger.debug("  ... not found.")
                continue

            if self._options.integrity_check != FileIntegrityCheckMode.NEVER:
                try:
                    file_entry.check_file_matches_checksums(file_path)
                except exceptions.FileInvalidChecksum:
                    logger.debug("  ... found, but erroneous checksum.")
                    continue

            logger.debug("  ... found!")
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
        if not repository_names:
            raise exceptions.FileDownloadException(
                "No repository associated to file '{}'".format(file_name)
            )

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
        output_path = self.download_dir / file_name

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
            response = requests.get(url, stream=True, timeout=5)
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

    @staticmethod
    def _assert_not_disabled_entry(file_entry: File):
        from few import get_config

        disabled_tags = get_config().file_disabled_tags
        if disabled_tags is None:
            return

        for tag in file_entry.tags:
            if tag in disabled_tags:
                raise exceptions.FileManagerDisabledAccess(
                    "File %s is disabled by tag %s" % (file_entry.name, tag),
                    file_name=file_entry.name,
                    disabled_tag=tag,
                )

    def _ensure_known_file(self, file_entry: File) -> FileCacheEntry:
        """Applies allowed strategies to obtain a file cache entry of declared file"""

        # 1 - Check file is not disabled
        self._assert_not_disabled_entry(file_entry)

        # 2 - Try to get file from existing cache
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

        # 3 - Try to find file locally
        if (
            cache_entry := self._try_add_local_file_to_cache(file_name=file_entry.name)
        ) is not None:
            return cache_entry

        # 4 - Raise error if download is disabled
        if self._options.on_missing_file == FileMissingAction.FAIL:
            raise exceptions.FileNotFoundLocally(
                "File '{}' is not found locally and download is disabled.".format(
                    file_entry.name
                )
            )

        # 5 - Try to download file
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

    def get_tags(self) -> typing.List[str]:
        """Get the list of file tags"""
        return self._registry.get_tags()

    def try_get_local_file(
        self, file_name: str, use_cache: bool = False
    ) -> typing.Optional[pathlib.Path]:
        """Try to get file locally and return its path"""
        optional_entry = (
            self._try_get_file_from_local_cache(file_name=file_name)
            if use_cache
            else self._get_file_if_locally_present(file_name=file_name)
        )
        return optional_entry.path if optional_entry is not None else None

    def get_file(self, file_name: str) -> pathlib.Path:
        """Get file locally and return its path"""
        return self._ensure_file(file_name=file_name).path

    def try_get_file(self, file_name: str) -> typing.Optional[pathlib.Path]:
        """Get file and, if no strategy works, return None"""
        try:
            return self.get_file(file_name)
        except exceptions.FileManagerException:
            return None

    def prefetch_files_by_list(
        self, file_names: typing.Iterable[str], skip_disabled: bool = False
    ):
        """Ensure all files in the given list are present (or raise errors for missing files)"""
        from few import get_logger

        errors = []
        for file_name in file_names:
            try:
                self._ensure_file(file_name=file_name)
            except exceptions.FileManagerDisabledAccess as e:
                if skip_disabled:
                    get_logger().debug(
                        "Skipping fetching file '%s': tag '%s' is disabled",
                        file_name,
                        e.disabled_tag,
                    )
                else:
                    errors.append(e)

            except exceptions.FileManagerException as e:
                errors.append(e)
        if errors:
            raise exceptions.FileManagerException(
                "Some files could not be prefetch."
            ) from exceptions.ExceptionGroup(
                "The following exceptions were raised while prefetching files.", errors
            )

    def prefetch_files_by_tag(self, tag: str, skip_disabled: bool = False):
        """Ensure all files matching a given tag are locally present (or raise errors)"""
        self.prefetch_files_by_list(
            (file.name for file in self._registry.get_files_by_tag(tag)), skip_disabled
        )

    def prefetch_all_files(
        self,
        discarded_tags: typing.Optional[list[str]] = None,
        skip_disabled: bool = False,
    ):
        """Ensure all files defined in registry are locally present (or raise errors)"""

        def keep_file(file: File, discarded_tags: typing.Optional[list[str]] = None):
            if discarded_tags is None:
                return True
            for discarded_tag in discarded_tags:
                if discarded_tag in file.tags:
                    return False
            return True

        self.prefetch_files_by_list(
            (
                file.name
                for file in self._registry.files
                if keep_file(file, discarded_tags)
            ),
            skip_disabled,
        )

    def open(self, file: os.PathLike, mode="r", **kwargs):
        """Wrapper for open() built-in with automatic file download if needed."""
        file_path = pathlib.Path(file)

        # Do not change behavior of open if file is already defined or absolute
        if file_path.is_file() or file_path.is_absolute() or len(file_path.parts) > 1:
            return open(file_path, mode, **kwargs)

        if "r" in mode:  # File is to be read, we must first ensure its fetched
            return open(self.get_file(file), mode=mode, **kwargs)

        # File is to be written, open it in the context of storage_path
        return open(self.storage_dir / file_path, mode=mode, **kwargs)
