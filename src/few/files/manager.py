"""The FEW file manager: find, verify and download the data files listed in the registry.

:class:`FileManager` is the entry point (global instance: :func:`few.get_file_manager`).
:meth:`FileManager.get_file` resolves a file name in this order: the in-memory cache, the
local search paths (download directory, storage directory, extra paths) with a checksum
check, then a download from the file's registry repositories (unless downloads are
disabled). The behaviour is set by :class:`FileManagerOptions`, built from the FEW
configuration (``file_storage_path``, ``file_download_path``, ``file_extra_paths``,
``file_integrity_check``, ``file_allow_download``, ``file_registry_path``,
``file_disabled_tags``).
"""

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
    """Mode for file-integrity (checksum) verification.

    ``ALWAYS``: check at each request; ``ONCE`` (default): check when a file is first found
    and keep the result in the cache; ``NEVER``: do not check, not even after a download.
    """

    ALWAYS = "always"  # Check file integrity at each request
    ONCE = "once"  # Check file integrity once and keep result in cache
    NEVER = "never"  # Do not check file integrity


class FileMissingAction(enum.Enum):
    """Action to take when a file is not found locally (or fails its integrity check).

    ``DOWNLOAD`` (default): download it from one of its repositories; ``FAIL``: raise
    :class:`~few.utils.exceptions.FileNotFoundLocally` (offline mode, set by
    ``file_allow_download = False``).
    """

    DOWNLOAD = "download"  # Download the file from one of its repos
    FAIL = "fail"  # Fail the program (to be used in offline mode)


class FileUnknownAction(enum.Enum):
    """Action to take when a requested file is not in the file registry.

    ``SEARCH_LOCALLY``: look in the search paths only; ``ATTEMPT_ALL_REPOS``: also try every
    registry repository; ``FAIL`` (default): raise
    :class:`~few.utils.exceptions.FileNotInRegistry`. No configuration option sets it, so
    :meth:`FileManagerOptions.from_config` keeps the default.
    """

    SEARCH_LOCALLY = "search_locally"  # Tyy to locate file locally
    ATTEMPT_ALL_REPOS = "attempt_all_repos"  # Try to locate file in known repositories
    FAIL = "fail"  # Fail the program


class FileManagerOptions(pydantic.BaseModel):
    """Options controlling the file manager behaviour (a pydantic model).

    Normally built by :meth:`from_config`; the fields are documented below.
    """

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
        """Search order: the download path, the storage path, then the extra paths.

        Relative extra paths are taken relative to the storage path.
        """
        return [self.download_path, self.storage_path] + [
            extra_path if extra_path.is_absolute() else self.storage_path / extra_path
            for extra_path in self.extra_paths
        ]

    @staticmethod
    def from_config(few_cfg: Configuration) -> FileManagerOptions:
        """Build the options from the FEW configuration.

        Defaults: storage path ``platformdirs.user_data_path("few", "v<version>")`` (created
        if missing); download path ``<storage>/download`` (always created);
        ``on_missing_file`` ``DOWNLOAD`` unless ``file_allow_download`` is false;
        ``on_unknown_file`` and ``download_max_attempts`` keep their class defaults.

        Args:
            few_cfg: The FEW configuration (:func:`few.get_config`).

        Returns:
            FileManagerOptions: The options.
        """
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
    """A located file: its name, path and, if it was downloaded, the download metadata."""

    name: str
    """File name"""

    path: pathlib.Path
    """Path to the file"""

    download_metadata: typing.Optional[FileDownloadMetadata] = None
    """Optional metadata associated to file download."""


class FileManager:
    """File manager: locates FEW data files, checks their integrity and downloads them.

    Loads and validates the registry (``config.file_registry_path``, default the packaged
    ``registry.yml``) and builds :class:`FileManagerOptions` from the configuration. Located
    files are cached in memory by name. FEW's modules obtain the global instance with
    :func:`few.get_file_manager` (created at the first FEW globals initialisation).

    Args:
        config: FEW configuration. Default ``None`` uses :func:`few.get_config`.

    Raises:
        few.utils.exceptions.InvalidInputFile: If the registry file is invalid.

    Example:
        .. code-block:: python

            from few import get_file_manager

            fm = get_file_manager()
            path = fm.get_file("ZNAmps_l10_m10_n55_DS2Outer.h5")  # local path, downloaded if needed
            fm.prefetch_files_by_tag("testfile")
    """

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
        """Directory in which files can be read or written (the storage path)."""
        return self._options.storage_path

    @property
    def download_dir(self) -> pathlib.Path:
        """Directory in which downloaded files are written (the download path)."""
        return self._options.download_path

    @property
    def options(self) -> FileManagerOptions:
        """The options of this file manager."""
        return self._options

    @property
    def registry(self) -> FileRegistry:
        """The file registry of this file manager."""
        return self._registry

    def _try_add_local_file_to_cache(
        self, file_name: str
    ) -> typing.Optional[FileCacheEntry]:
        """Look for a registry file in the search paths; cache and return the first good copy.

        A copy whose checksum does not match is skipped (unless the integrity check mode is
        ``NEVER``).

        Returns:
            FileCacheEntry or None: The new cache entry, or ``None`` if no valid copy exists.

        Raises:
            few.utils.exceptions.FileNotInRegistry: If ``file_name`` is not in the registry
                (also when called for an unknown file through
                :meth:`_get_file_if_locally_present`).
        """
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
        """Return the in-memory cache entry of ``file_name``, or ``None``."""
        if file_name in self._cache:
            return self._cache[file_name]
        return None

    def _get_file_if_locally_present(
        self, file_name: str
    ) -> typing.Optional[FileCacheEntry]:
        """Return the cache entry of a locally present file (cache, then search paths), or ``None``.

        Raises:
            few.utils.exceptions.FileNotInRegistry: If the file is neither cached nor in the
                registry (see :meth:`_try_add_local_file_to_cache`).
        """
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
        """Search the local paths for every registry file and cache those found (no download)."""
        for file in self._registry.files:
            self._try_add_local_file_to_cache(file.name)

    def _download_file_from_repos(
        self, file_name: str, repository_names: typing.Iterable[str]
    ) -> FileCacheEntry:
        """Download a file, trying each repository in turn, ``download_max_attempts`` rounds.

        Returns:
            FileCacheEntry: The downloaded file (not added to the cache).

        Raises:
            few.utils.exceptions.FileDownloadException: If ``repository_names`` is empty or
                every attempt failed (the individual errors are chained as an exception
                group).
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
        """Download a file from one repository into the download directory and verify it.

        Returns:
            FileCacheEntry: The downloaded file with its download metadata.

        Raises:
            few.utils.exceptions.FileDownloadException: On a download error.
            few.utils.exceptions.FileDownloadIntegrityError: If a registry file fails its
                checksum (unless the integrity check mode is ``NEVER``).
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
        """Stream ``url`` to ``output_path`` with a progress bar (``requests``, 5 s timeout).

        Args:
            url: Source URL.
            output_path: Destination file (overwritten).
            description: Progress-bar label. Default ``None`` means ``"Downloading..."``.
            chunk_size: Stream chunk size in bytes. Default ``2**15``.

        Raises:
            few.utils.exceptions.FileDownloadConnectionError: On a connection error.
            few.utils.exceptions.FileDownloadNotFound: On an HTTP 404.
            few.utils.exceptions.FileDownloadException: On another HTTP error.
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
        """Refuse an entry with a disabled tag (``file_disabled_tags`` of the global config).

        Raises:
            few.utils.exceptions.FileManagerDisabledAccess: If one of the entry's tags is
                disabled.
        """
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
        """Obtain a registry file: tag check, cache, search paths, then download if allowed.

        With ``ALWAYS`` integrity checks a cached entry is re-verified; since
        :meth:`~few.files.registry.File.check_file_matches_checksums` returns ``None`` on
        success, the lookup then continues with the search paths (a second check), and a
        checksum failure of the cached copy raises
        :class:`~few.utils.exceptions.FileInvalidChecksum`.

        Raises:
            few.utils.exceptions.FileManagerDisabledAccess: If a tag of the file is disabled.
            few.utils.exceptions.FileNotFoundLocally: If the file is not found locally and
                downloads are disabled.
            few.utils.exceptions.FileDownloadException: If the download fails.
        """

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
        """Obtain a file absent from the registry, following ``on_unknown_file``.

        With ``FAIL`` (the default) this raises
        :class:`~few.utils.exceptions.FileNotInRegistry`. The local lookup of the other two
        modes goes through :meth:`_try_add_local_file_to_cache`, which itself raises
        ``FileNotInRegistry`` for an unregistered name unless it is already cached.
        """
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
        """Obtain a file's cache entry (registry file or unknown file)."""
        # 1 - Check file is in registry
        file_entry = self._registry.get_file(file_name)

        if file_entry is not None:
            return self._ensure_known_file(file_entry)

        return self._ensure_unknown_file(file_name)

    def get_tags(self) -> typing.List[str]:
        """Return the sorted list of tags used in the registry."""
        return self._registry.get_tags()

    def try_get_local_file(
        self, file_name: str, use_cache: bool = False
    ) -> typing.Optional[pathlib.Path]:
        """Return the local path of a file if present, without downloading.

        Args:
            file_name: File name.
            use_cache: If ``True``, consult only the in-memory cache; if ``False`` (default),
                also search the local paths (and cache a hit).

        Returns:
            pathlib.Path or None: The path, or ``None`` if not found.

        Raises:
            few.utils.exceptions.FileNotInRegistry: With ``use_cache=False``, for a name that
                is neither cached nor in the registry.
        """
        optional_entry = (
            self._try_get_file_from_local_cache(file_name=file_name)
            if use_cache
            else self._get_file_if_locally_present(file_name=file_name)
        )
        return optional_entry.path if optional_entry is not None else None

    def get_file(self, file_name: str) -> pathlib.Path:
        """Return the local path of a data file, downloading it if needed and allowed.

        This is how FEW's models load their data (e.g. ``KerrEccEqFluxData.h5``,
        ``ZNAmps_l10_m10_n55_DS2Outer.h5``). With the default ``ONCE`` integrity mode the
        checksum is computed the first time a file is located in this process.

        Args:
            file_name: File name as listed in the registry.

        Returns:
            pathlib.Path: Path of the verified local file.

        Raises:
            few.utils.exceptions.FileManagerException: If the file cannot be obtained
                (subclasses ``FileNotInRegistry``, ``FileNotFoundLocally``,
                ``FileManagerDisabledAccess``, ``FileDownloadException``, ...).
        """
        return self._ensure_file(file_name=file_name).path

    def try_get_file(self, file_name: str) -> typing.Optional[pathlib.Path]:
        """Like :meth:`get_file`, but return ``None`` instead of raising a file-manager error."""
        try:
            return self.get_file(file_name)
        except exceptions.FileManagerException:
            return None

    def prefetch_files_by_list(
        self, file_names: typing.Iterable[str], skip_disabled: bool = False
    ):
        """Make sure every listed file is present locally, downloading as needed.

        All files are attempted; the failures are raised together at the end.

        Args:
            file_names: File names.
            skip_disabled: If ``True``, skip (with a debug log) files refused because of a
                disabled tag instead of counting them as failures. Default ``False``.

        Raises:
            few.utils.exceptions.FileManagerException: If any file could not be obtained
                (the individual errors are chained as an exception group).
        """
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
        """Make sure every registry file tagged ``tag`` is present locally.

        See :meth:`prefetch_files_by_list`.

        Args:
            tag: Registry tag (e.g. ``"testfile"``; :meth:`get_tags` lists them).
            skip_disabled: Skip files refused because of a disabled tag. Default ``False``.
        """
        self.prefetch_files_by_list(
            (file.name for file in self._registry.get_files_by_tag(tag)), skip_disabled
        )

    def prefetch_all_files(
        self,
        discarded_tags: typing.Optional[list[str]] = None,
        skip_disabled: bool = False,
    ):
        """Make sure every registry file is present (see :meth:`prefetch_files_by_list`).

        Args:
            discarded_tags: Skip files carrying any of these tags (``few_files fetch`` passes
                ``["deprecated"]``). Default ``None`` keeps all files.
            skip_disabled: Skip files refused because of a disabled tag. Default ``False``.
        """

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
        """Wrapper for the built-in ``open`` that fetches registry files on demand.

        An existing file, an absolute path or a path with a directory part is opened as is.
        A bare name opened for reading is resolved with :meth:`get_file`; for writing it is
        opened in the storage directory.

        Args:
            file: File name or path.
            mode (str): ``open`` mode. Default ``"r"``.
            **kwargs: Forwarded to ``open``.

        Returns:
            typing.IO: The open file object.
        """
        file_path = pathlib.Path(file)

        # Do not change behavior of open if file is already defined or absolute
        if file_path.is_file() or file_path.is_absolute() or len(file_path.parts) > 1:
            return open(file_path, mode, **kwargs)

        if "r" in mode:  # File is to be read, we must first ensure its fetched
            return open(self.get_file(file), mode=mode, **kwargs)

        # File is to be written, open it in the context of storage_path
        return open(self.storage_dir / file_path, mode=mode, **kwargs)
