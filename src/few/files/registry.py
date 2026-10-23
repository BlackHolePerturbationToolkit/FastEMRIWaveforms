"""Python representation of the file registry (pydantic models of ``registry.yml``).

The registry lists the data repositories (name and URL pattern) and, for every data file,
its repositories, checksums and tags. :meth:`FileRegistry.load_and_validate` reads the YAML
file, validates it against ``registry.schema.json`` and checks its internal consistency.
"""

from __future__ import annotations

import enum
import os
from typing import Dict, Iterator, List, Optional

import pydantic

from ..utils.exceptions import ExceptionGroup, FileInvalidChecksum, InvalidInputFile


class Repository(pydantic.BaseModel):
    """A data repository: ``name`` and ``url_pattern``.

    ``url_pattern`` is a %-format string with a ``%(filename)s`` (or ``%(file_name)s``)
    placeholder, e.g. ``"https://zenodo.org/records/15624459/files/%(filename)s"``.
    """

    name: str
    url_pattern: str

    def build_url(self, file_name: str) -> str:
        """Return the download URL of ``file_name`` in this repository."""
        return self.url_pattern % {"filename": file_name, "file_name": file_name}


class ChecksumAlgorithm(enum.Enum):
    """Supported checksum algorithms: ``sha256``, ``sha1``, ``md5``."""

    SHA256 = "sha256"
    SHA1 = "sha1"
    MD5 = "md5"


class Checksum(pydantic.BaseModel):
    """A checksum: ``algorithm`` and hexadecimal ``value``."""

    algorithm: ChecksumAlgorithm
    value: str

    @staticmethod
    def of_file(
        filepath: os.PathLike, algo: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    ) -> Checksum:
        """Compute the checksum of a file (read in 64 KiB blocks).

        Args:
            filepath: File to hash.
            algo: Algorithm. Default ``ChecksumAlgorithm.SHA256``.

        Returns:
            Checksum: The checksum of the file.
        """
        import hashlib

        HASHERS = {
            ChecksumAlgorithm.MD5: hashlib.md5,
            ChecksumAlgorithm.SHA1: hashlib.sha1,
            ChecksumAlgorithm.SHA256: hashlib.sha256,
        }
        BLOCK_SIZE = 2**16  # Read file by blocks of 16kB
        hasher = HASHERS[algo]()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(BLOCK_SIZE), b""):
                hasher.update(byte_block)
        return Checksum(algorithm=algo, value=hasher.hexdigest())


class File(pydantic.BaseModel):
    """Representation of a file entry in the registry.

    Fields: ``name``, ``repositories`` (names of the repositories holding it), ``checksums``
    (at least one) and ``tags`` (e.g. ``testfile``, ``deprecated``, ``kerr``,
    ``high_memory``).
    """

    name: str
    repositories: List[str] = []
    checksums: List[Checksum]  # min size is 1 by registry.schema.json
    tags: List[str] = []

    def check_file_matches_checksums(self, filepath: os.PathLike):
        """Check a local file against every checksum of the entry.

        Args:
            filepath: Local file.

        Returns:
            None: Returns nothing on success.

        Raises:
            few.utils.exceptions.FileInvalidChecksum: If one checksum differs.
            TypeError: If several checksums differ (the code then builds its
                ``ExceptionGroup`` without the message argument). Every entry of the packaged
                registry has a single checksum.
        """
        errors = []
        for entry_checksum in self.checksums:
            file_checksum = Checksum.of_file(
                filepath=filepath, algo=entry_checksum.algorithm
            )
            if entry_checksum.value != file_checksum.value:
                errors.append(
                    FileInvalidChecksum(
                        "File {} has invalid {} checksum (expected: {}, got {})".format(
                            filepath,
                            entry_checksum.algorithm,
                            entry_checksum.value,
                            file_checksum.value,
                        )
                    )
                )
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise FileInvalidChecksum(
                "File has failed multiple integrity checks"
            ) from ExceptionGroup(errors)


class FileRegistry(pydantic.BaseModel):
    """Representation of the file registry: ``repositories`` and ``files``.

    Built by :meth:`load_and_validate`; the pydantic validator :meth:`check_registry` runs
    the consistency checks on construction.

    Example:
        .. code-block:: python

            from few.files import FileRegistry

            registry = FileRegistry.load_and_validate()  # the packaged registry.yml
            registry.get_tags()
            entry = registry.get_file("KerrEccEqFluxData.h5")
    """

    repositories: List[Repository]
    files: List[File]

    @pydantic.model_validator(mode="after")
    def check_registry(self):
        """Run the three consistency checks (pydantic after-validator)."""
        self.check_repo_name_unique()
        self.check_file_name_unique()
        self.check_file_repositories_known()
        return self

    def check_repo_name_unique(self):
        """Raise :class:`~few.utils.exceptions.InvalidInputFile` on duplicate repository names."""
        if len(self.repositories) > len(set(self.repository_mapping.keys())):
            raise InvalidInputFile(
                "registry.yml is not a valid registry: duplicate {repositories.name} entries"
            )

    def check_file_name_unique(self):
        """Raise :class:`~few.utils.exceptions.InvalidInputFile` on duplicate file names."""
        if len(self.files) > len(set(self.file_mapping.keys())):
            raise InvalidInputFile(
                "registry.yml is not a valid registry: duplicate {files.name} entries"
            )

    def check_file_repositories_known(self):
        """Raise :class:`~few.utils.exceptions.InvalidInputFile` on an unknown file repository."""
        known_repos: List[str] = [repo.name for repo in self.repositories]
        for file in self.files:
            for repo in file.repositories:
                if repo not in known_repos:
                    raise InvalidInputFile(
                        "registry.yml is not a valid registry: "
                        "file '{}' refers to undefined repo '{}'.".format(
                            file.name, repo
                        )
                    )

    @property
    def repository_mapping(self) -> Dict[str, Repository]:
        """Repository name -> :class:`Repository` (rebuilt at each access)."""
        return {repo.name: repo for repo in self.repositories}

    @property
    def file_mapping(self) -> Dict[str, File]:
        """File name -> :class:`File` (rebuilt at each access)."""
        return {file.name: file for file in self.files}

    def get_repository(self, name: str) -> Optional[Repository]:
        """Return the repository called ``name``, or ``None``."""
        for repo in self.repositories:
            if repo.name == name:
                return repo
        return None

    @property
    def repository_names(self) -> Iterator[str]:
        """Generator over the repository names, in registry order."""
        for repository in self.repositories:
            yield repository.name

    def get_file(self, name: str) -> Optional[File]:
        """Return the file entry called ``name``, or ``None``."""
        for file in self.files:
            if file.name == name:
                return file
        return None

    def get_files_by_tag(self, tag: str) -> Iterator[File]:
        """Yield the file entries carrying ``tag``."""
        for file in self.files:
            if tag in file.tags:
                yield file

    def get_tags(self) -> List[str]:
        """Return the sorted list of tags used by the file entries."""
        tags = set()
        for file in self.files:
            for tag in file.tags:
                tags.add(tag)
        return sorted(tags)

    @staticmethod
    def load_and_validate(registry_path: Optional[os.PathLike] = None) -> FileRegistry:
        """Read a registry YAML file, validate it against the JSON schema and build it.

        Args:
            registry_path: Registry file. Default ``None`` means the ``registry.yml``
                packaged next to this module.

        Returns:
            FileRegistry: The validated registry.

        Raises:
            few.utils.exceptions.InvalidInputFile: If the file does not satisfy
                ``registry.schema.json`` (or the schema itself is invalid), or fails the
                consistency checks.
        """
        import json
        import pathlib

        import jsonschema
        import yaml

        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        if registry_path is None:
            registry_path = pathlib.Path(__file__).parent / "registry.yml"

        with open(registry_path, "r") as f:
            registry = yaml.load(f, Loader=Loader)

        with open(pathlib.Path(__file__).parent / "registry.schema.json", "r") as f:
            schema = json.load(f)

        try:
            jsonschema.validate(registry, schema)
        except jsonschema.SchemaError as e:
            raise InvalidInputFile("registry.schema.json is not a valid schema.") from e
        except jsonschema.ValidationError as e:
            raise InvalidInputFile(
                "{} is not a valid registry.".format(registry_path)
            ) from e

        return FileRegistry(**registry)
