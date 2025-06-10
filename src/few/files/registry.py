"""Python representation of the file registry"""

from __future__ import annotations

import enum
import os
from typing import Dict, Iterator, List, Optional

import pydantic

from ..utils.exceptions import ExceptionGroup, FileInvalidChecksum, InvalidInputFile


class Repository(pydantic.BaseModel):
    name: str
    url_pattern: str

    def build_url(self, file_name: str) -> str:
        """Build a url from its pattern"""
        return self.url_pattern % {"filename": file_name, "file_name": file_name}


class ChecksumAlgorithm(enum.Enum):
    SHA256 = "sha256"
    SHA1 = "sha1"
    MD5 = "md5"


class Checksum(pydantic.BaseModel):
    algorithm: ChecksumAlgorithm
    value: str

    @staticmethod
    def of_file(
        filepath: os.PathLike, algo: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    ) -> Checksum:
        """Compute file checksum"""
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
    """Representation of a file entry in the registry."""

    name: str
    repositories: List[str] = []
    checksums: List[Checksum]  # min size is 1 by registry.schema.json
    tags: List[str] = []

    def check_file_matches_checksums(self, filepath: os.PathLike):
        """Check whether a local file matches the entry checksums"""
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
    """Representation of the file registry."""

    repositories: List[Repository]
    files: List[File]

    @pydantic.model_validator(mode="after")
    def check_registry(self):
        self.check_repo_name_unique()
        self.check_file_name_unique()
        self.check_file_repositories_known()
        return self

    def check_repo_name_unique(self):
        if len(self.repositories) > len(set(self.repository_mapping.keys())):
            raise InvalidInputFile(
                "registry.yml is not a valid registry: duplicate {repositories.name} entries"
            )

    def check_file_name_unique(self):
        if len(self.files) > len(set(self.file_mapping.keys())):
            raise InvalidInputFile(
                "registry.yml is not a valid registry: duplicate {files.name} entries"
            )

    def check_file_repositories_known(self):
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
        return {repo.name: repo for repo in self.repositories}

    @property
    def file_mapping(self) -> Dict[str, File]:
        return {file.name: file for file in self.files}

    def get_repository(self, name: str) -> Optional[Repository]:
        """Get a repository by its name"""
        for repo in self.repositories:
            if repo.name == name:
                return repo
        return None

    @property
    def repository_names(self) -> Iterator[str]:
        for repository in self.repositories:
            yield repository.name

    def get_file(self, name: str) -> Optional[File]:
        """Get a file by its name"""
        for file in self.files:
            if file.name == name:
                return file
        return None

    def get_files_by_tag(self, tag: str) -> Iterator[File]:
        """Get files of given tag"""
        for file in self.files:
            if tag in file.tags:
                yield file

    def get_tags(self) -> List[str]:
        """Get the list of known file tags"""
        tags = set()
        for file in self.files:
            for tag in file.tags:
                tags.add(tag)
        return sorted(tags)

    @staticmethod
    def load_and_validate(registry_path: Optional[os.PathLike] = None) -> FileRegistry:
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
