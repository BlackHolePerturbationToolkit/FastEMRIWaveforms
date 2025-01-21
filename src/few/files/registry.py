"""Python representation of the file registry"""

import enum
import os
import pydantic

from ..utils.exceptions import InvalidInputFile
from typing import Dict, List, Optional

class Repository(pydantic.BaseModel):
    name: str
    url_pattern: str

class ChecksumAlgorithm(enum.Enum):
    SHA256 = "sha256"
    SHA1 = "sha1"
    MD5 = "md5"

class Checksum(pydantic.BaseModel):
    algorithm: ChecksumAlgorithm
    value: str

class File(pydantic.BaseModel):
    """Representation of a file entry in the registry."""

    name: str
    repositories: List[str]
    checksums: List[Checksum]
    tags: List[str] = []

class FileRegistry(pydantic.BaseModel):
    """Representation of the file registry."""

    repositories: List[Repository]
    files: List[File]

    @pydantic.model_validator(mode='after')
    def check_registry(self):
        self.check_repo_name_unique()
        self.check_file_name_unique()
        self.check_file_repositories_known()

    def check_repo_name_unique(self):
        if len(self.repositories) > len(set(self.repository_mapping.keys())):
            raise InvalidInputFile(
                "registry.yml is not a valid registry: duplicate {repositories.name} entries")

    def check_file_name_unique(self):
        if len(self.files) > len(set(self.file_mapping.keys())):
            raise InvalidInputFile(
                "registry.yml is not a valid registry: duplicate {files.name} entries")

    def check_file_repositories_known(self):
        known_repos: List[str] = [repo.name for repo in self.repositories]
        for file in self.files:
            for repo in file.repositories:
                if repo not in known_repos:
                    raise InvalidInputFile(
                        "registry.yml is not a valid registry: "
                        "file '{}' refers to undefined repo '{}'.".format(file.name, repo))

    @property
    def repository_mapping(self) -> Dict[str, Repository]:
        return {repo.name: repo for repo in self.repositories}

    @property
    def file_mapping(self) -> Dict[str, File]:
        return {file.name: file for file in self.files}


def load_validate_registry(registry_path: Optional[os.PathLike] = None) -> FileRegistry:
    import json
    import jsonschema
    import yaml
    import pathlib
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    if registry_path is None:
        registry_path = pathlib.Path(__file__).parent / 'registry.yml'

    with open(registry_path, 'r') as f:
        registry = yaml.load(f, Loader=Loader)

    with open(pathlib.Path(__file__).parent / 'registry.schema.json', 'r') as f:
        schema = json.load(f)

    try:
        jsonschema.validate(registry, schema)
    except jsonschema.SchemaError as e:
        raise InvalidInputFile("registry.schema.json is not a valid schema") from e
    except jsonschema.ValidationError as e:
        raise InvalidInputFile("registry.yml is not a valid registry.") from e

    return FileRegistry(**registry)
