"""Python representation of the file registry"""

import enum
import pydantic

from ..utils.exceptions import InvalidInputFile
from typing import List

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
    def check_file_repositories_known(self):
        known_repos: List[str] = [repo.name for repo in self.repositories]
        for file in self.files:
            for repo in file.repositories:
                if repo not in known_repos:
                    raise InvalidInputFile(
                        "registry.yml is not a valid registry: "
                        "file '{}' refers to undefined repo '{}'.".format(file.name, repo))


def load_validate_registry() -> FileRegistry:
    import json
    import jsonschema
    import yaml
    import pathlib
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(pathlib.Path(__file__).parent / 'registry.yml', 'r') as f:
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
