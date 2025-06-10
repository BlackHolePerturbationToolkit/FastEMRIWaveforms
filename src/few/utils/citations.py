# Collection of citations for modules in FastEMRIWaveforms package

"""
:code:`few.utils.citations`:

This module is used to collect citations for all modules in the package. This
module is then imported to add citations to module classes using their :code:`citation`
attribute.
"""

import abc
import enum
from typing import List, Optional, Union

from pydantic import BaseModel

from few.utils.exceptions import InvalidInputFile


class HyphenUnderscoreAliasModel(BaseModel):
    """Pydantic model were hyphen replace underscore in field names."""

    class Config:
        def hyphen_replace(field: str) -> str:
            return field.replace("_", "-")

        alias_generator = hyphen_replace
        extra = "ignore"
        frozen = True


class Author(HyphenUnderscoreAliasModel):
    """Description of a reference author."""

    family_names: str
    given_names: str
    orcid: Optional[str] = None
    affiliation: Optional[str] = None
    email: Optional[str] = None


class Publisher(HyphenUnderscoreAliasModel):
    """Description of a publisher."""

    name: str


class Identifier(HyphenUnderscoreAliasModel):
    """Description of an identifier."""

    type: str
    value: str
    description: Optional[str] = None


class ReferenceABC(HyphenUnderscoreAliasModel, abc.ABC):
    """Abstract base class for references."""

    @abc.abstractmethod
    def to_bibtex(self) -> str:
        """Convert a reference object to a BibTeX string representation."""


class ArxivIdentifier(BaseModel):
    """Class representing an arXiv identifier"""

    reference: str
    primary_class: Optional[str] = None


class ArticleReference(ReferenceABC):
    """Description of an article."""

    abbreviation: str
    authors: List[Author]
    title: str
    journal: str
    year: int
    month: Optional[int] = None
    issue: Optional[int] = None
    publisher: Optional[Publisher] = None
    pages: Optional[int] = None
    start: Optional[int] = None
    issn: Optional[str] = None
    doi: Optional[str] = None
    identifiers: Optional[List[Identifier]] = None

    @property
    def arxiv_preprint(self) -> Optional[ArxivIdentifier]:
        """
        Detect an arXiv identifier if any.

        an arXiv identifier is:
        - an identifier of type "other"
        - which starts with "arxiv:" (case insensitive)
        - whose second part is either:
          - The arXiv reference (e.g. "arxiv:1912.07609")
          - The primary class followed by '/' and the reference (e.g. "arxiv:gr-qc/1912.07609")
        """
        for identifier in self.identifiers:
            if identifier.type != "other":
                continue
            if not identifier.value.lower().startswith("arxiv:"):
                continue
            data = identifier.value.lower().removeprefix("arxiv:")
            primary_class, reference = (
                data.split("/", 1) if "/" in data else (None, data)
            )
            return ArxivIdentifier(primary_class=primary_class, reference=reference)
        return None

    def to_bibtex(self) -> str:
        """Build the BibTeX representation of an article."""
        arxiv_id = self.arxiv_preprint

        line_format = (
            """  {:<10} = \"{}\"""" if arxiv_id is None else """  {:<13} = \"{}\""""
        )

        def format_line(key: str, value: str, format: str = line_format) -> str:
            return format.format(key, value)

        lines = []
        lines.append("@article{" + self.abbreviation)
        lines.append(
            format_line(
                "author",
                " and ".join(
                    [
                        "{}, {}".format(author.family_names, author.given_names)
                        for author in self.authors
                    ]
                ),
            )
        )
        lines.append(format_line("title", "{" + self.title + "}"))
        lines.append(format_line("journal", self.journal))
        lines.append(format_line("year", str(self.year)))
        if self.month is not None:
            lines.append(format_line("month", str(self.month)))
        if self.issue is not None:
            lines.append(format_line("number", str(self.issue)))
        if self.publisher is not None:
            lines.append(format_line("publisher", str(self.publisher.name)))
        if self.start is not None:
            lines.append(
                format_line(
                    "pages",
                    str(self.start)
                    if self.pages is None
                    else "{}--{}".format(self.start, self.start + self.pages),
                )
            )
        if self.issn is not None:
            lines.append(format_line("issn", str(self.issn)))
        if self.doi is not None:
            lines.append(format_line("doi", str(self.doi)))
        if arxiv_id is not None:
            lines.append(format_line("archivePrefix", "arXiv"))
            lines.append(format_line("eprint", arxiv_id.reference))
            if arxiv_id.primary_class is not None:
                lines.append(format_line("primaryClass", arxiv_id.primary_class))

        return ",\n".join(lines) + "\n}"


class SoftwareReference(ReferenceABC):
    """Description of a Software"""

    authors: list[Author]
    title: str

    license: Optional[str] = None
    url: Optional[str] = None
    repository: Optional[str] = None
    identifiers: Optional[List[Identifier]] = None
    year: Optional[int] = None
    month: Optional[int] = None
    version: Optional[str] = None

    @property
    def doi(self) -> Optional[str]:
        """Return the first DOI in identifiers if any"""
        for identifier in self.identifiers:
            if identifier.type == "doi":
                return identifier.value
        return None

    def to_bibtex(self) -> str:
        """Build the BibTeX representation of a software."""

        def format_line(key: str, value: str) -> str:
            return """  {:<10} = \"{}\"""".format(key, value)

        lines = []
        lines.append("@software{" + self.title)
        lines.append(
            format_line(
                "author",
                " and ".join(
                    [
                        "{}, {}".format(author.family_names, author.given_names)
                        for author in self.authors
                    ]
                ),
            )
        )
        lines.append(format_line("title", "{" + self.title + "}"))
        if self.license is not None:
            lines.append(format_line("license", self.license))
        if self.url is not None:
            lines.append(format_line("url", self.url))
        if self.repository is not None:
            lines.append(format_line("repository", self.repository))
        if self.year is not None:
            lines.append(format_line("year", str(self.year)))
        if self.month is not None:
            lines.append(format_line("month", str(self.month)))
        if self.version is not None:
            lines.append(format_line("version", self.version))
        if self.doi is not None:
            lines.append(format_line("doi", str(self.doi)))

        return ",\n".join(lines) + "\n}"


Reference = Union[ArticleReference, SoftwareReference]


class REFERENCE(enum.Enum):
    FEW = "Chua:2020stf"
    LARGER_FEW = "Katz:2021yft"
    FEW_SOFTWARE = "FastEMRIWaveforms"
    ROMANNET = "Chua:2018woh"
    PN5 = "Fujita:2020zxe"
    KERR_SEPARATRIX = "Stein:2019buj"
    AAK1 = "Chua:2015mua"
    AAK2 = "Chua:2017ujo"
    AK = "Barack:2003fp"
    FD = "Speri:2023jte"

    def __str__(self) -> str:
        return str(self.value)


class CitationRegistry:
    __slots__ = "registry"

    registry: dict[str, Reference]

    def __init__(self, **kwargs):
        self.registry = kwargs

    def get(self, key: Union[str, REFERENCE]) -> Reference:
        """Return a Reference object from its key."""
        return self.registry[key if isinstance(key, str) else key.value]


def build_citation_registry() -> CitationRegistry:
    """Read the package CITATION.cff and build the corresponding registry."""
    import json
    import pathlib

    import jsonschema
    import yaml

    from few import __file__ as _few_root_file
    from few import _is_editable as is_editable

    few_root = pathlib.Path(_few_root_file).parent
    cff_root = few_root.parent.parent if is_editable else few_root
    citation_cff_path = cff_root / "CITATION.cff"

    with open(citation_cff_path, "rt") as fid:
        cff = yaml.safe_load(fid)

    with open(pathlib.Path(__file__).parent / "cff_1_2_0.schema.json", "r") as f:
        cff_schema = json.load(f)

    try:
        jsonschema.validate(cff, cff_schema)
    except jsonschema.SchemaError as e:
        raise InvalidInputFile("cff_1_2_0.schema.json is not a valid schema.") from e
    except jsonschema.exceptions.ValidationError as e:
        raise InvalidInputFile(
            "The file {} does not match its expected schema. Contact few developers.".format(
                citation_cff_path
            )
        ) from e

    def to_reference(ref_dict) -> Reference:
        if ref_dict["type"] == "article":
            return ArticleReference(**ref_dict)
        if ref_dict["type"] == "software":
            return SoftwareReference(**ref_dict)

        raise InvalidInputFile(
            "The file {} contains references whose type ({}) ".format(
                citation_cff_path, ref_dict["type"]
            )
            + "is not supported",
            ref_dict,
        )

    references = {ref["abbreviation"]: to_reference(ref) for ref in cff["references"]}

    return CitationRegistry(**references, **{cff["title"]: to_reference(cff)})


COMMON_REFERENCES = [REFERENCE.FEW, REFERENCE.LARGER_FEW, REFERENCE.FEW_SOFTWARE]


class Citable:
    """Base class for classes associated with specific citations."""

    registry: Optional[CitationRegistry] = None

    @classmethod
    def citation(cls) -> str:
        """Return the module references as a printable BibTeX string."""
        references = cls.module_references()

        if Citable.registry is None:
            from few import get_logger

            get_logger().debug("Building the Citation Registry from CITATION.cff")
            Citable.registry = build_citation_registry()

        bibtex_entries = [
            Citable.registry.get(str(key)).to_bibtex() for key in references
        ]
        return "\n\n".join(bibtex_entries)

    @classmethod
    def module_references(cls) -> List[Union[REFERENCE, str]]:
        """Method implemented by each class to define its list of references"""
        return COMMON_REFERENCES
