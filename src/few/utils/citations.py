# Collection of citations for modules in FastEMRIWaveforms package

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
:code:`few.utils.citations`:

This module is used to collect citations for all modules in the package. This
module is then imported to add citations to module classes using their :code:`citation`
attribute.
"""

import abc
import enum
from pydantic import BaseModel
from typing import Sequence
from few.utils.exceptions import InvalidInputFile

class HyphenUnderscoreAliasModel(BaseModel):
    """Pydantic model were hyphen replace underscore in field names."""
    class Config:
        alias_generator = lambda field : field.replace("_", "-")
        extra = 'ignore'
        frozen = True

class Author(HyphenUnderscoreAliasModel):
    """Description of a reference author."""
    family_names: str
    given_names: str
    orcid: str | None = None
    affiliation: str | None = None
    email: str | None = None

class Publisher(HyphenUnderscoreAliasModel):
    """Description of a publisher."""
    name: str

class Identifier(HyphenUnderscoreAliasModel):
    """Description of an identifier."""
    type: str
    value: str
    description: str | None = None

class ReferenceABC(HyphenUnderscoreAliasModel, abc.ABC):
    """Abstract base class for references."""

    @abc.abstractmethod
    def to_bibtex(self) -> str:
        """Convert a reference object to a BibTeX string representation."""

class ArticleReference(ReferenceABC):
    """Description of an article."""

    abbreviation: str
    authors: list[Author]
    title: str
    journal: str
    year: int
    month: int | None = None
    issue: int | None = None
    publisher: Publisher | None = None
    pages: int | None = None
    start: int | None = None
    issn : str | None = None
    doi: str | None = None

    def to_bibtex(self) -> str:
        """Build the BibTeX representation of an article."""

        def format_line(key: str, value: str) -> str:
            return """  {:<10} = "{}" """.format(key, value)
        lines = []
        lines.append("@article{" + self.abbreviation)
        lines.append(format_line("author", " and ".join(["{}, {}".format(author.family_names, author.given_names) for author in self.authors])))
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
            lines.append(format_line("pages", str(self.start) if self.pages is None
                                     else "{}--{}".format(self.start, self.start+self.pages)))
        if self.issn is not None:
            lines.append(format_line("issn", str(self.issn)))
        if self.doi is not None:
            lines.append(format_line("doi", str(self.doi)))


        return ",\n".join(lines) + "\n}"

class SoftwareReference(ReferenceABC):
    """Description of a Software"""

    authors: list[Author]
    title: str

    license: str | None = None
    url: str | None = None
    repository: str | None = None
    identifiers: list[Identifier] | None = None
    year: int | None = None
    month: int | None = None
    version: str | None = None

    @property
    def doi(self) -> str | None:
        """Return the first DOI in identifiers if any"""
        for identifier in self.identifiers:
            if identifier.type == "doi":
              return identifier.value
        return None

    def to_bibtex(self) -> str:
        """Build the BibTeX representation of a software."""

        def format_line(key: str, value: str) -> str:
            return """  {:<10} = "{}" """.format(key, value)

        lines = []
        lines.append("@software{" + self.title)
        lines.append(format_line("author", " and ".join(["{}, {}".format(author.family_names, author.given_names) for author in self.authors])))
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


type Reference = ArticleReference | SoftwareReference

class REFERENCE(enum.Enum):
    FEW = "Chua:2020stf"
    LARGER_FEW = "Chua:2020stf"
    FEW_SOFTWARE = "FastEMRIWaveforms"
    ROMANNET = "Chua:2018woh"
    PN5 = "Fujita:2020zxe"
    KERR_SEPARATRIX = "Stein:2019buj"
    AAK1 = "Chua:2015mua"
    AAK2 = "Chua:2017ujo"
    AK = "Barack:2003fp"
    FD = "Speri:2023jte"

class CitationRegistry:
    __slots__ = ('registry')

    registry: dict[str, Reference]

    def __init__(self, **kwargs):
        self.registry = kwargs

    def get(self, key: str | REFERENCE) -> Reference:
        """Return a Reference object from its key."""
        return self.registry[key if isinstance(key, str) else key.value]

def build_citation_registry() -> CitationRegistry:
    """Read the package CITATION.cff and build the corresponding registry."""
    import jsonschema
    import pathlib
    import cffconvert

    from few import __file__ as _few_root_file

    citation_cff_path = pathlib.Path(_few_root_file).parent / 'CITATION.cff'
    with open(citation_cff_path, 'rt') as fid:
        cff = cffconvert.Citation(fid.read())

    try:
        cff.validate()
    except jsonschema.exceptions.ValidationError as e:
        raise InvalidInputFile("The file {} does not match its expected schema. Contact few developers.".format(citation_cff_path)) from e

    def to_reference(ref_dict) -> Reference:
        if ref_dict["type"] == "article":
            return ArticleReference(**ref_dict)
        if ref_dict["type"] == "software":
            return SoftwareReference(**ref_dict)

        raise InvalidInputFile("The file {} contains references whose type ({}) ".format(citation_cff_path, ref_dict["type"]) +
                               "is not supported", ref_dict)

    references = {
        ref["abbreviation"]: to_reference(ref) for ref in cff.cffobj["references"]
    }

    return CitationRegistry(**references, **{cff.cffobj["title"] : to_reference(cff.cffobj)})

CITATIONS_REGISTRY = build_citation_registry()

few_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.FEW).to_bibtex() + "\n"
larger_few_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.LARGER_FEW).to_bibtex() + "\n"
romannet_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.ROMANNET).to_bibtex() + "\n"
Pn5_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.PN5).to_bibtex() + "\n"
kerr_separatrix_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.KERR_SEPARATRIX).to_bibtex() + "\n"
AAK_citation_1 = "\n" + CITATIONS_REGISTRY.get(REFERENCE.AAK1).to_bibtex() + "\n"
AAK_citation_2 = "\n" + CITATIONS_REGISTRY.get(REFERENCE.AAK2).to_bibtex() + "\n"
AK_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.AK).to_bibtex() + "\n"
fd_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.FD).to_bibtex() + "\n"
few_software_citation = "\n" + CITATIONS_REGISTRY.get(REFERENCE.FEW_SOFTWARE).to_bibtex() + "\n"

COMMON_REFERENCES = [REFERENCE.FEW, REFERENCE.LARGER_FEW, REFERENCE.FEW_SOFTWARE]

class CitableClass(abc.ABC):
    """Base class for classes associated with specific citations."""

    @classmethod
    def citation(cls) -> str:
        """Return the module references as a printable BibTeX string."""
        references = cls.module_references() + COMMON_REFERENCES
        bibtex_entries = [CITATIONS_REGISTRY.get(key.value).to_bibtex() for key in references]
        return "\n".join(bibtex_entries)

    @classmethod
    @abc.abstractmethod
    def module_references(cls) -> list[REFERENCE]:
        """Method implemented by each class to define its list of references"""
        return []
