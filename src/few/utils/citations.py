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

class ArxivIdentifier(BaseModel):
    """Class representing an arXiv identifier"""
    reference: str
    primary_class: str | None = None

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
    identifiers: list[Identifier] | None = None

    @property
    def arxiv_preprint(self) -> ArxivIdentifier | None:
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
            if identifier.type != "other": continue
            if not identifier.value.lower().startswith('arxiv:'): continue
            data = identifier.value.lower().removeprefix('arxiv:')
            primary_class, reference = data.split('/', 1) if '/' in data else (None, data)
            return ArxivIdentifier(primary_class=primary_class, reference=reference)
        return None

    def to_bibtex(self) -> str:
        """Build the BibTeX representation of an article."""
        arxiv_id = self.arxiv_preprint

        line_format = """  {:<10} = \"{}\"""" if arxiv_id is None else """  {:<13} = \"{}\""""
        def format_line(key: str, value: str, format: str = line_format) -> str:
            return format.format(key, value)

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
            return """  {:<10} = \"{}\"""".format(key, value)

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
    LARGER_FEW = "Katz:2021yft"
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

COMMON_REFERENCES = [REFERENCE.FEW, REFERENCE.LARGER_FEW, REFERENCE.FEW_SOFTWARE]

class Citable:
    """Base class for classes associated with specific citations."""

    @classmethod
    def citation(cls) -> str:
        """Return the module references as a printable BibTeX string."""
        references = cls.module_references()
        bibtex_entries = [CITATIONS_REGISTRY.get(key.value).to_bibtex() for key in references]
        return "\n\n".join(bibtex_entries)

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Method implemented by each class to define its list of references"""
        return COMMON_REFERENCES


def cli_citation():
    """Add CLI utility to retrieve the citation of a given class."""
    import argparse
    import importlib
    import sys
    parser = argparse.ArgumentParser(
        prog='few_citations',
        description='Export the citations associated to a given module of the FastEMRIWaveforms package',
    )
    parser.add_argument('module')
    args = parser.parse_args(sys.argv[1:])

    few_class: str = args.module

    if not few_class.startswith('few.'):
        raise ValueError("The requested class must be part of the 'few' package (e.g. 'few.amplitude.AmpInterp2D').")

    module_path, class_name = few_class.rsplit('.', 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError("Could not import module '{}'.".format(module_path)) from e

    try:
        class_ref = getattr(module, class_name)
    except AttributeError as e:
        raise ImportError("Could not import class '{}' (not found in module '{}')".format(class_name, module_path)) from e

    if not issubclass(class_ref, Citable):
        print("Class '{}' ".format(few_class) + "does not implement specific references.\n"
              "However, since you are using the FastEMRIWaveform software, "
              "you may cite the following references: \n")
        print(Citable.citation())
        return

    print(class_ref.citation())
