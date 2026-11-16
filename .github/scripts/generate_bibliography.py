#!/usr/bin/env python3
"""Generate PAPERS.bib from CITATION.cff.

CITATION.cff is the authoritative source of FEW citation metadata.
PAPERS.bib is a generated convenience file and must not be edited
manually.
"""

from pathlib import Path

from few.utils.citations import all_citations


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPOSITORY_ROOT / "../PAPERS.bib"

HEADER = """\
% ---------------------------------------------------------------------------
% This file is generated automatically from CITATION.cff.
%
% Do not edit PAPERS.bib manually.
%
% To update this file, edit CITATION.cff and run:
%
%     python scripts/generate_bibliography.py
%
% CITATION.cff is the authoritative source of FEW citation metadata.
% ---------------------------------------------------------------------------

"""


def main() -> None:
    """Generate PAPERS.bib from the FEW citation registry."""
    bibliography = all_citations().strip()

    OUTPUT_FILE.write_text(
        HEADER + bibliography + "\n",
        encoding="utf-8",
    )

    print(f"Generated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()