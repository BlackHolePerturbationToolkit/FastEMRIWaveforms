#!/usr/bin/env python3
"""Check that every contributor listed in CONTRIBUTORS.md appears in CITATION.cff.

Usage:
    python .github/scripts/check_contributors.py [--root PATH] [--strict] [--fix]

Exits with status 1 if any CONTRIBUTORS.md name is missing from the ``authors``
list of CITATION.cff, and prints a ready-to-paste YAML block for the missing
entries. ``--fix`` inserts those entries in alphabetical order instead, editing
the file as text so that comments, folded scalars and the existing quoting style
survive untouched; only given-names and family-names can be filled in, so
affiliation and ORCID still have to be added by hand. With ``--strict``,
CITATION.cff authors that are absent from
CONTRIBUTORS.md are also treated as errors (by default they are only reported
as warnings, since the citation metadata may legitimately credit people whose
contribution was not a code or data contribution).

Requires PyYAML (or ruamel.yaml).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path


def annotate(file: str, message: str) -> None:
    """Emit a GitHub Actions error annotation, so the failure shows in the PR.

    A no-op outside CI: the workflow-command syntax is only meaningful there.
    """
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    # Newlines would end the workflow command, so fold the message onto one line.
    flattened = " ".join(message.split())
    print(f"::error file={file}::{flattened}")

# Names that are spelled differently in the two files and refer to the same
# person. Keys and values are matched after normalisation, so only genuine
# spelling differences need an entry here.
ALIASES = {
    # CONTRIBUTORS.md v1.0.0 lists "Alvin Chu"; CITATION.cff has the full name.
    "alvin chu": "alvin j k chua",
}


def load_yaml(path: Path) -> dict:
    """Load a YAML file using whichever YAML library is available."""
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        pass
    try:
        from ruamel.yaml import YAML

        return YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    except ImportError:
        raise SystemExit(
            "error: this script requires PyYAML (pip install pyyaml) or ruamel.yaml"
        )


def normalize(name: str) -> str:
    """Fold a display name to a comparable form: ascii, lowercase, no punctuation."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(char for char in name if not unicodedata.combining(char))
    name = name.replace(".", " ").replace(",", " ")
    name = re.sub(r"[^\w\s-]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip().lower()


def match_key(name: str) -> tuple[str, str]:
    """Build a match key of (family name, first initial of given name).

    This lets "Zachary Nasipak" match "Zach Nasipak" while keeping distinct
    people with the same surname apart.
    """
    normalized = ALIASES.get(normalize(name), normalize(name))
    parts = normalized.split()
    if not parts:
        return ("", "")
    family = parts[-1]
    given_initial = parts[0][0] if len(parts) > 1 else ""
    return (family, given_initial)


# A version heading, e.g. "- FEW v2.1 (Kerr eccentric equatorial review update)"
VERSION_LINE = re.compile(r"^\s*[-*+]\s+FEW\b")
# A contributor entry: an indented bullet holding a name, optionally as a link.
CONTRIBUTOR_LINE = re.compile(r"^\s+[-*+]\s+(?P<entry>.+?)\s*$")
MARKDOWN_LINK = re.compile(r"\[(?P<text>[^\]]+)\]\([^)]*\)")


def parse_contributors(path: Path) -> list[str]:
    """Extract the unique contributor names from CONTRIBUTORS.md, in file order."""
    names: list[str] = []
    seen: set[tuple[str, str]] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if VERSION_LINE.match(line):
            continue
        match = CONTRIBUTOR_LINE.match(line)
        if not match:
            continue

        entry = match.group("entry")
        # Strip the trailing "</br>" that precedes a contribution description.
        entry = re.split(r"</?br\s*/?>", entry, maxsplit=1)[0].strip()
        link = MARKDOWN_LINK.search(entry)
        name = link.group("text").strip() if link else entry
        if not name or not normalize(name):
            continue

        key = match_key(name)
        if key in seen:
            continue
        seen.add(key)
        names.append(name)

    return names


def parse_citation_authors(path: Path) -> list[str]:
    """Extract the author display names from the top-level authors list."""
    data = load_yaml(path)
    authors = data.get("authors") or []
    names = []
    for author in authors:
        if "name" in author:  # entity-style author
            names.append(author["name"])
            continue
        given = author.get("given-names", "")
        family = author.get("family-names", "")
        names.append(f"{given} {family}".strip())
    return names


def split_name(name: str) -> tuple[str, str]:
    """Split a display name into (given names, family name)."""
    parts = name.split()
    if len(parts) < 2:
        return (name, "")
    return (" ".join(parts[:-1]), parts[-1])


def yaml_block(name: str) -> str:
    """Render a suggested CITATION.cff author entry for a missing contributor."""
    given, family = split_name(name)
    return f"  - given-names: {given}\n    family-names: {family}"


# The start of an author entry in the raw CITATION.cff text.
ENTRY_START = re.compile(r"^  - (?:given-names|family-names|name):")
FAMILY_NAMES = re.compile(r"^\s+family-names:\s*(?P<value>.+?)\s*$")


def insert_authors(path: Path, missing: list[str]) -> list[str]:
    """Splice entries for ``missing`` into the authors block of CITATION.cff.

    Edits the file as text rather than round-tripping the YAML, so comments,
    folded scalars and the file's mixed quoting style are left untouched. Only
    given-names and family-names are written; affiliation and ORCID cannot be
    inferred and must be filled in by hand.

    Returns the names that were inserted.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    try:
        block_start = next(
            i for i, line in enumerate(lines) if line.rstrip() == "authors:"
        )
    except StopIteration:
        raise SystemExit(f"error: no top-level 'authors:' key found in {path}")

    # The block ends at the next top-level key.
    block_end = len(lines)
    for i in range(block_start + 1, len(lines)):
        stripped = lines[i].rstrip("\n")
        if stripped and not stripped[0].isspace() and not stripped.startswith("#"):
            block_end = i
            break

    # Locate each existing entry: (line index of its first line, family name).
    entries: list[tuple[int, str]] = []
    for i in range(block_start + 1, block_end):
        if not ENTRY_START.match(lines[i]):
            continue
        family = ""
        for j in range(i, block_end):
            if j > i and ENTRY_START.match(lines[j]):
                break
            found = FAMILY_NAMES.match(lines[j])
            if found:
                family = found.group("value").strip("'\"")
                break
        entries.append((i, family))

    if not entries:
        raise SystemExit(f"error: no author entries parsed from {path}")

    # Find the alphabetical slot for each missing name.
    insertions: list[tuple[int, str, str]] = []
    for name in missing:
        _, family = split_name(name)
        sort_key = normalize(family)
        position = block_end
        for start, existing_family in entries:
            if normalize(existing_family) > sort_key:
                position = start
                break
        insertions.append((position, sort_key, yaml_block(name) + "\n"))

    # Apply top-down, offsetting by the lines already inserted. Sorting by
    # (position, sort_key) keeps several names sharing one slot in order.
    insertions.sort(key=lambda item: (item[0], item[1]))
    for offset, (position, _, block) in enumerate(insertions):
        lines.insert(position + offset, block)

    path.write_text("".join(lines), encoding="utf-8")
    return list(missing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root holding CONTRIBUTORS.md and CITATION.cff",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a CITATION.cff author is absent from CONTRIBUTORS.md",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "insert the missing names into CITATION.cff (names only; "
            "affiliation and ORCID must still be added by hand)"
        ),
    )
    args = parser.parse_args()

    contributors_path = args.root / "CONTRIBUTORS.md"
    citation_path = args.root / "CITATION.cff"
    for path in (contributors_path, citation_path):
        if not path.is_file():
            print(f"error: {path} not found")
            return 2

    contributors = parse_contributors(contributors_path)
    authors = parse_citation_authors(citation_path)

    if not contributors:
        print(f"error: no contributors parsed from {contributors_path}")
        return 2

    author_keys = {match_key(name) for name in authors}
    contributor_keys = {match_key(name) for name in contributors}

    missing = [name for name in contributors if match_key(name) not in author_keys]
    extra = [name for name in authors if match_key(name) not in contributor_keys]

    print(
        f"CONTRIBUTORS.md: {len(contributors)} unique contributors | "
        f"CITATION.cff: {len(authors)} authors"
    )

    if extra:
        label = "error" if args.strict else "warning"
        print(f"\n{label}: in CITATION.cff but not in CONTRIBUTORS.md:")
        for name in extra:
            print(f"  - {name}")

    if missing:
        if args.fix:
            inserted = insert_authors(citation_path, missing)
            print(f"\nadded {len(inserted)} author(s) to {citation_path.name}:")
            for name in inserted:
                print(f"  - {name}")
            print(
                "\nnote: only given-names and family-names were written. "
                "Add affiliation and ORCID by hand where known."
            )
            return 0 if not (extra and args.strict) else 1

        summary = ", ".join(missing)
        annotate(
            "CITATION.cff",
            f"{len(missing)} contributor(s) missing from the authors list: {summary}."
            " Run 'python .github/scripts/check_contributors.py --fix' locally, then add"
            " affiliation and ORCID by hand where known.",
        )
        print("\nerror: missing from the CITATION.cff authors list:")
        for name in missing:
            print(f"  - {name}")
        print("\nSuggested entries (add in alphabetical order by family name):")
        for name in missing:
            print(yaml_block(name))
        print(
            "\nTo fix, run this locally and commit the result:\n"
            "    python .github/scripts/check_contributors.py --fix\n"
            "Only the names are inserted; add affiliation and ORCID by hand"
            " where known."
        )
        return 1

    if extra and args.strict:
        return 1

    print("\nOK: every CONTRIBUTORS.md name is listed in CITATION.cff")
    return 0


if __name__ == "__main__":
    sys.exit(main())
