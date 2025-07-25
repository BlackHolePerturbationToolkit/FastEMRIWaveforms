# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import pathlib
import typing

import few

if typing.TYPE_CHECKING:
    import sphinx.application

# -- Project information -----------------------------------------------------

project = "few: Fast EMRI Waveforms"
copyright = "2020, Michael Katz, Alvin Chua, Niels Warburton"
author = "Michael Katz, Alvin Chua, Niels Warburton"

# The full version, including alpha/beta/rc tags
release = few.__version__


# -- Copy example notebook --
root_dir = pathlib.Path(__file__).parent.parent.parent
src_dir = root_dir / "examples"
trg_dir = root_dir / "docs" / "source" / "tutorial"
trg_dir.mkdir(parents=True, exist_ok=True)

for example in (
    "Trajectory_tutorial",
    "Amplitude_tutorial",
    "modeselect",
    "modesummation",
    "cubicspline",
    "swsh",
    "utility",
    "waveform",
):
    filename = example + ".ipynb"
    if not (trg_dir / filename).is_file():
        os.symlink(src_dir / filename, trg_dir / filename)
try:
    os.symlink(src_dir / "files", trg_dir / "files", target_is_directory=True)
except FileExistsError:
    pass

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx_tippy",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst", ".md"]
templates_path = []


# -- Extensions configuration ---------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"

doctest_global_setup = """
import few
"""
doctest_global_cleanup = """
few.utils.globals.reset(True)
"""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11", None),
}

linkcheck_ignore = [
    r"https://dx.doi.org/",
    r"https://hpc.pages.cnes.fr/.*",
    r"https://jupyterhub.cnes.fr/.*",
    r"https://bhptoolkit.org/FastEMRIWaveforms/.*",
]

myst_enable_extensions = [
    "colon_fence",
]

myst_heading_anchors = 2
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "vscode": None,
}


nbsphinx_allow_errors = True
nbsphinx_execute = "auto"
nbsphinx_kernel_name = "python3"
nbsphinx_requirejs_path = ""

if os.getenv("READTHEDOCS", None) is not None:
    nbsphinx_execute = "never"
    nbsphinx_allow_errors = False

tippy_add_class = "has-tippy"
tippy_js = (
    "https://unpkg.com/@popperjs/core@2",
    "https://unpkg.com/tippy.js@6",
    "https://unpkg.com/requirejs@2",
)


def skip(app, what, name, obj, would_skip, options):
    if name == "__call__":
        return False
    return would_skip


reftarget_aliases = {"CITATION.cff": "CITATION"}

with open("../../PAPERS.bib", "r") as f:
    papers_bib = f.read()

JINJA_ENV_PER_DOC = {
    "user/install": {
        "context": {
            "few_short_version": ".".join(str(v) for v in few.__version_tuple__[:3]),
        },
    },
    "readme": {
        "substitutions": {
            "(PAPERS.bib)": "(PAPERS.md)",
            "https://fastemriwaveforms.readthedocs.io/en/latest/user/install.html": "user/install.md",
            " Please see the [documentation](https://fastemriwaveforms.readthedocs.io/en/latest) for further information on these modules.": "",
        }
    },
    "PAPERS": {
        "context": {
            "inject_papers_bib": papers_bib,
        },
    },
}


def process_sources(
    app: sphinx.application.Sphinx, docname: str, source: list[str]
) -> None:
    """
    Render pages whose name is in JINJA_ENV_PER_DOC with Jinja2.

    This allows to use Jinja2 templating in RST/MD files without using specific
    directives.
    """

    if docname not in JINJA_ENV_PER_DOC:
        if app.verbosity > 0:
            print(f"process_jinja: ignoring '{docname}'")  # noqa: T201
        return

    import jinja2

    env = JINJA_ENV_PER_DOC[docname]
    context = env.get("context", {})
    filters = env.get("filters", {})
    substitutions = env.get("substitutions", {})

    jinja_env = jinja2.Environment()
    jinja_env.filters.update(filters)

    # Get page source
    src = source[0]

    # Apply substitutions
    for old, new in substitutions.items():
        if app.verbosity > 0:
            print(f"process_jinja: substituting '{old}' with '{new}' in '{docname}'")  # noqa: T201
        src = src.replace(old, new)

    # Apply Jinja2 rendering
    template = jinja_env.from_string(src)
    rendered = template.render(**context)
    source[0] = rendered

    if app.verbosity > 0:
        print(f"Rendered '{docname}' with Jinja2:")  # noqa: T201

        class repl:
            n: int  # Number of digits to use for line numbering
            cnt: int  # Current line number, increases with each call

            def __init__(self, n=3) -> None:
                self.cnt = 0
                self.n = n

            def __call__(self, *args) -> str:
                self.cnt += 1
                return f"  {self.cnt:0{self.n}d}: "

        import re

        n = max(
            len(str(len(rendered.splitlines()))),
            3,  # Ensure at least 3 digits for line numbers
        )
        line_counter = repl(n)
        print(re.sub(r"(?m)^", line_counter, rendered))  # noqa: T201


def substitute_ref_targets(_, doctree):
    from sphinx.addnodes import pending_xref

    for node in doctree.traverse(condition=pending_xref):
        if (alias := node.get("reftarget", None)) in reftarget_aliases:
            node["reftarget"] = reftarget_aliases[alias]


def process_includes(
    app: sphinx.application.Sphinx, path, docname: str, source: list[str]
) -> None:
    if app.verbosity > 0:
        print(f"process_includes: processing '{docname}' with path '{path}'")  # noqa: T201

    process_sources(app, docname, source)


def setup(app: sphinx.application.Sphinx) -> None:
    app.connect("autodoc-skip-member", skip)
    app.connect("doctree-read", substitute_ref_targets)
    app.connect("source-read", process_sources)
    app.connect("include-read", process_includes)


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_file = ["tippy.css"]

html_theme_options = {
    "prev_next_buttons_location": "both",
    "style_nav_header_background": "coral",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}
