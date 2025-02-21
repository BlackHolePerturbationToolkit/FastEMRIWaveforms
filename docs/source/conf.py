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

import pathlib
import os

import few

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
    "utility.ipynb",
    "Amplitude_tutorial.ipynb",
    "swsh.ipynb",
    "modeselect.ipynb",
    "cubicspline.ipynb",
    "modesummation.ipynb",
    "Trajectory_tutorial",
    "Amplitude_tutorial",
):
    filename = example + ".ipynb"
    if not (trg_dir / filename).is_file():
        os.symlink(src_dir / filename, trg_dir / filename)

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

intersphinx_mapping = {"python": ("https://docs.python.org/3.11", None)}

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
    nbsphinx_execute = "always"
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


def substitute_ref_targets(_, doctree):
    from sphinx.addnodes import pending_xref

    for node in doctree.traverse(condition=pending_xref):
        if (alias := node.get("reftarget", None)) in reftarget_aliases:
            node["reftarget"] = reftarget_aliases[alias]


def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.connect("doctree-read", substitute_ref_targets)


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_file = ["tippy.css"]

html_theme_options = {
    "prev_next_buttons_location": "both",
    "style_nav_header_background": "coral",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
}
