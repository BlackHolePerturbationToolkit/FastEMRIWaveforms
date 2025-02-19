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
html_theme = "sphinx_rtd_theme"
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

nbsphinx_requirejs_path = ""

intersphinx_mapping = {"python": ("https://docs.python.org/3.11", None)}

doctest_global_setup = """
import few
"""
doctest_global_cleanup = """
few.utils.globals.reset(True)
"""

myst_heading_anchors = 2

myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "vscode": None,
}

nbsphinx_allow_errors = False

nbsphinx_execute = "always"
nbsphinx_kernel_name = "python3"

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"


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
    # app.connect("missing-reference", resolve_intersphinx_aliases)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
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

tippy_add_class = "has-tippy"
tippy_js = (
    "https://unpkg.com/@popperjs/core@2",
    "https://unpkg.com/tippy.js@6",
    "https://unpkg.com/requirejs@2",
)
