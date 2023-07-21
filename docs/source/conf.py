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


# -- Project information -----------------------------------------------------

project = "few: Fast EMRI Waveforms"
copyright = "2020, Michael Katz, Alvin Chua, Niels Warburton"
author = "Michael Katz, Alvin Chua, Niels Warburton"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

import pypandoc

output = pypandoc.convert_file("../../README.md", "rst")
with open("README.rst", "w") as fp:
    fp.write(output)

import sys, os

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../../"

sys.path.insert(0, os.path.abspath(dir_path))
sys.path.insert(0, os.path.abspath(dir_path + "few/"))
sys.path.insert(0, os.path.abspath(dir_path + "few/amplitude/"))
sys.path.insert(0, os.path.abspath(dir_path + "few/trajectory/"))

import shutil

shutil.copy(
    dir_path + "examples/FastEMRIWaveforms_tutorial.ipynb",
    dir_path + "docs/source/tutorial/FastEMRIWaveforms_tutorial.ipynb",
)

shutil.copy(
    dir_path + "examples/Tutorial_FD_construction_single_mode.ipynb",
    dir_path + "docs/source/tutorial/Tutorial_FD_construction_single_mode.ipynb",
)

shutil.copy(
    dir_path + "examples/Tutorial_FrequencyDomain_Waveforms.ipynb",
    dir_path + "docs/source/tutorial/Tutorial_FrequencyDomain_Waveforms.ipynb",
)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
html_theme = "sphinx_rtd_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

source_suffix = [".rst"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

import sphinx_rtd_theme

autodoc_member_order = "bysource"


def skip(app, what, name, obj, would_skip, options):
    if name == "__call__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_nav_header_background": "coral",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
}
