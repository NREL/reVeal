# -*- coding: utf-8 -*-
"""
sphinx documentation config file
"""

# pylint:disable=invalid-name,redefined-builtin,unused-argument
import os
import sys

from reVeal._version import __version__


sys.path.insert(0, os.path.abspath("../../"))

project = "reVeal"
copyright = "2025, Alliance for Energy Innovation, LLC and Root Geospatial LLC"
author = "NLR: Michael Gleason, Pavlo Pinchuk, Victor Igwe, Travis Williams"

pkg = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
pkg = os.path.dirname(pkg)
sys.path.append(pkg)

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_click.ext",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3/", None)}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"

exclude_patterns = [
    "**.ipynb_checkpoints",
    "**__pycache__**",
    "**/includes/**",
]

pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "logo_only": True,
}
html_css_file = ["custom.css"]
html_context = {
    "display_github": True,
    "github_user": "nlr",
    "github_repo": "reVeal",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "source_suffix": source_suffix,
}
html_static_path = ["_static"]
html_logo = "_static/logo.png"

htmlhelp_basename = "reVealdoc"

latex_elements = {}

latex_documents = [
    (master_doc, "reVeal.tex", "reVeal Documentation", author, "manual"),
]

man_pages = [(master_doc, "reVeal", "reVeal Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "reVeal",
        "reVeal Documentation",
        author,
        "reVeal",
        "reVeal: the reV Extension for Analyzing Large Loads.",
        "Miscellaneous",
    ),
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
add_module_names = False  # Remove namespaces from class/method signatures
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
numpy_show_class_member = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False


def skip_pydantic_methods(app, what, name, obj, skip, options):
    """
    Helper function to skip listed methods from pydantic which are responsible for
    raising a number of sphinx lint warnings and errors.
    """
    if name in (
        "model_dump_json",
        "model_json_schema",
        "model_dump",
        "model_construct",
        "model_copy",
        "model_validate",
        "model_validate_json",
        "model_validate_strings",
    ):
        return True
    return None


def setup(app):
    """
    Apply the helper function for skipping pydantic methods.
    """
    app.connect("autodoc-skip-member", skip_pydantic_methods)


suppress_warnings = ["toc.not_included"]
