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
    "sphinx_click.ext",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
    "psycopg": ("https://www.psycopg.org/psycopg3/docs", None),
    "python": ("https://docs.python.org/3/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

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

# Avoid errors with self-signed certificates
tls_verify = False

html_theme = "pydata_sphinx_theme"
html_theme_options = {"navigation_depth": 4, "collapse_navigation": False}
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
# html_logo = "_static/logo.png"

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


def _skip_pydantic_methods(name, obj):
    return name in {
        "model_dump_json",
        "model_json_schema",
        "model_dump",
        "model_construct",
        "model_copy",
        "model_fields",
        "model_computed_fields",
        "model_rebuild",
        "model_parametrized_name",
        "model_post_init",
        "model_validate",
        "model_validate_json",
        "model_validate_strings",
        "copy",
        "construct",
        "dict",
        "from_orm",
        "json",
        "parse_file",
        "parse_obj",
        "parse_raw",
        "schema",
        "schema_json",
        "update_forward_refs",
        "validate",
    } and "BaseModel" in str(obj)


def _skip_builtin_methods(name, obj):
    if name in {
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
    } and "MutableMapping" in str(obj):
        return True

    if name in {"items", "keys", "values"} and "Mapping" in str(obj):
        return True

    return name in {"copy", "get"} and "UserDict" in str(obj)


def _skip_internal_api(name, obj):
    if (getattr(obj, "__doc__", None) or "").startswith("[NOT PUBLIC API]"):
        return True

    return name in {"copy", "fromkeys"} and "UsageTracker" in str(obj)


def _skip_member(app, what, name, obj, skip, options):
    if (
        _skip_internal_api(name, obj)
        or _skip_builtin_methods(name, obj)
        or _skip_pydantic_methods(name, obj)
    ):
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_member)


suppress_warnings = ["toc.not_included"]
