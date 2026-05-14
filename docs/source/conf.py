"""Sphinx configuration for MaldiAMRKit documentation."""

import os
import shutil
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../.."))

from maldiamrkit import __version__  # noqa: E402

# Copy notebooks from repo root into the Sphinx source tree so that nbsphinx
# processes real files rather than following a symlink.  Symlinked notebooks
# break image extraction on ReadTheDocs.
_here = Path(__file__).parent
_notebooks_src = _here.parent.parent / "notebooks"
_notebooks_dst = _here / "tutorials" / "notebooks"

if _notebooks_src.exists():
    if _notebooks_dst.is_symlink():
        _notebooks_dst.unlink()
    if _notebooks_dst.exists():
        shutil.rmtree(_notebooks_dst)
    shutil.copytree(_notebooks_src, _notebooks_dst)

# Project information
project = "MaldiAMRKit"
copyright = "2025-2026, Ettore Rocchi"
author = "Ettore Rocchi"

# The full version, including alpha/beta/rc tags
release = __version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_design",
    "sphinx_click",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Suppress warnings for shorthand types in NumPy-style docstrings
# that Sphinx cannot resolve
nitpick_ignore_regex = [
    (r"py:class", r"optional"),
    (r"py:class", r"default=.*"),
    (r"py:class", r"array-like"),
    (r"py:class", r"np\..*"),
    (r"py:class", r"pd\..*"),
    (r"py:class", r"ndarray"),
    (r"py:class", r"Path"),
    (r"py:class", r"arrays"),
    (r"py:class", r"tuples"),
    (r"py:class", r"callable"),
    (r"py:class", r"ignored"),
    (r"py:class", r"self"),
    (r"py:class", r"transformer"),
    (r"py:class", r"typing\..*"),
    (r"py:class", r"InputLayout"),
    (r"py:class", r"DatasetLayout"),
    (r"py:class", r"matplotlib\..*"),
    (r"py:class", r"pandas\..*"),
    (r"py:class", r"umap\..*"),
    (r"py:class", r"MaldiSet"),
    (r"py:class", r"PreprocessingPipeline.*"),
    (r"py:class", r"maldiamrkit\..*"),
    (r"py:class", r"\d+"),
    (r"py:class", r"\{.*"),
    (r"py:class", r"\".*"),
    (r"py:meth", r"PreprocessingPipeline\..*"),
    (r"py:func", r"DatasetBuilder\.build"),
    (r"py:data", r"typing\..*"),
    (r"py:obj", r"maldiamrkit\.evaluation\.(vme|me)_scorer"),
    # Numpy-style docstring type annotations that Napoleon parses as
    # individual class references (e.g. "Series, shape (n, n)").
    (r"py:class", r"Axes"),
    (r"py:class", r"DataFrame"),
    (r"py:class", r"Series"),
    (r"py:class", r"shape"),
    (r"py:class", r"n"),
    (r"py:class", r"n - 1"),
    (r"py:class", r"None\}"),
    (r"py:class", r"mz_min"),
    (r"py:class", r"mz_max"),
]

# Static files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/maldiamrkit.png"

html_theme_options = {
    # Logo configuration
    "logo": {
        "text": "MaldiAMRKit",
        "image_light": "_static/maldiamrkit.png",
        "image_dark": "_static/maldiamrkit.png",
    },
    # Top navigation bar layout
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 4,
    # Icon links (GitHub, PyPI)
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/EttoreRocchi/MaldiAMRKit",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/MaldiAMRKit/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    # Sidebar behaviour
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "collapse_navigation": True,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    # Syntax highlighting
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

# Sidebar configuration: no left sidebar on the landing page
html_sidebars = {
    "**": ["sidebar-nav-bs"],
    "index": [],
}

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
