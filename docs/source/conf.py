"""Sphinx configuration for MaldiAMRKit documentation."""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../.."))

# Project information
project = "MaldiAMRKit"
copyright = "2025, Ettore Rocchi"
author = "Ettore Rocchi"

# The full version, including alpha/beta/rc tags
release = "0.7.0"
version = "0.7"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_design",
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
napoleon_use_ivar = False
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

# Templates and static files
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

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

# -- nbsphinx configuration --------------------------------------------------

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
