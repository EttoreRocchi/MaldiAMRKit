"""Sphinx configuration for MaldiAMRKit documentation."""
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'MaldiAMRKit'
copyright = '2024, Ettore Rocchi'
author = 'Ettore Rocchi'

# The full version, including alpha/beta/rc tags
release = '0.5.0'
version = '0.5'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'nbsphinx',
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
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Templates and static files
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/maldiamrkit.png'

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'navigation_depth': 4,
    'titles_only': False,
    'collapse_navigation': False,
}

# -- nbsphinx configuration --------------------------------------------------

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
