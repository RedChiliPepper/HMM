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

project = 'attractor_analysis'
copyright = '2025, E B'
author = 'E B'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google-style docstrings
    'myst_parser',         # For Markdown support
    'autoapi.extension',   # For automatic Python API docs
]

html_theme = 'sphinx_rtd_theme'


autoapi_dirs = ['../../src']  # Path to your Python package
autoapi_ignore = ['*test*', '*shell*', '*experiments*', '*logs*', '*data*', '*__pycache__*']  # Ignore test/experiment folders
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'noindex'  # Add this to prevent duplicate indexing
]

"""
# Disable AutoAPI completely if not needed
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True
}
"""
autodoc_default_options = {
    'ignore-module-all': True,
    'noindex': True
}

#autodoc_mock_imports = ['test.test_slds_analyzer', 'test.test_utils']


autoapi_add_toctree_entry = False
autoapi_keep_files = False



# Add these paths to sys.path
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../test/'))
sys.path.insert(0, os.path.abspath('../../experiments/'))
sys.path.insert(0, os.path.abspath('../../scripts/'))
sys.path.insert(0, os.path.abspath('../../data/'))
sys.path.insert(0, os.path.abspath('../../logs/'))
