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
import os
import sys
sys.path.insert(0, os.path.abspath('..\..'))


# -- Project information -----------------------------------------------------

project = 'SAMPO'
copyright = '2023, Research Center Strong Artificial Intelligence in Industry'
author = 'Research Center Strong Artificial Intelligence in Industry'

# The full version, including alpha/beta/rc tags
release = ''
version = ''


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.duration', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.doctest',
#               'sphinx.ext.todo', 'sphinx.ext.coverage', 'sphinx.ext.ifconfig']

extensions = ['autoapi.extension'
              ]

# todo_include_todos = True
source_suffix = '.rst'
master_doc = 'index'
add_function_parentheses = True

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
html_theme = 'furo'
html_theme_path = ['themes']
html_title = 'SAMPO'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for Code Examples output ---------------------------------------------------

code_example_dir = "examples"

# -- autoapi configuration ---------------------------------------------------

autoapi_dirs = ['../../sampo']
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
