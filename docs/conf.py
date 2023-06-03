# -*- coding: utf-8 -*-

import sys, os

sys.path.insert(0, os.path.abspath('extensions'))

extensions = ['sphinx.ext.duration', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.doctest',
              'sphinx.ext.todo', 'sphinx.ext.coverage', 'sphinx.ext.ifconfig']

todo_include_todos = True
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = []
add_function_parentheses = True
#add_module_names = True
# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

project = u'Music for Geeks and Nerds'
copyright = u'2012, Pedro Kroger'

version = ''
release = ''

# -- Options for HTML output ---------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = ['themes']
html_title = "Music for Geeks and Nerds"
#html_short_title = None
#html_logo = None
#html_favicon = None
# html_static_path = ['_static']
# html_domain_indices = False
# html_use_index = False
# html_show_sphinx = False
# htmlhelp_basename = 'MusicforGeeksandNerdsdoc'
# html_show_sourcelink = False

# -- Options for Code Examples output ---------------------------------------------------

code_example_dir = "examples"
