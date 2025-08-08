# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # kececilayout modülünü dahil et

project = 'kececilayout'
copyright = '2025, Mehmet Keçeci'
author = 'Mehmet Keçeci'
release = '0.2.7'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'logo_only': True,
}

# intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'igraph': ('https://igraph.org/python/doc/', None),
    'rustworkx': ('https://qiskit.org/rustworkx/', None),
    'networkit': ('https://networkit.github.io/documentation/', None),
    'graphillion': ('https://graphillion.readthedocs.io/en/latest/', None),
}
