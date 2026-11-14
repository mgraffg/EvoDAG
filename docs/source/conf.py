# Configuration file for the Sphinx documentation builder.
#
# For a full list of options see:
# http://www.sphinx-doc.org/en/master/config

# The package is installed (editable) by the builds that render these docs
# (website.yml, test.yml, the test suite) instead of being reached through
# sys.path, so conf.py only imports it to read its version. The autodoc
# directives also import the compiled extensions, so they must be built.

import EvoDAG

# -- Project information -------------------------------------------------

project = "EvoDAG"
copyright = "Mario Graff Guerrero"
author = "Mario Graff Guerrero"
version = EvoDAG.__version__
release = EvoDAG.__version__

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# The docstrings name some types that are not importable objects (array_like,
# for instance); those are not dangling references.
nitpick_ignore_regex = [
    (r"py:class", r"array_like"),
    (r"py:class", r"optional"),
    (r"py:class", r"file-like"),
    (r"py:class", r"list of .*"),
    (r"py:class", r"(int|float|str|bool|dict|class|tuple|list|iterable)( or .*)?"),
]

source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = []

pygments_style = "sphinx"
highlight_language = "python"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
add_function_parentheses = False
add_module_names = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"
htmlhelp_basename = "EvoDAGdoc"
