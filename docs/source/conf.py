# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'ISAT-SAM'
copyright = '2025, Shuwei Ji and Hongyuan Zhang'
author = 'Shuwei Ji and Hongyuan Zhang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # 支持 Google/NumPy 风格注释
]

autodoc_default_options = {
    'members': True,          # 提取类成员
    'member-order': 'bysource', # 按源码顺序排列
    'special-members': '', # 提取特殊方法（如 __init__）
    'undoc-members': False,   # 不提取无文档的成员（避免干扰）
    'exclude-members': '__weakref__' # 排除不需要的成员
}
autodoc_typehints = 'description'

napoleon_google_docstring = True

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

locale_dirs = ['locales/']
gettext_compact = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'navigation_depth': 6,
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
}
html_logo = "_static/ISAT_new_64.svg"
html_favicon = "_static/ISAT_new_64.svg"
html_show_sourcelink = True
html_static_path = ['_static']

