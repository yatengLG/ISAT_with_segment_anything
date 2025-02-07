# -*- coding: utf-8 -*-

import os


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return cur_dir

def get_locale_path():
    return  os.path.join(os.path.abspath(os.path.dirname(__file__)), 'locale')


def setup(app):
    app.add_message_catalog('sphinx', get_locale_path())
    app.add_html_theme('sphinx_pdj_theme', get_html_theme_path())
