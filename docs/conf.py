# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import types
from typing import Any


DOCS_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _ensure_mock_modules() -> None:
    """Register lightweight stand-ins for optional dependencies."""

    def passthrough_decorator(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

    modules = {
        "aopt": types.ModuleType("aopt"),
        "aopt.utils": types.ModuleType("aopt.utils"),
        "aopt.utils.numba": types.ModuleType("aopt.utils.numba"),
        "numba": types.ModuleType("numba"),
    }

    modules["aopt"].utils = modules["aopt.utils"]
    modules["aopt.utils"].numba = modules["aopt.utils.numba"]
    modules["aopt.utils.numba"].rgc = passthrough_decorator

    class _DummyNumbaType:
        def __init__(self, tp: type) -> None:
            self.name = getattr(tp, "__name__", str(tp))

    def _typeof(value, _sig=Any):
        return _DummyNumbaType(type(value))

    modules["numba"].typeof = _typeof

    for name, module in modules.items():
        sys.modules.setdefault(name, module)

    try:
        import numpy as _np  # type: ignore
    except ModuleNotFoundError:
        pass
    else:
        # Restore aliases removed in NumPy 2.x so legacy code keeps working.
        fallback_scalars = {
            "float_": getattr(_np, "float64", None),
            "int_": getattr(_np, "int64", None),
            "complex_": getattr(_np, "complex128", None),
        }
        for attr, value in fallback_scalars.items():
            if value is not None and not hasattr(_np, attr):
                setattr(_np, attr, value)


_ensure_mock_modules()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'numpy_buffermap'
copyright = '2025, '
author = ''
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "both"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "anytree": ("https://anytree.readthedocs.io/en/latest/", None),
}
intersphinx_disabled_reftypes = ["any"]
nitpick_ignore = [("any", "TreeError")]
