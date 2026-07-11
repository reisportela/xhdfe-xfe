"""Python package interface for xhdfe."""

from __future__ import annotations

import importlib

from ._version import __version__
from ._help import help_path, help_text, print_help

_CORE_EXPORTS = {
    "AbsorptionMethod",
    "ConvergenceCriterion",
    "HdfeRegressor",
    "StandardErrorType",
}

__all__ = [
    "__version__",
    "AbsorptionMethod",
    "ConvergenceCriterion",
    "HdfeRegressor",
    "StandardErrorType",
    "help_path",
    "help_text",
    "print_help",
]


def _load_core():
    try:
        from . import py_hdfe_v11 as core
    except ImportError as exc:
        raise ImportError(
            "The xhdfe compiled extension is not available. Build or install the "
            "package with `python -m pip install .` from the repository root."
        ) from exc
    return core


def __getattr__(name: str):
    if name in _CORE_EXPORTS:
        return getattr(_load_core(), name)
    if name == "akm":
        return importlib.import_module(".akm", __name__)
    if name == "gelbach":
        return importlib.import_module(".gelbach", __name__)
    raise AttributeError(f"module 'xhdfe' has no attribute {name!r}")
