"""Python package interface for xhdfe."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from ._version import __version__
from ._help import help_path, help_text, print_help

_CORE_EXPORTS = {
    "AbsorptionMethod",
    "ConvergenceCriterion",
    "HdfeRegressor",
    "PrecisionWarning",
    "StandardErrorType",
}

_FORMULA_EXPORTS = {
    "PreparedFormula",
    "feols",
    "prepare_formula",
}

_PACKAGED_DLL_DIRECTORY_HANDLES: dict[str, object] = {}

__all__ = [
    "__version__",
    "AbsorptionMethod",
    "ConvergenceCriterion",
    "HdfeRegressor",
    "PrecisionWarning",
    "StandardErrorType",
    "PreparedFormula",
    "feols",
    "prepare_formula",
    "help_path",
    "help_text",
    "print_help",
]


def _register_packaged_dll_directory(
    package_dir=None, *, add_dll_directory=None
):
    """Register bundled MinGW runtimes before loading the native extension."""

    if add_dll_directory is None:
        if os.name != "nt":
            return None
        add_dll_directory = getattr(os, "add_dll_directory", None)

    directory = (
        Path(__file__).resolve().parent
        if package_dir is None
        else Path(package_dir).resolve()
    )
    has_packaged_dependency = any(
        candidate.is_file() and candidate.suffix.lower() == ".dll"
        for candidate in directory.iterdir()
    )
    if not has_packaged_dependency:
        return None
    if add_dll_directory is None:
        raise ImportError(
            "This xhdfe installation contains bundled MinGW runtime DLLs, but "
            "this Python runtime does not provide os.add_dll_directory()."
        )

    key = str(directory)
    if key not in _PACKAGED_DLL_DIRECTORY_HANDLES:
        try:
            _PACKAGED_DLL_DIRECTORY_HANDLES[key] = add_dll_directory(key)
        except OSError as exc:
            raise ImportError(
                "xhdfe could not register its packaged DLL directory before "
                "loading the compiled extension."
            ) from exc
    return _PACKAGED_DLL_DIRECTORY_HANDLES[key]


_register_packaged_dll_directory()


def _load_core():
    try:
        from . import py_hdfe_v11 as core
    except ImportError as exc:
        raise ImportError(
            "The xhdfe compiled extension is not available. Build or install the "
            "package with `python -m pip install .` from the repository root."
        ) from exc
    _attach_maketables_hooks(core)
    return core


def _attach_maketables_hooks(core):
    """Expose the maketables plug-in format on the native regressor.

    The hooks are duck-typed attributes read by maketables when a user renders a
    table; nothing is imported from it and no optional dependency is pulled in.
    See :mod:`xhdfe._maketables`. Attaching is idempotent and never allowed to
    break estimation: a core build that rejects attribute injection simply keeps
    the array API untabulatable, which is how it behaved before.
    """
    regressor = getattr(core, "HdfeRegressor", None)
    if regressor is None:
        return
    from . import _maketables

    try:
        _maketables.attach(regressor)
    except (AttributeError, TypeError):
        pass


def __getattr__(name: str):
    if name in _CORE_EXPORTS:
        return getattr(_load_core(), name)
    if name in _FORMULA_EXPORTS:
        return getattr(importlib.import_module("._formula", __name__), name)
    if name == "akm":
        return importlib.import_module(".akm", __name__)
    if name == "gelbach":
        return importlib.import_module(".gelbach", __name__)
    raise AttributeError(f"module 'xhdfe' has no attribute {name!r}")
