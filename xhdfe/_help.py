from __future__ import annotations

import argparse
import sys
from importlib import resources

from ._version import __version__

_HELP_PACKAGE = "xhdfe.help"
_HELP_FILE = "xhdfe.md"


def _help_resource():
    return resources.files(_HELP_PACKAGE).joinpath(_HELP_FILE)


def help_text() -> str:
    """Return the packaged xhdfe Python help text."""
    return _help_resource().read_text(encoding="utf-8")


def help_path() -> str:
    """Return the resource path for the packaged xhdfe Python help file."""
    return str(_help_resource())


def print_help(file=None) -> None:
    """Print the packaged xhdfe Python help text."""
    text = help_text()
    target = sys.stdout if file is None else file
    target.write(text)
    if not text.endswith("\n"):
        target.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show xhdfe Python package help.")
    parser.add_argument("--path", action="store_true", help="print the packaged help file path")
    parser.add_argument("--version", action="store_true", help="print the package version")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
    elif args.path:
        print(help_path())
    else:
        print_help()
    return 0
