from __future__ import annotations

import argparse
import sys
from importlib import resources

from ._version import __version__

_HELP_PACKAGE = "xhdfe.help"
_HELP_FILES = {
    "xhdfe": "xhdfe.md",
    "gelbach": "gelbach.md",
}


def _help_resource(topic: str = "xhdfe"):
    try:
        filename = _HELP_FILES[topic]
    except KeyError as exc:
        choices = ", ".join(sorted(_HELP_FILES))
        raise ValueError(f"unknown help topic {topic!r}; choose from: {choices}") from exc
    return resources.files(_HELP_PACKAGE).joinpath(filename)


def help_text(topic: str = "xhdfe") -> str:
    """Return packaged help for ``xhdfe`` or the ``gelbach`` companion."""
    return _help_resource(topic).read_text(encoding="utf-8")


def help_path(topic: str = "xhdfe") -> str:
    """Return the resource path for a packaged xhdfe help topic."""
    return str(_help_resource(topic))


def print_help(file=None, *, topic: str = "xhdfe") -> None:
    """Print packaged help; the default keeps the historical xhdfe topic."""
    text = help_text(topic)
    target = sys.stdout if file is None else file
    target.write(text)
    if not text.endswith("\n"):
        target.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show xhdfe Python package help.")
    parser.add_argument(
        "topic", nargs="?", default="xhdfe", choices=sorted(_HELP_FILES),
        help="help topic (default: xhdfe)",
    )
    parser.add_argument("--path", action="store_true", help="print the packaged help file path")
    parser.add_argument("--version", action="store_true", help="print the package version")
    parser.add_argument("--topics", action="store_true", help="list available help topics")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
    elif args.topics:
        print("\n".join(sorted(_HELP_FILES)))
    elif args.path:
        print(help_path(args.topic))
    else:
        print_help(topic=args.topic)
    return 0
