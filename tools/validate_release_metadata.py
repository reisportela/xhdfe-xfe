#!/usr/bin/env python3
"""Validate the unified xhdfe release identity before building artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(relative: str) -> str:
    path = ROOT / relative
    _require(path.is_file(), f"required release file is missing: {relative}")
    return path.read_text(encoding="utf-8")


def _capture(relative: str, pattern: str) -> str:
    match = re.search(pattern, _read(relative), flags=re.MULTILINE)
    _require(match is not None, f"could not read release identity from {relative}")
    return match.group(1)


def _package_text_files(relative: str) -> set[str]:
    files: set[str] = set()
    for line in _read(relative).splitlines():
        match = re.fullmatch(r"f\s+(\S+)", line.strip())
        if match and Path(match.group(1)).suffix in {".ado", ".sthlp"}:
            files.add("stata/" + match.group(1))
    return files


def _package_files(relative: str) -> set[str]:
    files: set[str] = set()
    for line in _read(relative).splitlines():
        match = re.fullmatch(r"f\s+(\S+)", line.strip())
        if match:
            files.add("stata/" + match.group(1))
    return files


def validate(expected_version: str) -> None:
    parts = expected_version.split(".")
    _require(
        len(parts) == 4 and all(part.isdigit() for part in parts),
        "expected version must have the form MAJOR.MINOR.PATCH.YYYYMMDD",
    )
    base_version = ".".join(parts[:3])
    release_date = datetime.strptime(parts[3], "%Y%m%d")
    iso_date = release_date.strftime("%Y-%m-%d")
    english_months = (
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    )
    stata_date = (
        f"{release_date.day:02d}{english_months[release_date.month - 1]}"
        f"{release_date.year}"
    )

    exact_versions = {
        "xhdfe/_version.py": _capture(
            "xhdfe/_version.py", r'^__version__\s*=\s*"([^"]+)"'
        ),
        "pyproject.toml": _capture(
            "pyproject.toml", r'^version\s*=\s*"([^"]+)"'
        ),
        "CMakeLists.txt": _capture(
            "CMakeLists.txt", r"^project\(xhdfe VERSION ([^ )]+)"
        ),
        "r/xhdfe/DESCRIPTION": _capture(
            "r/xhdfe/DESCRIPTION", r"^Version:\s*(\S+)"
        ),
    }
    for relative, actual in exact_versions.items():
        _require(
            actual == expected_version,
            f"{relative}: expected {expected_version}, found {actual}",
        )

    citation = _read("CITATION.cff")
    _require(
        re.search(rf'^version:\s*"{re.escape(base_version)}"$', citation, re.MULTILINE)
        is not None,
        f"CITATION.cff must record version {base_version}",
    )
    _require(
        re.search(rf"^date-released:\s*{re.escape(iso_date)}$", citation, re.MULTILINE)
        is not None,
        f"CITATION.cff must record date {iso_date}",
    )

    pyproject = _read("pyproject.toml")
    _require(
        re.search(r'formulaic>=1\.2\.1,<2', pyproject) is not None,
        "pyproject.toml is missing the Formulaic optional extra",
    )
    _require(
        re.search(r'pandas>=1\.3', pyproject) is not None,
        "pyproject.toml is missing the pandas formula-extra dependency",
    )
    _require(
        f"Package documentation version: {expected_version}" in _read("xhdfe/help/xhdfe.md"),
        "xhdfe/help/xhdfe.md has a stale package version",
    )
    _require(
        f"xhdfe {expected_version}" in _read("xhdfe/help/gelbach.md"),
        "xhdfe/help/gelbach.md has a stale shared version",
    )
    html_help = _read("xhdfe_py_hdfe_v11_help.html")
    _require(expected_version in html_help, "legacy HTML help has a stale version")
    _require("Optional formula interface" in html_help, "legacy HTML help omits formulas")
    _require("pandas&gt;=1.3" in html_help, "legacy HTML help omits the pandas formula dependency")

    _require(
        _capture("stata/xhdfe.pkg", r"^v\s+(\S+)") == base_version,
        "stata/xhdfe.pkg version is not aligned",
    )
    _require(
        _capture("stata/xfe.pkg", r"^v\s+(\S+)") == "1.11.0",
        "stata/xfe.pkg must remain at version 1.11.0",
    )

    production_text = _package_text_files("stata/xhdfe.pkg")
    production_text.update(_package_text_files("stata/xfe.pkg"))
    _require(production_text, "Stata package manifests contain no text files")
    for relative in sorted(production_text):
        header = "\n".join(_read(relative).splitlines()[:12]).lower()
        _require(
            stata_date in header,
            f"{relative}: expected release date {stata_date} in the header",
        )

    package_files = _package_files("stata/xhdfe.pkg")
    package_files.update(_package_files("stata/xfe.pkg"))
    # The public source tree deliberately omits compiled plugins. Their names
    # remain mandatory here; the staged net-install validator checks the actual
    # CI-built binaries after platform assembly.
    generated_plugins = {"stata/xhdfe.plugin", "stata/xfe.plugin"}
    _require(
        generated_plugins.issubset(package_files),
        "Stata package manifests must reference both generated plugins",
    )
    for relative in sorted(package_files - generated_plugins):
        _require(
            (ROOT / relative).is_file(),
            f"required Stata package file is missing: {relative}",
        )
    notice = _read("NOTICE")
    _require(
        re.search(r"GCC Runtime\s+Library Exception 3\.1", notice) is not None
        and "winpthreads" in notice,
        "NOTICE omits GNU/MinGW runtime licensing",
    )
    _require(
        (ROOT / "LICENSE").read_bytes() == (ROOT / "stata/LICENSE").read_bytes()
        and (ROOT / "NOTICE").read_bytes() == (ROOT / "stata/NOTICE").read_bytes(),
        "Stata LICENSE/NOTICE copies are not byte-identical to the project files",
    )

    for relative in (
        "stata/xhdfe.ado",
        "stata/xhdfe_p.ado",
        "stata/xhdfe_estat.ado",
        "stata/xhdfe.sthlp",
        "stata/xhdfegpu.ado",
        "stata/xhdfegpu.sthlp",
    ):
        header = "\n".join(_read(relative).splitlines()[:12])
        _require(
            re.search(rf"\b{re.escape(base_version)}\b", header) is not None,
            f"{relative}: expected xhdfe version {base_version} in the header",
        )

    _require(
        (ROOT / "stata/xhdfe.ado").read_bytes()
        == (ROOT / "share/xhdfe_estimation_cpp/stata/xhdfe.ado").read_bytes(),
        "share mirror of xhdfe.ado is not byte-identical",
    )
    _require(
        (ROOT / "stata/xhdfe.sthlp").read_bytes()
        == (ROOT / "share/xhdfe_estimation_cpp/stata/xhdfe.sthlp").read_bytes(),
        "share mirror of xhdfe.sthlp is not byte-identical",
    )

    release_note = ROOT / f"docs/releases/RELEASE_NOTES_{expected_version}.md"
    _require(release_note.is_file(), f"release note is missing: {release_note}")
    _require(
        release_note.name in _read("docs/releases/README.md"),
        "docs/releases/README.md does not link the current release note",
    )
    _require(
        base_version in _read("r/xhdfe/R/docs.R")
        and base_version in _read("r/xhdfe/man/xhdfe-package.Rd"),
        "R package overview documentation has a stale shared version",
    )

    print(
        "Unified release metadata OK: "
        f"{expected_version}; Stata date {stata_date}; "
        f"{len(production_text)} production Stata text files"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    validate(args.expected_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
