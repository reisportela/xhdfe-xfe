#!/usr/bin/env python3
"""Validate the unified xhdfe release identity before building artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
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
    test_extra = re.search(r"(?ms)^test\s*=\s*\[(.*?)^\]", pyproject)
    _require(test_extra is not None, "pyproject.toml is missing the test extra")
    for requirement in (
        '"formulaic>=1.2.1,<2"',
        '"pandas>=1.3"',
        '"setuptools>=69"',
        '"maketables>=0.1.7"',
    ):
        _require(
            requirement in test_extra.group(1),
            f"pyproject.toml test extra is missing {requirement}",
        )
    readme = _read("README.md")
    _require(
        f"`Version {base_version}`" in readme,
        f"README.md header must record version {base_version}",
    )
    _require(
        f"* Version {base_version}." in readme,
        f"README.md citation must record version {base_version}",
    )
    for required_credit in (
        "by Alexander Fischer and Kristof Schröder",
        "Direct algorithmic starting point for `xhdfe`'s "
        "graph-preconditioned MLSMR/additive-Schwarz route.",
        "algorithmic contribution and should be cited as such.",
        "We thank Alexander Fischer and Kristof Schröder",
    ):
        _require(
            required_credit in readme,
            "README.md is missing the required Fischer-Schroder credit: "
            + required_credit,
        )
    readme_normalized = re.sub(r"\s+", " ", readme)
    for required_credit in (
        "We are especially grateful to Marta Silva",
        "scrutinized numerical precision and the accuracy of the estimates",
        "group()` / `individual()` models",
    ):
        _require(
            required_credit in readme_normalized,
            "README.md is missing the required Marta Silva credit: "
            + required_credit,
        )
    _require(
        "if(APPLE OR WIN32)" in _read("CMakeLists.txt"),
        "CMakeLists.txt must disable native CPU tuning by default on Windows",
    )
    _require(
        f"Package documentation version: {expected_version}" in _read("xhdfe/help/xhdfe.md"),
        "xhdfe/help/xhdfe.md has a stale package version",
    )
    _require(
        f"xhdfe {expected_version}" in _read("xhdfe/help/gelbach.md"),
        "xhdfe/help/gelbach.md has a stale shared version",
    )
    _require(
        _capture("stata/xhdfe.pkg", r"^v\s+(\S+)") == base_version,
        "stata/xhdfe.pkg version is not aligned",
    )
    _require(
        _capture("stata/xfepout.pkg", r"^v\s+(\S+)") == "1.13.2",
        "stata/xfepout.pkg must record version 1.13.2",
    )
    for retired in (
        "stata/xfe.ado",
        "stata/xfe.sthlp",
        "stata/xfe.pkg",
        "stata/tools/build-xfe-plugin.sh",
    ):
        _require(
            not (ROOT / retired).exists(),
            f"retired Stata frontend file must not be shipped: {retired}",
        )
    xfepout_ado = _read("stata/xfepout.ado")
    _require(
        re.search(r"^program define xfepout\b", xfepout_ado, re.MULTILINE)
        is not None,
        "stata/xfepout.ado does not define the xfepout command",
    )
    _require(
        'ereturn local cmd "xfepout"' in xfepout_ado,
        "stata/xfepout.ado does not store e(cmd)=xfepout",
    )
    stata_toc = _read("stata/stata.toc")
    _require(
        re.search(r"^p\s+xfepout\b", stata_toc, re.MULTILINE) is not None
        and re.search(r"^p\s+xfe\b", stata_toc, re.MULTILINE) is None,
        "stata/stata.toc must publish xfepout and must not publish xfe",
    )

    production_text = _package_text_files("stata/xhdfe.pkg")
    production_text.update(_package_text_files("stata/xfepout.pkg"))
    _require(production_text, "Stata package manifests contain no text files")
    for relative in sorted(production_text):
        header = "\n".join(_read(relative).splitlines()[:12]).lower()
        _require(
            stata_date in header,
            f"{relative}: expected release date {stata_date} in the header",
        )

    package_files = _package_files("stata/xhdfe.pkg")
    package_files.update(_package_files("stata/xfepout.pkg"))
    # The public source tree deliberately omits compiled plugins. Their names
    # remain mandatory here; the staged net-install validator checks the actual
    # CI-built binaries after platform assembly.
    generated_plugins = {"stata/xhdfe.plugin", "stata/xfepout.plugin"}
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
    source_asset = (
        "https://github.com/reisportela/xhdfe-xfe/releases/download/"
        f"v{expected_version}/xhdfe-{expected_version}-corresponding-source.zip"
    )
    _require(source_asset in notice, "NOTICE has a stale corresponding-source URL")
    license_hashes = {
        "GCC-13.2.0-COPYING3":
            "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
        "GCC-13.2.0-COPYING.RUNTIME":
            "9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74",
        "mingw-w64-11.0.1-winpthreads-COPYING":
            "63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177",
        "dlfcn-win32-1.4.1-COPYING":
            "4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7",
    }
    for filename, expected_hash in license_hashes.items():
        path = ROOT / "third_party" / "licenses" / filename
        _require(path.is_file(), f"missing runtime license: {path}")
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash,
            f"runtime license hash mismatch: {filename}",
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
