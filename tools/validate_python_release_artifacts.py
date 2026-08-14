#!/usr/bin/env python3
"""Fail-closed validation for xhdfe Python wheel and source artifacts."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
from typing import Iterable, Optional
import zipfile


_MINGW_RUNTIME_DLL_RE = re.compile(
    r"^(?:libgcc_s_.+|libstdc\+\+-\d+|libgomp-\d+|libwinpthread-\d+|"
    r"libatomic-\d+|libssp-\d+|libquadmath-\d+)\.dll$",
    re.IGNORECASE,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _metadata_checks(text: str, expected_version: str, label: str) -> None:
    metadata = Parser().parsestr(text)
    _require(
        metadata.get("Version") == expected_version,
        f"{label}: expected version {expected_version}, got {metadata.get('Version')!r}",
    )
    extras = set(metadata.get_all("Provides-Extra") or ())
    _require("formula" in extras, f"{label}: missing Provides-Extra: formula")
    requirements = metadata.get_all("Requires-Dist") or ()
    normalized_requirements = [item.lower().replace(" ", "") for item in requirements]
    formula_markers = ('extra=="formula"', "extra=='formula'")
    _require(
        any(
            "formulaic<2,>=1.2.1" in item
            and any(marker in item for marker in formula_markers)
            for item in normalized_requirements
        ),
        f"{label}: missing formulaic>=1.2.1,<2 requirement for the formula extra",
    )
    _require(
        any(
            "pandas>=1.3" in item
            and any(marker in item for marker in formula_markers)
            for item in normalized_requirements
        ),
        f"{label}: missing pandas>=1.3 requirement for the formula extra",
    )


def _runtime_notice_checks(text: str, label: str) -> None:
    _require(
        re.search(r"GCC Runtime\s+Library Exception 3\.1", text) is not None
        and "winpthreads" in text,
        f"{label} omits GNU/MinGW runtime licensing",
    )


def _dll_bootstrap_checks(text: str, label: str) -> None:
    _require(
        "add_dll_directory" in text
        and "_PACKAGED_DLL_DIRECTORY_HANDLES" in text
        and "_register_packaged_dll_directory()" in text,
        f"{label} omits the retained packaged-DLL directory bootstrap",
    )


def _member_with_suffix(names: Iterable[str], suffix: str, label: str) -> str:
    matches = [
        name
        for name in names
        if name == suffix.lstrip("/") or name.endswith(suffix)
    ]
    _require(len(matches) == 1, f"{label}: expected one {suffix}, found {matches}")
    return matches[0]


def _wheel_notice_member(names: Iterable[str], filename: str) -> str:
    pattern = re.compile(
        rf"\.dist-info/(?:licenses/)?{re.escape(filename)}$",
        re.IGNORECASE,
    )
    matches = [name for name in names if pattern.search(name)]
    _require(
        len(matches) == 1,
        f"wheel: expected one packaged {filename}, found {matches}",
    )
    return matches[0]


def _pe_dependencies(objdump: Path, binary: Path) -> set[str]:
    result = subprocess.run(
        [str(objdump), "-p", str(binary)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        errors="replace",
    )
    _require(
        result.returncode == 0,
        f"objdump failed for {binary.name}: {result.stderr.strip() or result.stdout.strip()}",
    )
    pattern = re.compile(r"^\s*DLL Name:\s*(\S+)\s*$", re.IGNORECASE)
    return {
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := pattern.match(line)) is not None
    }


def _validate_mingw_closure(wheel: Path, objdump: Path) -> None:
    _require(objdump.is_file(), f"MinGW objdump not found: {objdump}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        pyds = list(root.rglob("*.pyd"))
        _require(len(pyds) == 1, f"wheel must contain one .pyd, found {pyds}")
        runtime_paths = [
            path
            for path in root.rglob("*.dll")
            if _MINGW_RUNTIME_DLL_RE.fullmatch(path.name)
        ]
        _require(
            all(path.parent == pyds[0].parent for path in runtime_paths),
            "wheel GNU runtime DLLs must be beside the native extension",
        )
        runtime_files = {path.name.lower(): path for path in runtime_paths}
        _require(
            len(runtime_files) == len(runtime_paths),
            "wheel contains duplicate case-insensitive GNU runtime DLL names",
        )
        root_dependencies = _pe_dependencies(objdump, pyds[0])
        _require(
            any(name.lower() == "libgomp-1.dll" for name in root_dependencies),
            "MinGW wheel .pyd is not linked to libgomp-1.dll; OpenMP is missing",
        )
        pending = sorted(
            name
            for name in root_dependencies
            if _MINGW_RUNTIME_DLL_RE.fullmatch(name)
        )
        _require(pending, "MinGW wheel .pyd has no detected GNU runtime dependency")
        visited: set[str] = set()
        while pending:
            name = pending.pop(0)
            key = name.lower()
            if key in visited:
                continue
            visited.add(key)
            _require(key in runtime_files, f"wheel is missing required runtime DLL {name}")
            for dependency in _pe_dependencies(objdump, runtime_files[key]):
                if (
                    _MINGW_RUNTIME_DLL_RE.fullmatch(dependency)
                    and dependency.lower() not in visited
                ):
                    pending.append(dependency)
        _require(
            visited == set(runtime_files),
            "wheel contains unreferenced GNU runtime DLLs: "
            + ", ".join(sorted(set(runtime_files) - visited)),
        )


def validate_wheel(
    wheel: Path,
    expected_version: str,
    mingw_objdump: Optional[Path] = None,
) -> None:
    _require(wheel.is_file(), f"wheel not found: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        init_name = _member_with_suffix(names, "/xhdfe/__init__.py", "wheel")
        _dll_bootstrap_checks(
            archive.read(init_name).decode("utf-8"), "wheel xhdfe/__init__.py"
        )
        _member_with_suffix(names, "/xhdfe/_formula.py", "wheel")
        _member_with_suffix(names, "/xhdfe/help/xhdfe.md", "wheel")
        _wheel_notice_member(names, "LICENSE")
        notice_name = _wheel_notice_member(names, "NOTICE")
        notice = archive.read(notice_name).decode("utf-8")
        _runtime_notice_checks(notice, "wheel NOTICE")
        metadata_name = _member_with_suffix(names, ".dist-info/METADATA", "wheel")
        _metadata_checks(
            archive.read(metadata_name).decode("utf-8"),
            expected_version,
            "wheel METADATA",
        )
    if mingw_objdump is not None:
        _validate_mingw_closure(wheel, mingw_objdump)


def validate_sdist(sdist: Path, expected_version: str) -> None:
    _require(sdist.is_file(), f"sdist not found: {sdist}")
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        _require(len(roots) == 1, f"sdist must have one top-level directory: {roots}")
        root = next(iter(roots))
        for suffix in (
            "/pyproject.toml",
            "/setup.py",
            "/LICENSE",
            "/NOTICE",
            "/xhdfe/__init__.py",
            "/xhdfe/_formula.py",
            "/xhdfe/help/xhdfe.md",
            "/tests/test_formula_frontend.py",
            "/tests/test_windows_runtime_packaging.py",
        ):
            expected = root + suffix
            _require(expected in names, f"sdist is missing {expected}")
        init_member = archive.extractfile(root + "/xhdfe/__init__.py")
        _require(init_member is not None, "sdist xhdfe/__init__.py could not be read")
        _dll_bootstrap_checks(
            init_member.read().decode("utf-8"), "sdist xhdfe/__init__.py"
        )
        notice_member = archive.extractfile(root + "/NOTICE")
        _require(notice_member is not None, "sdist NOTICE could not be read")
        notice = notice_member.read().decode("utf-8")
        _runtime_notice_checks(notice, "sdist NOTICE")
        metadata_name = root + "/PKG-INFO"
        _require(metadata_name in names, f"sdist is missing {metadata_name}")
        member = archive.extractfile(metadata_name)
        _require(member is not None, "sdist PKG-INFO could not be read")
        _metadata_checks(
            member.read().decode("utf-8"),
            expected_version,
            "sdist PKG-INFO",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--mingw-objdump", type=Path)
    args = parser.parse_args()
    _require(args.wheel is not None or args.sdist is not None, "supply a wheel or sdist")
    if args.wheel is not None:
        validate_wheel(args.wheel, args.expected_version, args.mingw_objdump)
    if args.sdist is not None:
        validate_sdist(args.sdist, args.expected_version)
    print("Python release artifact closure OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
