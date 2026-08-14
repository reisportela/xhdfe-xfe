#!/usr/bin/env python3
"""Fail-closed validation for xhdfe Python wheel and source artifacts."""

from __future__ import annotations

import argparse
from email.parser import Parser
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
from typing import Iterable, Optional
import zipfile


_WINDOWS_HOST_DLL_RE = re.compile(
    r"^(?:api|ext)-ms-win-.+\.dll$",
    re.IGNORECASE,
)
_LICENSED_BUNDLED_DLL_RE = re.compile(
    r"^(?:libgcc_s_.+|libstdc\+\+-\d+|libgomp-\d+|libwinpthread-\d+|"
    r"libatomic-\d+|libssp-\d+|libquadmath-\d+)\.dll$",
    re.IGNORECASE,
)
_LICENSED_BUNDLED_DLL_NAMES = frozenset({"libdl.dll"})
# Keep this conservative policy aligned with setup.py. Unknown names are
# package dependencies, never silently assumed to exist on a target.
# Source: delvewheel 1.13.0 baseline lists at the pinned upstream commit.
# https://github.com/adang1345/delvewheel/tree/6c13cea9ba5092f327c6766eb427e5371bf498b3/dll_lists
_WINDOWS_HOST_DLL_NAMES = frozenset(
    {
        "advapi32.dll",
        "bcrypt.dll",
        "cfgmgr32.dll",
        "combase.dll",
        "comctl32.dll",
        "comdlg32.dll",
        "crypt32.dll",
        "dwmapi.dll",
        "gdi32.dll",
        "imagehlp.dll",
        "imm32.dll",
        "iphlpapi.dll",
        "kernel32.dll",
        "kernelbase.dll",
        "mpr.dll",
        "msvcrt.dll",
        "netapi32.dll",
        "normaliz.dll",
        "nsi.dll",
        "ntdll.dll",
        "ole32.dll",
        "oleaut32.dll",
        "powrprof.dll",
        "psapi.dll",
        "rpcrt4.dll",
        "sechost.dll",
        "secur32.dll",
        "setupapi.dll",
        "shcore.dll",
        "shell32.dll",
        "shlwapi.dll",
        "ucrtbase.dll",
        "user32.dll",
        "userenv.dll",
        "version.dll",
        "winhttp.dll",
        "winmm.dll",
        "winspool.drv",
        "ws2_32.dll",
        "wtsapi32.dll",
    }
)


def _is_windows_host_dll(
    dll_name: str, python_host_dll_names: frozenset[str]
) -> bool:
    name = Path(dll_name).name.lower()
    return (
        name in _WINDOWS_HOST_DLL_NAMES
        or name in python_host_dll_names
        or bool(_WINDOWS_HOST_DLL_RE.fullmatch(name))
    )


def _is_licensed_bundled_dll(dll_name: str) -> bool:
    """Return whether NOTICE has an explicit license family for this DLL."""

    name = Path(dll_name).name.lower()
    return (
        name in _LICENSED_BUNDLED_DLL_NAMES
        or bool(_LICENSED_BUNDLED_DLL_RE.fullmatch(name))
    )


def _python_host_dll_names(pyd: Path) -> frozenset[str]:
    match = re.search(r"\.cp(\d{2,3})-", pyd.name, re.IGNORECASE)
    _require(match is not None, f"cannot derive CPython ABI from {pyd.name}")
    abi = match.group(1)
    return frozenset({"python3.dll", f"python{abi}.dll", f"python{abi}_d.dll"})


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
        and "winpthreads" in text
        and "Copyright (c) 2011 mingw-w64 project" in text
        and "Copyright (c) 2010 Lockless Inc." in text,
        f"{label} omits GNU/MinGW runtime licensing",
    )
    _require(
        "dlfcn-win32" in text and "MIT License" in text,
        f"{label} omits libdl/dlfcn-win32 runtime licensing",
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
    env = os.environ.copy()
    env.update({"LANG": "C", "LC_ALL": "C"})
    result = subprocess.run(
        [str(objdump), "-p", str(binary)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        errors="replace",
        env=env,
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


def _pe_architecture(objdump: Path, binary: Path) -> str:
    env = os.environ.copy()
    env.update({"LANG": "C", "LC_ALL": "C"})
    result = subprocess.run(
        [str(objdump), "-f", str(binary)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        errors="replace",
        env=env,
    )
    _require(
        result.returncode == 0,
        f"objdump architecture check failed for {binary.name}: "
        f"{result.stderr.strip() or result.stdout.strip()}",
    )
    match = re.search(
        r"\b(pe(?:i)?-[a-z0-9_-]+)\s*$",
        result.stdout,
        re.IGNORECASE | re.MULTILINE,
    )
    _require(match is not None, f"could not identify architecture for {binary.name}")
    return match.group(1).lower()


def _validate_mingw_closure(wheel: Path, objdump: Path) -> None:
    _require(objdump.is_file(), f"MinGW objdump not found: {objdump}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        pyds = list(root.rglob("*.pyd"))
        _require(len(pyds) == 1, f"wheel must contain one .pyd, found {pyds}")
        python_host_dll_names = _python_host_dll_names(pyds[0])
        runtime_paths = list(root.rglob("*.dll"))
        uncovered_licenses = sorted(
            path.name
            for path in runtime_paths
            if not _is_licensed_bundled_dll(path.name)
        )
        _require(
            not uncovered_licenses,
            "wheel contains non-system DLLs without an explicit release "
            "license ledger: " + ", ".join(uncovered_licenses),
        )
        _require(
            all(path.parent == pyds[0].parent for path in runtime_paths),
            "wheel non-system DLLs must be beside the native extension",
        )
        runtime_files = {path.name.lower(): path for path in runtime_paths}
        _require(
            len(runtime_files) == len(runtime_paths),
            "wheel contains duplicate case-insensitive DLL names",
        )
        extension_architecture = _pe_architecture(objdump, pyds[0])
        for runtime in runtime_paths:
            _require(
                _pe_architecture(objdump, runtime) == extension_architecture,
                f"wheel runtime DLL has wrong architecture: {runtime.name}",
            )
        root_dependencies = _pe_dependencies(objdump, pyds[0])
        _require(
            any(name.lower() == "libgomp-1.dll" for name in root_dependencies),
            "MinGW wheel .pyd is not linked to libgomp-1.dll; OpenMP is missing",
        )
        pending = sorted(
            name
            for name in root_dependencies
            if not _is_windows_host_dll(name, python_host_dll_names)
        )
        _require(pending, "MinGW wheel .pyd has no detected non-system dependency")
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
                    not _is_windows_host_dll(dependency, python_host_dll_names)
                    and dependency.lower() not in visited
                ):
                    pending.append(dependency)
        _require(
            visited == set(runtime_files),
            "wheel contains unreferenced non-system DLLs: "
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
