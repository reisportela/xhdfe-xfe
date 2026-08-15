#!/usr/bin/env python3
"""Fail-closed validation for xhdfe Python wheel and source artifacts."""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Iterable, Optional
import zipfile


_WINDOWS_HOST_DLL_RE = re.compile(
    r"^(?:api|ext)-ms-win-.+\.dll$",
    re.IGNORECASE,
)
_WINDOWS_RUNTIME_MANIFEST = "_windows_runtime_manifest.json"
_WINDOWS_RUNTIME_FORMAT = "xhdfe-windows-runtime-closure-v1"
_WINDOWS_RUNTIME_LICENSES = (
    (
        re.compile(
            r"^(?:libgcc_s_.+|libstdc\+\+-\d+|libgomp-\d+|libatomic-\d+|"
            r"libssp-\d+|libquadmath-\d+)\.dll$",
            re.IGNORECASE,
        ),
        "GPL-3.0-or-later WITH GCC-exception-3.1",
    ),
    (
        re.compile(r"^libwinpthread-\d+\.dll$", re.IGNORECASE),
        "MIT AND BSD-3-Clause",
    ),
    (re.compile(r"^libdl\.dll$", re.IGNORECASE), "MIT"),
)
_WINDOWS_RUNTIME_RESOLUTION_METHODS = frozenset(
    {
        "compiler-adjacent",
        "compiler-print-file-name",
        "explicit-runtime-source",
        "requester-directory",
        "runtime-search-directory",
    }
)
_PE_X86_64_ARCHITECTURES = frozenset({"pe-x86-64", "pei-x86-64"})
_RUNTIME_LICENSE_HASHES = {
    "GCC-13.2.0-COPYING3":
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
    "GCC-13.2.0-COPYING.RUNTIME":
        "9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74",
    "mingw-w64-11.0.1-winpthreads-COPYING":
        "63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177",
    "dlfcn-win32-1.4.1-COPYING":
        "4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7",
}
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

    return _windows_runtime_license(dll_name) is not None


def _windows_runtime_license(dll_name: str) -> str | None:
    name = Path(dll_name).name
    for pattern, license_expression in _WINDOWS_RUNTIME_LICENSES:
        if pattern.fullmatch(name):
            return license_expression
    return None


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
    allowed_suffixes = {
        f".dist-info/{filename}".casefold(),
        f".dist-info/licenses/{filename}".casefold(),
    }
    if filename in _RUNTIME_LICENSE_HASHES:
        # PEP 639 preserves the source-relative path of entries selected by
        # `license-files = ["third_party/licenses/*"]` under `.dist-info/licenses`.
        allowed_suffixes.add(
            f".dist-info/licenses/third_party/licenses/{filename}".casefold()
        )
    matches = [
        name
        for name in names
        if any(name.casefold().endswith(suffix) for suffix in allowed_suffixes)
    ]
    _require(
        len(matches) == 1,
        f"wheel: expected one packaged {filename}, found {matches}",
    )
    return matches[0]


def _wheel_runtime_license_checks(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    for filename, expected_hash in _RUNTIME_LICENSE_HASHES.items():
        member = _wheel_notice_member(names, filename)
        _require(
            _sha256_bytes(archive.read(member)) == expected_hash,
            f"wheel: {filename} hash mismatch",
        )


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_without_duplicate_keys(data: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}: invalid UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label}: top level must be an object")
    return value


def _checked_dependencies(value: object, label: str) -> list[str]:
    _require(isinstance(value, list), f"{label}: dependencies must be a list")
    dependencies = value
    _require(
        all(
            isinstance(name, str)
            and name
            and Path(name).name == name
            and PureWindowsPath(name).name == name
            for name in dependencies
        ),
        f"{label}: dependency names must be non-empty basenames",
    )
    _require(
        dependencies == sorted(dependencies, key=str.casefold),
        f"{label}: dependencies must be deterministically sorted",
    )
    _require(
        len({name.casefold() for name in dependencies}) == len(dependencies),
        f"{label}: duplicate case-insensitive dependency names",
    )
    return dependencies


def _checked_member_name(value: object, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label}: invalid member name")
    _require(
        Path(value).name == value and PureWindowsPath(value).name == value,
        f"{label}: member must be a basename",
    )
    return value


def _checked_hash(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label}: expected a lowercase SHA-256 digest",
    )
    return value


def _checked_size(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label}: size must be a non-negative integer",
    )
    return value


def _checked_source_path(value: object, member: str, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label}: missing source_path")
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    selected = windows_path if windows_path.is_absolute() else posix_path
    _require(selected.is_absolute(), f"{label}: source_path must be absolute")
    _require(
        "." not in selected.parts and ".." not in selected.parts,
        f"{label}: source_path must be canonical",
    )
    _require(
        selected.name.casefold() == member.casefold(),
        f"{label}: source_path basename does not match member",
    )
    return value


def _records_by_member(
    records: object, expected_fields: frozenset[str], label: str
) -> dict[str, dict[str, object]]:
    _require(isinstance(records, list), f"{label} must be a list")
    members: list[str] = []
    result: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        record_label = f"{label}[{index}]"
        _require(isinstance(record, dict), f"{record_label} must be an object")
        _require(
            frozenset(record) == expected_fields,
            f"{record_label} fields differ from the {_WINDOWS_RUNTIME_FORMAT} schema",
        )
        member = _checked_member_name(record["member"], record_label)
        key = member.casefold()
        _require(key not in result, f"{label}: duplicate member {member}")
        members.append(member)
        result[key] = record
    _require(
        members == sorted(members, key=str.casefold),
        f"{label} must be deterministically sorted by member",
    )
    return result


def _validate_pe_runtime_ledger(
    *,
    roots: dict[str, bytes],
    runtimes: dict[str, bytes],
    ledger_bytes: bytes,
    objdump: Path,
    python_host_dll_names: frozenset[str],
    label: str,
) -> None:
    _require(objdump.is_file(), f"MinGW objdump not found: {objdump}")
    all_names = list(roots) + list(runtimes)
    _require(
        len({name.casefold() for name in all_names}) == len(all_names),
        f"{label}: duplicate case-insensitive PE member names",
    )
    ledger = _json_without_duplicate_keys(ledger_bytes, f"{label} runtime ledger")
    _require(
        frozenset(ledger) == frozenset({"format", "roots", "runtimes"}),
        f"{label}: runtime ledger has unexpected top-level fields",
    )
    _require(
        ledger["format"] == _WINDOWS_RUNTIME_FORMAT,
        f"{label}: unsupported runtime ledger format {ledger['format']!r}",
    )
    root_records = _records_by_member(
        ledger["roots"],
        frozenset(
            {"architecture", "dependencies", "member", "member_sha256", "size"}
        ),
        f"{label} roots",
    )
    runtime_records = _records_by_member(
        ledger["runtimes"],
        frozenset(
            {
                "architecture",
                "dependencies",
                "license",
                "member",
                "member_sha256",
                "resolution_method",
                "size",
                "source_path",
                "source_sha256",
            }
        ),
        f"{label} runtimes",
    )
    actual_roots = {name.casefold(): (name, data) for name, data in roots.items()}
    actual_runtimes = {
        name.casefold(): (name, data) for name, data in runtimes.items()
    }
    _require(
        set(root_records) == set(actual_roots),
        f"{label}: ledger roots differ from packaged roots",
    )
    _require(
        set(runtime_records) == set(actual_runtimes),
        f"{label}: ledger DLLs differ from packaged DLLs",
    )
    _require(actual_roots, f"{label}: no PE roots were supplied")
    _require(actual_runtimes, f"{label}: no runtime DLLs were supplied")

    with tempfile.TemporaryDirectory() as tmp:
        inspection_root = Path(tmp)
        paths: dict[str, Path] = {}
        for key, (name, data) in {**actual_roots, **actual_runtimes}.items():
            path = inspection_root / name
            path.write_bytes(data)
            paths[key] = path

        actual_dependencies: dict[str, list[str]] = {}
        for key, record in {**root_records, **runtime_records}.items():
            member_kind = "root" if key in root_records else "runtime"
            record_label = f"{label} {member_kind} {record['member']}"
            actual_name, actual_bytes = (
                actual_roots[key] if key in root_records else actual_runtimes[key]
            )
            _require(
                record["member"] == actual_name,
                f"{record_label}: member case differs from packaged member",
            )
            member_hash = _checked_hash(
                record["member_sha256"], f"{record_label} member_sha256"
            )
            _require(
                member_hash == _sha256_bytes(actual_bytes),
                f"{record_label}: packaged bytes do not match member_sha256",
            )
            _require(
                _checked_size(record["size"], record_label) == len(actual_bytes),
                f"{record_label}: packaged size does not match ledger",
            )
            dependencies = _checked_dependencies(
                record["dependencies"], record_label
            )
            inspected_dependencies = sorted(
                _pe_dependencies(objdump, paths[key]), key=str.casefold
            )
            _require(
                dependencies == inspected_dependencies,
                f"{record_label}: PE dependencies do not match ledger",
            )
            actual_dependencies[key] = inspected_dependencies
            architecture = _pe_architecture(objdump, paths[key])
            _require(
                isinstance(architecture, str)
                and architecture in _PE_X86_64_ARCHITECTURES,
                f"{record_label}: expected PE x86-64, got {architecture}",
            )
            _require(
                record["architecture"] == architecture,
                f"{record_label}: PE architecture does not match ledger",
            )

            if key in runtime_records:
                source_hash = _checked_hash(
                    record["source_sha256"], f"{record_label} source_sha256"
                )
                _require(
                    source_hash == member_hash,
                    f"{record_label}: source and packaged hashes differ",
                )
                _checked_source_path(record["source_path"], actual_name, record_label)
                _require(
                    isinstance(record["resolution_method"], str)
                    and record["resolution_method"]
                    in _WINDOWS_RUNTIME_RESOLUTION_METHODS,
                    f"{record_label}: unsupported resolution method",
                )
                expected_license = _windows_runtime_license(actual_name)
                _require(
                    expected_license is not None
                    and record["license"] == expected_license,
                    f"{record_label}: missing or incorrect license mapping",
                )

        root_dependencies = [
            dependency
            for key in root_records
            for dependency in actual_dependencies[key]
        ]
        _require(
            any(name.casefold() == "libgomp-1.dll" for name in root_dependencies),
            f"{label}: no root is linked to libgomp-1.dll; OpenMP is missing",
        )
        pending = sorted(
            (
                dependency
                for dependency in root_dependencies
                if not _is_windows_host_dll(dependency, python_host_dll_names)
            ),
            key=str.casefold,
        )
        _require(pending, f"{label}: roots have no detected non-system dependency")
        visited: set[str] = set()
        while pending:
            name = pending.pop(0)
            key = name.casefold()
            if key in visited:
                continue
            _require(
                key in actual_runtimes,
                f"{label}: missing required runtime DLL {name}",
            )
            visited.add(key)
            for dependency in actual_dependencies[key]:
                if (
                    not _is_windows_host_dll(dependency, python_host_dll_names)
                    and dependency.casefold() not in visited
                ):
                    pending.append(dependency)
        _require(
            visited == set(actual_runtimes),
            f"{label}: PE graph does not equal packaged DLL set; unreferenced: "
            + ", ".join(sorted(set(actual_runtimes) - visited)),
        )


def _validate_mingw_closure(wheel: Path, objdump: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        pyd_infos = [info for info in infos if info.filename.lower().endswith(".pyd")]
        dll_infos = [info for info in infos if info.filename.lower().endswith(".dll")]
        manifest_infos = [
            info
            for info in infos
            if PurePosixPath(info.filename).name.casefold()
            == _WINDOWS_RUNTIME_MANIFEST.casefold()
        ]
        _require(len(pyd_infos) == 1, f"wheel must contain one .pyd, found {pyd_infos}")
        _require(
            len(manifest_infos) == 1,
            f"wheel must contain one {_WINDOWS_RUNTIME_MANIFEST}, found {manifest_infos}",
        )
        package_parent = PurePosixPath(pyd_infos[0].filename).parent
        _require(
            PurePosixPath(manifest_infos[0].filename).parent == package_parent,
            "wheel runtime ledger must be beside the native extension",
        )
        _require(
            all(PurePosixPath(info.filename).parent == package_parent for info in dll_infos),
            "wheel non-system DLLs must be beside the native extension",
        )
        member_names = [
            PurePosixPath(info.filename).name
            for info in pyd_infos + dll_infos + manifest_infos
        ]
        _require(
            len({name.casefold() for name in member_names}) == len(member_names),
            "wheel contains duplicate case-insensitive PE or ledger members",
        )
        roots = {
            PurePosixPath(info.filename).name: archive.read(info) for info in pyd_infos
        }
        runtimes = {
            PurePosixPath(info.filename).name: archive.read(info) for info in dll_infos
        }
        ledger_bytes = archive.read(manifest_infos[0])
    python_host_dll_names = _python_host_dll_names(
        Path(next(iter(roots)))
    )
    _validate_pe_runtime_ledger(
        roots=roots,
        runtimes=runtimes,
        ledger_bytes=ledger_bytes,
        objdump=objdump,
        python_host_dll_names=python_host_dll_names,
        label="wheel",
    )


def _runtime_dll_paths(runtime_dir: Path) -> list[Path]:
    runtime_paths = sorted(
        (
            path
            for path in runtime_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".dll"
        ),
        key=lambda path: str(path).casefold(),
    )
    _require(
        all(path.parent == runtime_dir for path in runtime_paths),
        "runtime DLLs must be direct children of the runtime directory",
    )
    return runtime_paths


def validate_pe_runtime_directory(
    binaries: list[Path], runtime_dir: Path, ledger: Path, objdump: Path
) -> None:
    _require(runtime_dir.is_dir(), f"runtime directory not found: {runtime_dir}")
    _require(ledger.is_file(), f"runtime ledger not found: {ledger}")
    _require(binaries, "supply at least one PE root binary")
    _require(all(path.is_file() for path in binaries), "a PE root binary is missing")
    root_names = [path.name for path in binaries]
    _require(
        len({name.casefold() for name in root_names}) == len(root_names),
        "PE roots have duplicate case-insensitive basenames",
    )
    runtime_paths = _runtime_dll_paths(runtime_dir)
    roots = {path.name: path.read_bytes() for path in binaries}
    runtimes = {path.name: path.read_bytes() for path in runtime_paths}
    _require(
        len(runtimes) == len(runtime_paths),
        "runtime directory has duplicate case-insensitive DLL names",
    )
    _validate_pe_runtime_ledger(
        roots=roots,
        runtimes=runtimes,
        ledger_bytes=ledger.read_bytes(),
        objdump=objdump,
        python_host_dll_names=frozenset(),
        label="PE runtime directory",
    )


def _resolve_runtime_source(
    dll_name: str,
    explicit_sources: list[Path],
    search_dirs: list[Path],
) -> tuple[Path, str]:
    key = dll_name.casefold()
    explicit_candidates = [
        path for path in explicit_sources if path.name.casefold() == key
    ]
    candidates = list(explicit_candidates)
    for directory in search_dirs:
        candidates.extend(
            path
            for path in directory.iterdir()
            if path.is_file() and path.name.casefold() == key
        )
    canonical_candidates = {path.resolve() for path in candidates}
    _require(
        canonical_candidates,
        f"could not resolve non-system PE dependency {dll_name}",
    )
    candidates_by_hash: dict[str, list[Path]] = {}
    for candidate in canonical_candidates:
        digest = _sha256_bytes(candidate.read_bytes())
        candidates_by_hash.setdefault(digest, []).append(candidate)
    if len(candidates_by_hash) != 1:
        paths = ", ".join(sorted(str(path) for path in canonical_candidates))
        raise RuntimeError(
            f"ambiguous runtime sources with different bytes for {dll_name}: {paths}"
        )
    explicit_canonical = {path.resolve() for path in explicit_candidates}
    source = sorted(
        canonical_candidates,
        key=lambda path: (
            path not in explicit_canonical,
            str(path).casefold(),
            str(path),
        ),
    )[0]
    method = (
        "explicit-runtime-source"
        if source in explicit_canonical
        else "runtime-search-directory"
    )
    return source, method


def build_pe_runtime_ledger(
    binaries: list[Path],
    runtime_sources: list[Path],
    runtime_dir: Path,
    ledger: Path,
    objdump: Path,
    runtime_search_dirs: Optional[list[Path]] = None,
) -> None:
    search_dirs = [path.resolve() for path in (runtime_search_dirs or [])]
    explicit_sources = [path.resolve() for path in runtime_sources]
    _require(objdump.is_file(), f"MinGW objdump not found: {objdump}")
    _require(runtime_dir.is_dir(), f"runtime directory not found: {runtime_dir}")
    _require(ledger.parent.is_dir(), f"ledger parent directory not found: {ledger.parent}")
    _require(binaries, "supply at least one PE root binary")
    _require(
        explicit_sources or search_dirs,
        "supply a runtime source DLL or runtime search directory",
    )
    _require(all(path.is_file() for path in binaries), "a PE root binary is missing")
    _require(
        all(path.is_file() for path in explicit_sources),
        "a runtime source DLL is missing",
    )
    _require(
        all(path.suffix.casefold() == ".dll" for path in explicit_sources),
        "every runtime source must be a DLL",
    )
    _require(
        all(path.is_dir() for path in search_dirs),
        "a runtime search directory is missing",
    )
    root_names = [path.name for path in binaries]
    _require(
        len({name.casefold() for name in root_names}) == len(root_names),
        "PE roots have duplicate case-insensitive basenames",
    )

    root_records = []
    root_bytes: dict[str, bytes] = {}
    root_architectures: set[str] = set()
    pending: list[str] = []
    for binary in sorted(binaries, key=lambda path: path.name.casefold()):
        data = binary.read_bytes()
        architecture = _pe_architecture(objdump, binary)
        _require(
            architecture in _PE_X86_64_ARCHITECTURES,
            f"PE root {binary} is not x86-64: {architecture}",
        )
        root_architectures.add(architecture)
        dependencies = sorted(_pe_dependencies(objdump, binary), key=str.casefold)
        pending.extend(
            dependency
            for dependency in dependencies
            if not _is_windows_host_dll(dependency, frozenset())
        )
        root_bytes[binary.name] = data
        root_records.append(
            {
                "architecture": architecture,
                "dependencies": dependencies,
                "member": binary.name,
                "member_sha256": _sha256_bytes(data),
                "size": len(data),
            }
        )
    _require(
        len(root_architectures) == 1,
        "PE roots do not share one x86-64 architecture",
    )
    root_architecture = next(iter(root_architectures))

    resolved: dict[str, dict[str, object]] = {}
    while pending:
        dll_name = pending.pop(0)
        key = dll_name.casefold()
        if key in resolved:
            continue
        source, resolution_method = _resolve_runtime_source(
            dll_name, explicit_sources, search_dirs
        )
        architecture = _pe_architecture(objdump, source)
        _require(
            architecture == root_architecture,
            f"runtime source {source} has wrong architecture: {architecture}; "
            f"expected {root_architecture}",
        )
        license_expression = _windows_runtime_license(source.name)
        _require(
            license_expression is not None,
            f"runtime source {source.name} has no explicit license mapping",
        )
        dependencies = sorted(_pe_dependencies(objdump, source), key=str.casefold)
        resolved[key] = {
            "architecture": architecture,
            "dependencies": dependencies,
            "license": license_expression,
            "resolution_method": resolution_method,
            "source": source,
        }
        pending.extend(
            dependency
            for dependency in dependencies
            if not _is_windows_host_dll(dependency, frozenset())
        )

    explicit_keys = {path.name.casefold() for path in explicit_sources}
    _require(
        explicit_keys <= set(resolved),
        "explicit runtime sources include unreferenced DLLs: "
        + ", ".join(sorted(explicit_keys - set(resolved))),
    )
    _require(resolved, "PE roots have no detected non-system dependency")

    existing_paths = _runtime_dll_paths(runtime_dir)
    existing_by_key = {path.name.casefold(): path for path in existing_paths}
    _require(
        len(existing_by_key) == len(existing_paths),
        "runtime directory has duplicate case-insensitive DLL names",
    )
    _require(
        set(existing_by_key) <= set(resolved),
        "runtime directory contains unreferenced DLLs: "
        + ", ".join(sorted(set(existing_by_key) - set(resolved))),
    )
    runtime_dir_resolved = runtime_dir.resolve()
    runtime_records = []
    for key in sorted(resolved):
        details = resolved[key]
        source = details["source"]
        source_data = source.read_bytes()
        destination = runtime_dir / source.name
        existing = existing_by_key.get(key)
        if existing is not None:
            _require(
                existing.name == source.name
                and existing.is_file()
                and existing.resolve().parent == runtime_dir_resolved,
                f"runtime destination is not the expected direct file: {existing}",
            )
            _require(
                existing.read_bytes() == source_data,
                f"runtime destination conflicts with source: {existing}",
            )
        else:
            shutil.copy2(source, destination)
        member_data = destination.read_bytes()
        source_hash = _sha256_bytes(source_data)
        member_hash = _sha256_bytes(member_data)
        _require(
            source_hash == member_hash,
            f"runtime destination does not match source: {destination}",
        )
        runtime_records.append(
            {
                "architecture": details["architecture"],
                "dependencies": details["dependencies"],
                "license": details["license"],
                "member": destination.name,
                "member_sha256": member_hash,
                "resolution_method": details["resolution_method"],
                "size": len(member_data),
                "source_path": str(source),
                "source_sha256": source_hash,
            }
        )

    manifest = {
        "format": _WINDOWS_RUNTIME_FORMAT,
        "roots": root_records,
        "runtimes": runtime_records,
    }
    ledger_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    packaged_bytes = {
        path.name: path.read_bytes() for path in _runtime_dll_paths(runtime_dir)
    }
    _validate_pe_runtime_ledger(
        roots=root_bytes,
        runtimes=packaged_bytes,
        ledger_bytes=ledger_bytes,
        objdump=objdump,
        python_host_dll_names=frozenset(),
        label="generated PE runtime directory",
    )
    ledger.write_bytes(ledger_bytes)


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
        _wheel_runtime_license_checks(archive)
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
            "/tests/test_corresponding_source_bundle.py",
            "/tools/build_corresponding_source_bundle.py",
            "/tools/validate_corresponding_source_bundle.py",
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
        for filename, expected_hash in _RUNTIME_LICENSE_HASHES.items():
            license_name = root + "/third_party/licenses/" + filename
            _require(license_name in names, f"sdist is missing {license_name}")
            license_member = archive.extractfile(license_name)
            _require(license_member is not None, f"sdist could not read {license_name}")
            _require(
                hashlib.sha256(license_member.read()).hexdigest() == expected_hash,
                f"sdist {filename} hash mismatch",
            )
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
    parser.add_argument("--expected-version")
    parser.add_argument("--mingw-objdump", type=Path)
    parser.add_argument("--pe-binary", action="append", type=Path, default=[])
    parser.add_argument("--runtime-source", action="append", type=Path, default=[])
    parser.add_argument("--runtime-search-dir", action="append", type=Path, default=[])
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--runtime-ledger", type=Path)
    parser.add_argument("--write-runtime-ledger", type=Path)
    args = parser.parse_args()
    write_mode = bool(
        args.runtime_source
        or args.runtime_search_dir
        or args.write_runtime_ledger is not None
    )
    directory_mode = not write_mode and bool(
        args.pe_binary or args.runtime_dir is not None or args.runtime_ledger is not None
    )
    _require(
        args.wheel is not None or args.sdist is not None or directory_mode or write_mode,
        "supply a wheel, sdist, or PE runtime directory",
    )
    if args.wheel is not None:
        _require(args.expected_version is not None, "wheel validation needs --expected-version")
        validate_wheel(args.wheel, args.expected_version, args.mingw_objdump)
    if args.sdist is not None:
        _require(args.expected_version is not None, "sdist validation needs --expected-version")
        validate_sdist(args.sdist, args.expected_version)
    if directory_mode:
        _require(
            args.pe_binary
            and args.runtime_dir is not None
            and args.runtime_ledger is not None
            and args.mingw_objdump is not None,
            "PE directory validation needs --pe-binary, --runtime-dir, "
            "--runtime-ledger, and --mingw-objdump",
        )
        validate_pe_runtime_directory(
            args.pe_binary,
            args.runtime_dir,
            args.runtime_ledger,
            args.mingw_objdump,
        )
    if write_mode:
        _require(
            args.pe_binary
            and (args.runtime_source or args.runtime_search_dir)
            and args.runtime_dir is not None
            and args.write_runtime_ledger is not None
            and args.mingw_objdump is not None
            and args.runtime_ledger is None,
            "PE ledger generation needs --pe-binary, a --runtime-source or "
            "--runtime-search-dir, --runtime-dir, --write-runtime-ledger, "
            "and --mingw-objdump",
        )
        build_pe_runtime_ledger(
            args.pe_binary,
            args.runtime_source,
            args.runtime_dir,
            args.write_runtime_ledger,
            args.mingw_objdump,
            args.runtime_search_dir,
        )
    print("Python release artifact closure OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
