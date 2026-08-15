#!/usr/bin/env python3
"""Fail-closed validation for an xhdfe corresponding-source ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


SCHEMA_VERSION = 1
REQUIRED_COMPONENTS = {
    "gcc",
    "mingw-w64",
    "dlfcn-win32",
    "winlibs-recipes",
    "winlibs-tools",
}
REQUIRED_PROVIDERS = {
    "ubuntu-mingw-gcc",
    "ubuntu-mingw-w64",
    "manylinux-libgomp",
}
RUNTIME_PROVIDERS = {
    "windows-python-libgcc": "gcc",
    "windows-python-libstdcxx": "gcc",
    "windows-python-libgomp": "gcc",
    "windows-python-libwinpthread": "mingw-w64",
    "windows-python-libdl": "dlfcn-win32",
    "windows-stata-libgomp": "ubuntu-mingw-gcc",
    "windows-stata-libwinpthread": "ubuntu-mingw-w64",
    "manylinux-wheel-libgomp": "manylinux-libgomp",
}
LICENSE_HASHES = {
    "GCC-13.2.0-COPYING3":
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
    "GCC-13.2.0-COPYING.RUNTIME":
        "9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74",
    "mingw-w64-11.0.1-winpthreads-COPYING":
        "63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177",
    "dlfcn-win32-1.4.1-COPYING":
        "4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7",
}
CUDA_LICENSE_HASHES = {
    "NVIDIA-CUDA-12.6-EULA.pdf":
        "7c2dc636ad47cf67a0efb97d9c11246efcc471ac9d11eb8efceae3bfd56d8649",
    "NVIDIA-CCCL-2.5.0-LICENSE":
        "01b767dcd7d36f42efb608076741cf83f154a995e198028cb698aadc3a43b63b",
}
HASH_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
DSC_SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+([0-9]+)\s+(\S+)$")
DEBIAN_SOURCE_PROVIDERS = {"ubuntu-mingw-gcc", "ubuntu-mingw-w64"}


def sha256_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metadata(entry: dict[str, object]) -> dict[str, str]:
    value = entry.get("metadata")
    require(isinstance(value, dict), f"{entry.get('id')}: invalid metadata")
    return value  # type: ignore[return-value]


def dsc_sha256_records(payload: bytes, label: str) -> list[tuple[str, int, str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label}: DSC is not UTF-8 text") from error
    starts = [index for index, line in enumerate(lines) if line == "Checksums-Sha256:"]
    require(len(starts) == 1, f"{label}: expected exactly one Checksums-Sha256 field")
    records: list[tuple[str, int, str]] = []
    names: set[str] = set()
    for line in lines[starts[0] + 1 :]:
        if not line.startswith((" ", "\t")):
            break
        match = DSC_SHA256_LINE.fullmatch(line.strip())
        require(match is not None, f"{label}: malformed Checksums-Sha256 line: {line!r}")
        digest, size_text, name = match.groups()
        require(
            PurePosixPath(name).name == name and "/" not in name and "\\" not in name,
            f"{label}: unsafe DSC payload name: {name!r}",
        )
        require(name not in names, f"{label}: duplicate DSC payload name: {name}")
        names.add(name)
        records.append((digest.lower(), int(size_text), name))
    require(bool(records), f"{label}: empty Checksums-Sha256 field")
    return records


def validate_debian_source_entry(
    archive: zipfile.ZipFile, root: str, entry: dict[str, object]
) -> None:
    provider_id = str(entry["id"])
    bundle_path = str(entry["bundle_path"])
    records = entry["files"]
    require(isinstance(records, list), f"{provider_id}: invalid source files")
    record_map = {
        str(record["path"]): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    require(len(record_map) == len(records), f"{provider_id}: invalid source records")
    dsc_paths = sorted(path for path in record_map if path.endswith(".dsc"))
    require(bool(dsc_paths), f"{provider_id}: provider source contains no DSC")
    closed = set(dsc_paths)
    for dsc_path in dsc_paths:
        label = f"{provider_id}/{dsc_path}"
        member = f"{root}/{bundle_path}/{dsc_path}"
        for expected_hash, expected_size, name in dsc_sha256_records(
            archive.read(member), label
        ):
            relative = (PurePosixPath(dsc_path).parent / name).as_posix()
            record = record_map.get(relative)
            require(record is not None, f"{label}: missing Debian source payload: {name}")
            require(
                record.get("sha256") == expected_hash,
                f"{label}: Debian source payload hash mismatch: {name}",
            )
            require(
                record.get("size") == expected_size,
                f"{label}: Debian source payload size mismatch: {name}",
            )
            closed.add(relative)
    require(
        set(record_map) == closed,
        f"{provider_id}: DSC source closure mismatch",
    )


def validate_entry_files(
    archive: zipfile.ZipFile,
    root: str,
    entry: dict[str, object],
    expected_bundle_path: str,
    referenced: set[str],
) -> None:
    bundle_path = entry.get("bundle_path")
    records = entry.get("files")
    require(bundle_path == expected_bundle_path, f"{entry.get('id')}: invalid bundle path")
    require(isinstance(records, list) and bool(records), f"{entry.get('id')}: no files")
    tree = hashlib.sha256()
    relative_paths: set[str] = set()
    for record in records:
        require(isinstance(record, dict), f"{entry.get('id')}: invalid file record")
        relative = record.get("path")
        expected = record.get("sha256")
        size = record.get("size")
        require(isinstance(relative, str) and safe_member(relative), "unsafe source path")
        require(relative not in relative_paths, f"duplicate source record: {relative}")
        relative_paths.add(relative)
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"invalid source hash: {relative}",
        )
        require(type(size) is int and size >= 0, f"invalid source size: {relative}")
        member = f"{root}/{bundle_path}/{relative}"
        try:
            info = archive.getinfo(member)
        except KeyError as error:
            raise ValueError(f"missing source payload member: {member}") from error
        require(member not in referenced, f"source member referenced twice: {member}")
        referenced.add(member)
        actual = sha256_member(archive, member)
        require(actual == expected, f"source hash mismatch: {member}")
        require(info.file_size == size, f"source size mismatch: {member}")
        tree.update(f"{actual}  {relative}\n".encode("utf-8"))
    require(
        tree.hexdigest() == entry.get("tree_sha256"),
        f"{entry.get('id')}: source tree hash mismatch",
    )


def validate(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing corresponding-source archive: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP member names")
        require(all(safe_member(name) for name in names), "unsafe ZIP member path")
        roots = {PurePosixPath(name).parts[0] for name in names}
        require(len(roots) == 1, "archive must have exactly one root directory")
        root = next(iter(roots))
        require(
            re.fullmatch(r"xhdfe-[0-9A-Za-z.+-]+-corresponding-source", root)
            is not None,
            f"unexpected archive root: {root}",
        )

        provenance = json.loads(
            archive.read(f"{root}/PROVENANCE.json"),
            object_pairs_hook=no_duplicate_json_keys,
        )
        require(provenance.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
        require(provenance.get("package") == "xhdfe", "package mismatch")
        require(root == f"xhdfe-{provenance.get('version')}-corresponding-source", "version/root mismatch")

        manifest_data = archive.read(f"{root}/MANIFEST.sha256").decode("utf-8")
        manifest: dict[str, str] = {}
        for line in manifest_data.splitlines():
            match = HASH_LINE.fullmatch(line)
            require(match is not None, f"malformed MANIFEST.sha256 line: {line!r}")
            digest, relative = match.groups()
            require(safe_member(relative), f"unsafe manifest path: {relative}")
            require(relative not in manifest, f"duplicate manifest path: {relative}")
            manifest[relative] = digest
        actual_members = {
            name[len(root) + 1 :]
            for name in names
            if name.startswith(root + "/") and name != f"{root}/MANIFEST.sha256"
        }
        require(set(manifest) == actual_members, "manifest does not exactly close ZIP files")
        for relative, expected in manifest.items():
            require(
                sha256_member(archive, f"{root}/{relative}") == expected,
                f"manifest hash mismatch: {relative}",
            )

        components = provenance.get("components")
        providers = provenance.get("provider_sources")
        runtimes = provenance.get("runtime_binaries")
        require(isinstance(components, list), "components ledger missing")
        require(isinstance(providers, list), "provider source ledger missing")
        require(isinstance(runtimes, list), "runtime ledger missing")
        for label, entries in (
            ("component", components),
            ("provider", providers),
            ("runtime", runtimes),
        ):
            for entry in entries:
                require(isinstance(entry, dict), f"invalid {label} ledger entry")
                object_id = entry.get("id")
                require(
                    isinstance(object_id, str) and SAFE_ID.fullmatch(object_id) is not None,
                    f"invalid {label} ID: {object_id!r}",
                )
        component_map = {entry["id"]: entry for entry in components}
        provider_map = {entry["id"]: entry for entry in providers}
        runtime_map = {entry["id"]: entry for entry in runtimes}
        require(len(component_map) == len(components), "duplicate source component IDs")
        require(len(provider_map) == len(providers), "duplicate provider source IDs")
        require(len(runtime_map) == len(runtimes), "duplicate runtime evidence IDs")
        require(
            not (set(component_map) & set(provider_map)),
            "component/provider source IDs overlap",
        )
        require(REQUIRED_COMPONENTS <= set(component_map), "source component closure incomplete")
        require(REQUIRED_PROVIDERS <= set(provider_map), "provider-source closure incomplete")
        require(set(RUNTIME_PROVIDERS) <= set(runtime_map), "runtime evidence closure incomplete")

        referenced_sources: set[str] = set()
        for entry in components:
            validate_entry_files(
                archive,
                root,
                entry,
                f"sources/components/{entry['id']}",
                referenced_sources,
            )
        for entry in providers:
            validate_entry_files(
                archive,
                root,
                entry,
                f"sources/providers/{entry['id']}",
                referenced_sources,
            )
            if entry["id"] in DEBIAN_SOURCE_PROVIDERS:
                validate_debian_source_entry(archive, root, entry)
        actual_sources = {
            name for name in names if name.startswith(f"{root}/sources/")
        }
        require(
            referenced_sources == actual_sources,
            "source payload files are not exactly closed by PROVENANCE.json",
        )

        pinned_versions = {"gcc": "13.2.0", "mingw-w64": "11.0.1", "dlfcn-win32": "1.4.1"}
        for object_id, version in pinned_versions.items():
            item_metadata = metadata(component_map[object_id])
            require(item_metadata.get("version") == version, f"{object_id}: version mismatch")
            require(bool(item_metadata.get("upstream_url")), f"{object_id}: upstream URL missing")
        for object_id in {"winlibs-recipes", "winlibs-tools"}:
            item_metadata = metadata(component_map[object_id])
            require(bool(item_metadata.get("commit")), f"{object_id}: commit missing")
            require(bool(item_metadata.get("upstream_url")), f"{object_id}: upstream URL missing")
        for object_id in {"ubuntu-mingw-gcc", "ubuntu-mingw-w64"}:
            item_metadata = metadata(provider_map[object_id])
            for field in ("source_package", "source_version", "upstream_url"):
                require(bool(item_metadata.get(field)), f"{object_id}: {field} missing")
        manylinux = metadata(provider_map["manylinux-libgomp"])
        for field in ("nevra", "sourcerpm", "provider_path", "upstream_url"):
            require(bool(manylinux.get(field)), f"manylinux-libgomp: {field} missing")

        known_sources = set(component_map) | set(provider_map)
        for runtime_id, entry in runtime_map.items():
            require(isinstance(entry, dict), f"{runtime_id}: invalid runtime entry")
            require(entry.get("provider_id") in known_sources, f"{runtime_id}: unknown provider")
            if runtime_id in RUNTIME_PROVIDERS:
                require(
                    entry.get("provider_id") == RUNTIME_PROVIDERS[runtime_id],
                    f"{runtime_id}: provider mismatch",
                )
            for field in (
                "release_path",
                "packaged_sha256",
                "packaged_size",
                "provider_binary_path",
                "provider_binary_sha256",
                "provider_binary_size",
                "byte_identical_to_provider",
            ):
                require(field in entry, f"{runtime_id}: {field} missing")
            require(
                isinstance(entry["release_path"], str) and bool(entry["release_path"]),
                f"{runtime_id}: invalid release path",
            )
            require(
                isinstance(entry["provider_binary_path"], str)
                and bool(entry["provider_binary_path"]),
                f"{runtime_id}: invalid provider binary path",
            )
            require(
                re.fullmatch(r"[0-9a-f]{64}", str(entry["packaged_sha256"])) is not None,
                f"{runtime_id}: invalid packaged hash",
            )
            require(
                re.fullmatch(r"[0-9a-f]{64}", str(entry["provider_binary_sha256"])) is not None,
                f"{runtime_id}: invalid provider hash",
            )
            require(
                type(entry["packaged_size"]) is int and entry["packaged_size"] > 0,
                f"{runtime_id}: invalid packaged size",
            )
            require(
                type(entry["provider_binary_size"]) is int
                and entry["provider_binary_size"] > 0,
                f"{runtime_id}: invalid provider size",
            )
            require(
                isinstance(entry["byte_identical_to_provider"], bool),
                f"{runtime_id}: invalid byte-identity field",
            )
            require(
                entry["byte_identical_to_provider"]
                is (entry["packaged_sha256"] == entry["provider_binary_sha256"]),
                f"{runtime_id}: byte-identity claim contradicts recorded hashes",
            )
            if runtime_id != "manylinux-wheel-libgomp":
                require(entry["byte_identical_to_provider"] is True, f"{runtime_id}: provider bytes differ")

        cuda = provenance.get("cuda")
        require(isinstance(cuda, dict), "CUDA scope ledger missing")
        require(isinstance(cuda.get("included"), bool), "invalid CUDA included flag")
        expected_licenses = dict(LICENSE_HASHES)
        if cuda.get("included") is True:
            require(cuda.get("toolkit") == "CUDA 12.6", "CUDA toolkit version mismatch")
            require(
                cuda.get("redistributable_static_components")
                == ["libcudadevrt", "libcudart_static"],
                "CUDA static-runtime scope mismatch",
            )
            require(
                cuda.get("compile_time_inputs") == ["libdevice.10.bc", "CUB/CCCL 2.5.0"],
                "CUDA compile-time input scope mismatch",
            )
            expected_licenses.update(CUDA_LICENSE_HASHES)
        for name, expected in expected_licenses.items():
            member = f"{root}/licenses/{name}"
            require(member in names, f"missing required license: {name}")
            require(sha256_member(archive, member) == expected, f"license hash mismatch: {name}")
        actual_licenses = {
            name for name in names if name.startswith(f"{root}/licenses/")
        }
        expected_license_members = {
            f"{root}/licenses/{name}" for name in expected_licenses
        }
        require(
            actual_licenses == expected_license_members,
            "license directory contains an unledgered or missing file",
        )
        license_records = provenance.get("license_files")
        require(isinstance(license_records, list), "license ledger missing")
        license_map = {
            record.get("path"): record
            for record in license_records
            if isinstance(record, dict)
        }
        require(
            len(license_map) == len(license_records),
            "invalid or duplicate license ledger records",
        )
        require(
            set(license_map) == {f"licenses/{name}" for name in expected_licenses},
            "license ledger does not exactly close required licenses",
        )
        for relative, record in license_map.items():
            name = PurePosixPath(relative).name
            info = archive.getinfo(f"{root}/{relative}")
            require(record.get("sha256") == expected_licenses[name], f"license ledger hash mismatch: {name}")
            require(record.get("size") == info.file_size, f"license ledger size mismatch: {name}")

    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        provenance = validate(args.archive)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(
        "Corresponding-source closure OK: "
        f"xhdfe {provenance['version']} | "
        f"{len(provenance['components'])} upstream components | "
        f"{len(provenance['provider_sources'])} provider sources | "
        f"{len(provenance['runtime_binaries'])} runtime mappings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
