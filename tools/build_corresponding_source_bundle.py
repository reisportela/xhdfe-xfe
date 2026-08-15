#!/usr/bin/env python3
"""Build the release corresponding-source bundle from local, explicit inputs.

This tool never downloads anything.  The release job must supply the exact
upstream source archives/checkouts, distribution source packages, and runtime
binaries that correspond to the binaries being released.  The resulting ZIP
contains source bytes plus a machine-readable provenance ledger; runtime
binaries are hashed for linkage evidence but remain in their normal release
assets rather than being duplicated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
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
PINNED_COMPONENT_VERSIONS = {
    "gcc": "13.2.0",
    "mingw-w64": "11.0.1",
    "dlfcn-win32": "1.4.1",
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
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]{8}$")
DSC_SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+([0-9]+)\s+(\S+)$")
DEBIAN_SOURCE_PROVIDERS = {"ubuntu-mingw-gcc", "ubuntu-mingw-w64"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assignments(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires NAME=VALUE: {value!r}")
        name, item = value.split("=", 1)
        if not SAFE_ID.fullmatch(name):
            raise ValueError(f"unsafe identifier in {option}: {name!r}")
        if not item:
            raise ValueError(f"empty value in {option}: {name!r}")
        if name in parsed:
            raise ValueError(f"duplicate {option} identifier: {name}")
        parsed[name] = item
    return parsed


def parse_metadata(values: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for value in values:
        if "=" not in value or "." not in value.split("=", 1)[0]:
            raise ValueError(
                f"--metadata requires OBJECT.FIELD=VALUE: {value!r}"
            )
        key, item = value.split("=", 1)
        object_id, field = key.split(".", 1)
        if not SAFE_ID.fullmatch(object_id) or not SAFE_ID.fullmatch(field):
            raise ValueError(f"unsafe metadata key: {key!r}")
        if not item:
            raise ValueError(f"empty metadata value: {key!r}")
        fields = result.setdefault(object_id, {})
        if field in fields:
            raise ValueError(f"duplicate metadata key: {key}")
        fields[field] = item
    return result


def checked_input(path_text: str) -> Path:
    path = Path(path_text)
    if not path.exists():
        raise ValueError(f"missing input: {path}")
    if path.is_symlink():
        raise ValueError(f"symbolic-link input is not allowed: {path}")
    if not (path.is_file() or path.is_dir()):
        raise ValueError(f"input is not a regular file or directory: {path}")
    return path


def dsc_sha256_records(payload: bytes, label: str) -> list[tuple[str, int, str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label}: DSC is not UTF-8 text") from error
    starts = [index for index, line in enumerate(lines) if line == "Checksums-Sha256:"]
    if len(starts) != 1:
        raise ValueError(f"{label}: expected exactly one Checksums-Sha256 field")
    records: list[tuple[str, int, str]] = []
    names: set[str] = set()
    for line in lines[starts[0] + 1 :]:
        if not line.startswith((" ", "\t")):
            break
        match = DSC_SHA256_LINE.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"{label}: malformed Checksums-Sha256 line: {line!r}")
        digest, size_text, name = match.groups()
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"{label}: unsafe DSC payload name: {name!r}")
        if name in names:
            raise ValueError(f"{label}: duplicate DSC payload name: {name}")
        names.add(name)
        records.append((digest.lower(), int(size_text), name))
    if not records:
        raise ValueError(f"{label}: empty Checksums-Sha256 field")
    return records


def validate_debian_source_directory(source: Path, provider_id: str) -> None:
    if not source.is_dir():
        raise ValueError(f"{provider_id}: Debian provider source must be a directory")
    files = sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix(),
    )
    dsc_files = [item for item in files if item.suffix == ".dsc"]
    if not dsc_files:
        raise ValueError(f"{provider_id}: provider source contains no DSC")
    closed = {item.relative_to(source).as_posix() for item in dsc_files}
    for dsc in dsc_files:
        label = f"{provider_id}/{dsc.relative_to(source).as_posix()}"
        for expected_hash, expected_size, name in dsc_sha256_records(
            dsc.read_bytes(), label
        ):
            payload = dsc.parent / name
            relative = payload.relative_to(source).as_posix()
            if not payload.is_file():
                raise ValueError(f"{label}: missing Debian source payload: {name}")
            if payload.stat().st_size != expected_size:
                raise ValueError(f"{label}: Debian source payload size mismatch: {name}")
            if sha256(payload) != expected_hash:
                raise ValueError(f"{label}: Debian source payload hash mismatch: {name}")
            closed.add(relative)
    actual = {item.relative_to(source).as_posix() for item in files}
    if actual != closed:
        extras = sorted(actual - closed)
        missing = sorted(closed - actual)
        raise ValueError(
            f"{provider_id}: DSC source closure mismatch; "
            f"unreferenced={extras}, missing={missing}"
        )


def copy_payload(source: Path, destination: Path) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=False)
    if source.is_file():
        files = [(source, Path(source.name))]
    else:
        files = []
        for item in sorted(source.rglob("*"), key=lambda p: p.as_posix()):
            if item.is_symlink():
                raise ValueError(f"symbolic link in source payload: {item}")
            if item.is_file():
                files.append((item, item.relative_to(source)))
            elif not item.is_dir():
                raise ValueError(f"special file in source payload: {item}")
        if not files:
            raise ValueError(f"empty source directory: {source}")

    records: list[dict[str, object]] = []
    for item, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target)
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(target),
                "size": target.stat().st_size,
            }
        )
    return records


def tree_sha256(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['sha256']}  {record['path']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def require_metadata(
    metadata: dict[str, dict[str, str]], object_id: str, fields: set[str]
) -> None:
    missing = fields - set(metadata.get(object_id, {}))
    if missing:
        raise ValueError(
            f"{object_id}: missing metadata fields: {', '.join(sorted(missing))}"
        )


def make_entry(
    kind: str,
    object_id: str,
    source: Path,
    stage_root: Path,
    metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    relative = Path("sources") / kind / object_id
    records = copy_payload(source, stage_root / relative)
    return {
        "id": object_id,
        "bundle_path": relative.as_posix(),
        "files": records,
        "tree_sha256": tree_sha256(records),
        "metadata": metadata.get(object_id, {}),
    }


def add_licenses(
    license_dir: Path, stage_root: Path, contains_cuda: bool
) -> list[dict[str, object]]:
    expected = dict(LICENSE_HASHES)
    if contains_cuda:
        expected.update(CUDA_LICENSE_HASHES)
    destination = stage_root / "licenses"
    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for name, expected_hash in sorted(expected.items()):
        source = checked_input(str(license_dir / name))
        actual = sha256(source)
        if actual != expected_hash:
            raise ValueError(
                f"license hash mismatch for {name}: {actual} != {expected_hash}"
            )
        target = destination / name
        shutil.copyfile(source, target)
        records.append(
            {"path": f"licenses/{name}", "sha256": actual, "size": target.stat().st_size}
        )
    return records


def validate_release_inputs(
    components: dict[str, str],
    providers: dict[str, str],
    runtimes: dict[str, str],
    runtime_providers: dict[str, str],
    provider_binaries: dict[str, str],
    metadata: dict[str, dict[str, str]],
) -> None:
    overlap = set(components) & set(providers)
    if overlap:
        raise ValueError(
            "component/provider source identifiers overlap: "
            + ", ".join(sorted(overlap))
        )
    missing_components = REQUIRED_COMPONENTS - set(components)
    missing_providers = REQUIRED_PROVIDERS - set(providers)
    missing_runtimes = set(RUNTIME_PROVIDERS) - set(runtimes)
    if missing_components:
        raise ValueError(
            "missing source components: " + ", ".join(sorted(missing_components))
        )
    if missing_providers:
        raise ValueError(
            "missing provider sources: " + ", ".join(sorted(missing_providers))
        )
    if missing_runtimes:
        raise ValueError(
            "missing runtime evidence: " + ", ".join(sorted(missing_runtimes))
        )
    if set(runtime_providers) != set(runtimes):
        raise ValueError("every --runtime-binary needs exactly one --runtime-provider")
    if set(provider_binaries) != set(runtimes):
        raise ValueError("every --runtime-binary needs exactly one --provider-binary")

    for runtime_id, expected_provider in RUNTIME_PROVIDERS.items():
        actual_provider = runtime_providers.get(runtime_id)
        if actual_provider != expected_provider:
            raise ValueError(
                f"{runtime_id}: provider {actual_provider!r}; expected {expected_provider!r}"
            )
    known_sources = set(components) | set(providers)
    for runtime_id, provider_id in runtime_providers.items():
        if provider_id not in known_sources:
            raise ValueError(f"{runtime_id}: unknown provider/source {provider_id}")

    for component_id, version in PINNED_COMPONENT_VERSIONS.items():
        require_metadata(metadata, component_id, {"version", "upstream_url"})
        if metadata[component_id]["version"] != version:
            raise ValueError(
                f"{component_id}: expected version {version}, got "
                f"{metadata[component_id]['version']}"
            )
    for component_id in {"winlibs-recipes", "winlibs-tools"}:
        require_metadata(metadata, component_id, {"commit", "upstream_url"})
    for provider_id in {"ubuntu-mingw-gcc", "ubuntu-mingw-w64"}:
        require_metadata(
            metadata, provider_id, {"source_package", "source_version", "upstream_url"}
        )
    require_metadata(
        metadata,
        "manylinux-libgomp",
        {"nevra", "sourcerpm", "provider_path", "upstream_url"},
    )
    for runtime_id in runtimes:
        require_metadata(metadata, runtime_id, {"release_path", "provider_path"})


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def build_archive(args: argparse.Namespace) -> None:
    if not VERSION.fullmatch(args.version):
        raise ValueError(f"invalid release version: {args.version!r}")
    components = parse_assignments(args.component, "--component")
    providers = parse_assignments(args.provider_source, "--provider-source")
    runtimes = parse_assignments(args.runtime_binary, "--runtime-binary")
    runtime_providers = parse_assignments(
        args.runtime_provider, "--runtime-provider"
    )
    provider_binaries = parse_assignments(
        args.provider_binary, "--provider-binary"
    )
    metadata = parse_metadata(args.metadata)
    validate_release_inputs(
        components,
        providers,
        runtimes,
        runtime_providers,
        provider_binaries,
        metadata,
    )
    for provider_id in sorted(DEBIAN_SOURCE_PROVIDERS):
        validate_debian_source_directory(
            checked_input(providers[provider_id]), provider_id
        )

    output = Path(args.output).resolve()
    if output.exists() and not args.force:
        raise ValueError(f"output already exists (use --force): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    license_dir = Path(args.license_dir).resolve()
    root_name = f"xhdfe-{args.version}-corresponding-source"

    with tempfile.TemporaryDirectory(prefix="xhdfe-corresponding-source-") as temp:
        stage_root = Path(temp) / root_name
        stage_root.mkdir()

        component_entries = [
            make_entry(
                "components", object_id, checked_input(path), stage_root, metadata
            )
            for object_id, path in sorted(components.items())
        ]
        provider_entries = [
            make_entry(
                "providers", object_id, checked_input(path), stage_root, metadata
            )
            for object_id, path in sorted(providers.items())
        ]
        license_entries = add_licenses(license_dir, stage_root, args.contains_cuda)

        runtime_entries: list[dict[str, object]] = []
        for runtime_id, path_text in sorted(runtimes.items()):
            packaged = checked_input(path_text)
            provider_binary = checked_input(provider_binaries[runtime_id])
            packaged_hash = sha256(packaged)
            provider_hash = sha256(provider_binary)
            runtime_entries.append(
                {
                    "id": runtime_id,
                    "release_path": metadata[runtime_id]["release_path"],
                    "packaged_name": packaged.name,
                    "packaged_sha256": packaged_hash,
                    "packaged_size": packaged.stat().st_size,
                    "provider_id": runtime_providers[runtime_id],
                    "provider_binary_path": metadata[runtime_id]["provider_path"],
                    "provider_binary_sha256": provider_hash,
                    "provider_binary_size": provider_binary.stat().st_size,
                    "byte_identical_to_provider": packaged_hash == provider_hash,
                }
            )

        for entry in runtime_entries:
            if entry["id"] != "manylinux-wheel-libgomp" and not entry[
                "byte_identical_to_provider"
            ]:
                raise ValueError(
                    f"{entry['id']}: packaged runtime is not an unmodified provider copy"
                )

        provenance = {
            "schema_version": SCHEMA_VERSION,
            "package": "xhdfe",
            "version": args.version,
            "components": component_entries,
            "provider_sources": provider_entries,
            "runtime_binaries": runtime_entries,
            "license_files": license_entries,
            "cuda": {
                "included": bool(args.contains_cuda),
                "toolkit": "CUDA 12.6" if args.contains_cuda else None,
                "redistributable_static_components": (
                    ["libcudadevrt", "libcudart_static"]
                    if args.contains_cuda
                    else []
                ),
                "compile_time_inputs": (
                    ["libdevice.10.bc", "CUB/CCCL 2.5.0"]
                    if args.contains_cuda
                    else []
                ),
                "corresponding_source_scope": (
                    "NVIDIA components are governed by the CUDA EULA / CCCL "
                    "license and are not GPL Corresponding Source."
                    if args.contains_cuda
                    else None
                ),
            },
        }
        write_text(
            stage_root / "PROVENANCE.json",
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        )
        write_text(
            stage_root / "README.md",
            f"""# xhdfe {args.version} corresponding source

This release asset contains the exact source payloads and provider source
packages supplied to the release job for redistributed GNU/MinGW runtime
binaries. `PROVENANCE.json` maps each released runtime hash to its provider
binary and source payload. `MANIFEST.sha256` authenticates every file here.

The builder performs no downloads: every source and evidence file is an
explicit local input. Host paths in the ledger identify the CI provider binary
that was hashed; the source payload itself is stored below `sources/`.

The CUDA plugin incorporates NVIDIA redistributable static runtime code and
compile-time inputs. Those NVIDIA components are governed by the CUDA 12.6
EULA and the CCCL 2.5.0 license included under `licenses/`; they are not GPL
Corresponding Source. The GNU/MinGW source obligations are closed independently.
""",
        )

        manifest_lines = []
        for item in sorted(stage_root.rglob("*"), key=lambda p: p.as_posix()):
            if item.is_file() and item.name != "MANIFEST.sha256":
                relative = item.relative_to(stage_root).as_posix()
                manifest_lines.append(f"{sha256(item)}  {relative}\n")
        write_text(stage_root / "MANIFEST.sha256", "".join(manifest_lines))

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary_output = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary_output,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                for item in sorted(stage_root.rglob("*"), key=lambda p: p.as_posix()):
                    if not item.is_file():
                        continue
                    relative = item.relative_to(stage_root.parent).as_posix()
                    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    with item.open("rb") as source, archive.open(info, "w") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            os.replace(temporary_output, output)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()

    validator = Path(__file__).with_name("validate_corresponding_source_bundle.py")
    subprocess.run([sys.executable, str(validator), str(output)], check=True)
    print(f"Wrote {output} ({output.stat().st_size} bytes; SHA-256 {sha256(output)})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Required source IDs:
  components: gcc, mingw-w64, dlfcn-win32, winlibs-recipes, winlibs-tools
  providers:  ubuntu-mingw-gcc, ubuntu-mingw-w64, manylinux-libgomp

The eight release runtime IDs and their fixed provider mappings are recorded in
RUNTIME_PROVIDERS near the top of this script. Extra runtimes are allowed only
with --runtime-provider, --provider-binary, release_path/provider_path metadata,
and a provider ID present in the supplied component/provider source closure.
""",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="pinned upstream source archive or checkout (repeatable)",
    )
    parser.add_argument(
        "--provider-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="distribution source-package file/directory (repeatable)",
    )
    parser.add_argument(
        "--runtime-binary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="released runtime byte sequence to hash (repeatable)",
    )
    parser.add_argument(
        "--runtime-provider",
        action="append",
        default=[],
        metavar="RUNTIME=SOURCE",
        help="map runtime evidence to a component/provider source",
    )
    parser.add_argument(
        "--provider-binary",
        action="append",
        default=[],
        metavar="RUNTIME=PATH",
        help="compiler/provider runtime byte sequence and path",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="OBJECT.FIELD=VALUE",
        help="pinned version, URL, package identity, or release path",
    )
    parser.add_argument("--contains-cuda", action="store_true")
    parser.add_argument(
        "--license-dir",
        default=str(Path(__file__).resolve().parents[1] / "third_party" / "licenses"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        build_archive(args)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
