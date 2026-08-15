#!/usr/bin/env python3
"""Fail-closed provenance gates for the manylinux/CUDA release build.

This helper has three deliberately narrow commands:

* ``nvcc-wrapper`` records every nvcc argv and captures ``nvcc --dryrun``
  output for link invocations before executing the real command unchanged;
* ``cuda-ledger`` inventories the exact static CUDA inputs used by the Stata
  plugin build and rejects unexpected NVIDIA link dependencies; and
* ``wheel-ledger`` inventories every private library grafted by auditwheel and
  ties the expected libgomp payload to its installed RPM and exact source RPM.

It is intended to run inside the pinned manylinux release container.  Missing
files, ambiguous providers, incomplete traces, or unexpected libraries are
release failures rather than best-effort warnings.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import zipfile


SCHEMA_VERSION = "xhdfe-linux-release-provenance-v1"
_NVIDIA_LIBRARY_PREFIXES = (
    "cuda",
    "cublas",
    "cufft",
    "cufile",
    "culibos",
    "cupti",
    "curand",
    "cusolver",
    "cusparse",
    "npp",
    "nv",
)
_ALLOWED_STATIC_CUDA_LIBRARIES = {"cudadevrt", "cudart_static"}


class ProvenanceError(RuntimeError):
    """A release-provenance invariant was not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if check and completed.returncode != 0:
        rendered = " ".join(argv)
        raise ProvenanceError(
            f"command failed ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _rpm_info_for_path(path: Path) -> dict[str, str]:
    canonical = path.resolve(strict=True)
    query = "%{NAME}\t%{EPOCHNUM}\t%{VERSION}\t%{RELEASE}\t%{ARCH}\t%{SOURCERPM}"
    result = _run(["rpm", "-qf", "--qf", query, str(canonical)])
    fields = result.stdout.strip().split("\t")
    if len(fields) != 6 or any(not field for field in fields):
        raise ProvenanceError(f"incomplete RPM ownership metadata for {canonical}")
    name, epoch, version, release, arch, source_rpm = fields
    epoch_prefix = "" if epoch in {"0", "(none)", "(null)"} else f"{epoch}:"
    return {
        "name": name,
        "epoch": epoch,
        "version": version,
        "release": release,
        "arch": arch,
        "nevra": f"{name}-{epoch_prefix}{version}-{release}.{arch}",
        "source_rpm": source_rpm,
    }


def _file_record(path: Path) -> dict[str, Any]:
    canonical = path.resolve(strict=True)
    if not canonical.is_file():
        raise ProvenanceError(f"not a regular file: {canonical}")
    return {
        "source_path": str(canonical),
        "sha256": _sha256_file(canonical),
        "size": canonical.stat().st_size,
        "rpm": _rpm_info_for_path(canonical),
    }


def _unique_existing(paths: Iterable[Path], label: str) -> Path:
    canonical: dict[str, Path] = {}
    for candidate in paths:
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            canonical[str(resolved)] = resolved
    if len(canonical) != 1:
        listed = ", ".join(sorted(canonical)) or "none"
        raise ProvenanceError(f"expected one canonical {label}; found: {listed}")
    return next(iter(canonical.values()))


def _cuda_component(cuda_root: Path, relative_candidates: list[str], label: str) -> Path:
    candidates = [cuda_root / relative for relative in relative_candidates]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        # Fail closed on ambiguity but tolerate an architecture directory name
        # changing in a future CUDA 12.x RPM layout.
        basename = Path(relative_candidates[0]).name
        existing = list(cuda_root.rglob(basename))
    return _unique_existing(existing, label)


def _readelf_needed(path: Path) -> list[str]:
    result = _run(["readelf", "--wide", "--dynamic", str(path)])
    # Parse the stable ELF tag, not readelf's locale-dependent description.
    needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout)
    return sorted(set(needed))


def _is_nvidia_library(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith("lib"):
        lowered = lowered[3:]
    return lowered.startswith(_NVIDIA_LIBRARY_PREFIXES)


def _library_tokens(text: str) -> set[str]:
    names: set[str] = set()
    for linked, archived in re.findall(
        r"(?:^|[\s\"'])(?:-l([A-Za-z0-9_+.-]+)|(?:[^\s\"']*/)?lib([A-Za-z0-9_+.-]+)\.(?:a|so(?:\.[0-9.]+)*))",
        text,
        flags=re.MULTILINE,
    ):
        name = linked or archived
        if name:
            names.add(name)
    # Also cover a future nvcc trace that spells an archive/shared object by
    # direct path without the conventional ``lib`` filename prefix.
    for direct in re.findall(
        r"(?:^|[\s\"'])(?:[^\s\"']*/)?([A-Za-z0-9_+.-]+)\.(?:a|so(?:\.[0-9.]+)*)",
        text,
        flags=re.MULTILINE,
    ):
        names.add(direct[3:] if direct.startswith("lib") else direct)
    return names


def _nvcc_wrapper(arguments: list[str]) -> int:
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments:
        raise ProvenanceError("nvcc-wrapper received no nvcc arguments")

    real_nvcc = Path(os.environ["XHDFE_REAL_NVCC"]).resolve(strict=True)
    trace_path = Path(os.environ["XHDFE_NVCC_INVOCATIONS"])
    dryrun_path = Path(os.environ["XHDFE_NVCC_LINK_DRYRUN"])
    compile_only = any(
        argument in {"-c", "--compile", "-dc", "--device-c", "-dw", "--device-w"}
        for argument in arguments
    )
    mode = "compile" if compile_only else "link"
    record = {
        "argv": [str(real_nvcc), *arguments],
        "cwd": os.getcwd(),
        "mode": mode,
        "real_nvcc": str(real_nvcc),
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)

    if mode == "link":
        dryrun = _run([str(real_nvcc), "--dryrun", *arguments], check=False)
        dryrun_path.parent.mkdir(parents=True, exist_ok=True)
        with dryrun_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write("=== XHDFE_NVCC_LINK_DRYRUN_BEGIN ===\n")
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
            handle.write(f"returncode={dryrun.returncode}\n")
            handle.write("--- stdout ---\n")
            handle.write(dryrun.stdout)
            handle.write("--- stderr ---\n")
            handle.write(dryrun.stderr)
            handle.write("=== XHDFE_NVCC_LINK_DRYRUN_END ===\n")
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
        if dryrun.returncode != 0:
            raise ProvenanceError(
                f"nvcc --dryrun failed for link invocation ({dryrun.returncode})"
            )

    os.execv(str(real_nvcc), [str(real_nvcc), *arguments])
    return 127  # pragma: no cover - os.execv does not return


def _cuda_ledger(args: argparse.Namespace) -> int:
    nvcc = Path(args.nvcc).resolve(strict=True)
    cuda_root = nvcc.parent.parent.resolve(strict=True)
    components = {
        "libcudart_static.a": _cuda_component(
            cuda_root,
            [
                "targets/x86_64-linux/lib/libcudart_static.a",
                "targets/x86_64-linux/lib64/libcudart_static.a",
                "lib64/libcudart_static.a",
            ],
            "libcudart_static.a",
        ),
        "libcudadevrt.a": _cuda_component(
            cuda_root,
            [
                "targets/x86_64-linux/lib/libcudadevrt.a",
                "targets/x86_64-linux/lib64/libcudadevrt.a",
                "lib64/libcudadevrt.a",
            ],
            "libcudadevrt.a",
        ),
        "libdevice.10.bc": _cuda_component(
            cuda_root,
            ["nvvm/libdevice/libdevice.10.bc"],
            "libdevice.10.bc",
        ),
        "cub/cub.cuh": _cuda_component(
            cuda_root,
            [
                "targets/x86_64-linux/include/cub/cub.cuh",
                "include/cub/cub.cuh",
            ],
            "cub/cub.cuh",
        ),
        "cub/version.cuh": _cuda_component(
            cuda_root,
            [
                "targets/x86_64-linux/include/cub/version.cuh",
                "include/cub/version.cuh",
            ],
            "cub/version.cuh",
        ),
    }
    cub_version_text = components["cub/version.cuh"].read_text(
        encoding="utf-8", errors="replace"
    )
    cub_version_match = re.search(
        r"^\s*#\s*define\s+CUB_VERSION\s+(\d+)\b",
        cub_version_text,
        flags=re.MULTILINE,
    )
    if cub_version_match is None or cub_version_match.group(1) != "200500":
        found = cub_version_match.group(1) if cub_version_match else "missing"
        raise ProvenanceError(f"expected CUB/CCCL 2.5.0 (200500), found {found}")

    trace_path = Path(args.trace_jsonl).resolve(strict=True)
    trace_entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProvenanceError(
                f"invalid nvcc trace JSON at {trace_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise ProvenanceError(f"non-object nvcc trace entry at line {line_number}")
        trace_entries.append(entry)
    link_entries = [entry for entry in trace_entries if entry.get("mode") == "link"]
    if len(link_entries) != 2:
        raise ProvenanceError(f"expected exactly two nvcc link invocations; found {len(link_entries)}")

    link_outputs: set[str] = set()
    for entry in link_entries:
        argv = entry.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ProvenanceError("malformed argv in nvcc link trace")
        for index, item in enumerate(argv[:-1]):
            if item == "-o":
                link_outputs.add(Path(argv[index + 1]).name)
    if link_outputs != {"xhdfe.plugin", "xfe.plugin"}:
        raise ProvenanceError(f"unexpected nvcc link outputs: {sorted(link_outputs)}")

    dryrun_path = Path(args.link_dryrun).resolve(strict=True)
    dryrun_text = dryrun_path.read_text(encoding="utf-8", errors="replace")
    if dryrun_text.count("=== XHDFE_NVCC_LINK_DRYRUN_BEGIN ===") != 2:
        raise ProvenanceError("nvcc dryrun trace does not contain exactly two link sections")
    if dryrun_text.count("returncode=0") != 2:
        raise ProvenanceError("one or more nvcc link dryruns did not succeed")

    linked_libraries = _library_tokens(dryrun_text)
    linked_nvidia = {name for name in linked_libraries if _is_nvidia_library(name)}
    missing_expected = _ALLOWED_STATIC_CUDA_LIBRARIES - linked_nvidia
    unexpected_nvidia = linked_nvidia - _ALLOWED_STATIC_CUDA_LIBRARIES
    if missing_expected:
        raise ProvenanceError(
            f"nvcc link trace is missing expected static CUDA libraries: {sorted(missing_expected)}"
        )
    if unexpected_nvidia:
        raise ProvenanceError(
            f"nvcc link trace contains unexpected NVIDIA libraries: {sorted(unexpected_nvidia)}"
        )

    plugin_records: list[dict[str, Any]] = []
    for value in args.plugin:
        plugin = Path(value).resolve(strict=True)
        needed = _readelf_needed(plugin)
        nvidia_needed = [name for name in needed if _is_nvidia_library(name)]
        if nvidia_needed:
            raise ProvenanceError(
                f"{plugin} has forbidden shared NVIDIA NEEDED entries: {nvidia_needed}"
            )
        plugin_records.append(
            {
                "path": str(plugin),
                "sha256": _sha256_file(plugin),
                "size": plugin.stat().st_size,
                "needed": needed,
            }
        )
    if {Path(record["path"]).name for record in plugin_records} != {
        "xhdfe.plugin.linux-cuda",
        "xfe.plugin.linux-cuda",
    }:
        raise ProvenanceError("CUDA ledger must cover exactly the two staged CUDA plugins")

    cub_rpm = _rpm_info_for_path(components["cub/cub.cuh"])
    package_files = _run(["rpm", "-ql", cub_rpm["nevra"]]).stdout.splitlines()
    cccl_license_candidates = [
        Path(item)
        for item in package_files
        if re.search(r"(?:^|/)(?:license|copying|notice)(?:\.[^/]*)?$", item, re.IGNORECASE)
        and Path(item).is_file()
    ]
    cccl_licenses: dict[str, Path] = {}
    for candidate in cccl_license_candidates:
        canonical = candidate.resolve(strict=True)
        cccl_licenses[str(canonical)] = canonical
    if not cccl_licenses:
        raise ProvenanceError(
            f"the CUB provider {cub_rpm['nevra']} does not expose an installed license file"
        )

    toolkit_eula = _unique_existing(
        [cuda_root / "EULA.txt", cuda_root / "EULA", cuda_root / "LICENSE.txt"],
        "CUDA toolkit EULA",
    )
    license_dir = Path(args.license_dir)
    license_dir.mkdir(parents=True, exist_ok=True)
    license_records: list[dict[str, Any]] = []
    license_sources = [("cuda-toolkit-eula", toolkit_eula)] + [
        (f"cuda-cccl-license-{index}", path)
        for index, path in enumerate(sorted(cccl_licenses.values()), 1)
    ]
    for label, source in license_sources:
        suffix = source.suffix if source.suffix else ".txt"
        destination = license_dir / f"{label}{suffix}"
        shutil.copyfile(source, destination)
        record = _file_record(source)
        record.update(
            {
                "label": label,
                "artifact_path": destination.as_posix(),
                "artifact_sha256": _sha256_file(destination),
            }
        )
        if record["artifact_sha256"] != record["sha256"]:
            raise ProvenanceError(f"license copy differs from source: {source}")
        license_records.append(record)

    nvcc_version = _run([str(nvcc), "--version"]).stdout.strip()
    if "release 12.6" not in nvcc_version or "V12.6.85" not in nvcc_version:
        raise ProvenanceError(
            "expected CUDA 12.6 Update 3 nvcc 12.6.85; got:\n" + nvcc_version
        )
    payload = {
        "schema": SCHEMA_VERSION,
        "kind": "cuda-static-link-provenance",
        "nvcc": {
            "path": str(nvcc),
            "sha256": _sha256_file(nvcc),
            "version_output": nvcc_version,
            "rpm": _rpm_info_for_path(nvcc),
        },
        "cuda_root": str(cuda_root),
        "cccl_cub_version": "2.5.0",
        "components": [
            {"name": name, **_file_record(path)} for name, path in sorted(components.items())
        ],
        "licenses": license_records,
        "link_trace": {
            "invocations_path": str(trace_path),
            "invocations_sha256": _sha256_file(trace_path),
            "dryrun_path": str(dryrun_path),
            "dryrun_sha256": _sha256_file(dryrun_path),
            "link_invocations": link_entries,
            "linked_libraries": sorted(linked_libraries),
            "nvidia_libraries": sorted(linked_nvidia),
            "allowed_nvidia_static_libraries": sorted(_ALLOWED_STATIC_CUDA_LIBRARIES),
        },
        "plugins": plugin_records,
    }
    _write_json(Path(args.output), payload)
    print(f"CUDA_PROVENANCE_PASS output={args.output}")
    return 0


def _zip_member_record(archive: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as wheel:
        payload = wheel.read(member)
    return {"member": member, "sha256": _sha256_bytes(payload), "size": len(payload)}


def _wheel_ledger(args: argparse.Namespace) -> int:
    raw_wheel = Path(args.raw_wheel).resolve(strict=True)
    repaired_wheel = Path(args.repaired_wheel).resolve(strict=True)
    provider = Path(args.libgomp_provider).resolve(strict=True)
    source_rpm = Path(args.libgomp_source_rpm).resolve(strict=True)

    provider_rpm = _rpm_info_for_path(provider)
    if not provider.name.startswith("libgomp.so"):
        raise ProvenanceError(f"libgomp provider path has unexpected basename: {provider}")
    if source_rpm.name != provider_rpm["source_rpm"]:
        raise ProvenanceError(
            f"source RPM mismatch: expected {provider_rpm['source_rpm']}, got {source_rpm.name}"
        )
    source_query = _run(
        ["rpm", "-qp", "--qf", "%{NAME}\t%{EPOCHNUM}\t%{VERSION}\t%{RELEASE}\t%{ARCH}", str(source_rpm)]
    ).stdout.strip().split("\t")
    if len(source_query) != 5 or source_query[4] not in {"src", "nosrc"}:
        raise ProvenanceError(f"not a valid source RPM: {source_rpm}")
    source_epoch_prefix = (
        "" if source_query[1] in {"0", "(none)", "(null)"} else f"{source_query[1]}:"
    )
    source_nevra = (
        f"{source_query[0]}-{source_epoch_prefix}{source_query[2]}-"
        f"{source_query[3]}.{source_query[4]}"
    )

    with zipfile.ZipFile(raw_wheel) as archive:
        raw_extensions = [
            name
            for name in archive.namelist()
            if name.endswith(".so") and not any(part.endswith(".libs") for part in PurePosixPath(name).parts)
        ]
    with zipfile.ZipFile(repaired_wheel) as archive:
        repaired_extensions = [
            name
            for name in archive.namelist()
            if name.endswith(".so") and not any(part.endswith(".libs") for part in PurePosixPath(name).parts)
        ]
        private_members = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and any(part.endswith(".libs") for part in PurePosixPath(name).parts[:-1])
        ]
    if len(raw_extensions) != 1 or len(repaired_extensions) != 1:
        raise ProvenanceError(
            "expected exactly one extension module in both raw and repaired wheels"
        )
    if len(private_members) != 1 or not PurePosixPath(private_members[0]).name.startswith("libgomp-"):
        raise ProvenanceError(
            f"expected auditwheel to graft only libgomp; found: {sorted(private_members)}"
        )

    private_member = private_members[0]
    private_basename = PurePosixPath(private_member).name
    with tempfile.TemporaryDirectory(prefix="xhdfe-wheel-ledger-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(raw_wheel) as archive:
            archive.extract(raw_extensions[0], stage / "raw")
        with zipfile.ZipFile(repaired_wheel) as archive:
            archive.extract(repaired_extensions[0], stage / "repaired")
            archive.extract(private_member, stage / "repaired")
        raw_needed = _readelf_needed(stage / "raw" / raw_extensions[0])
        repaired_needed = _readelf_needed(stage / "repaired" / repaired_extensions[0])
        private_needed = _readelf_needed(stage / "repaired" / private_member)
        raw_ldd = _run(["ldd", str(stage / "raw" / raw_extensions[0])]).stdout

    if not any(name.startswith("libgomp.so") for name in raw_needed):
        raise ProvenanceError(f"raw wheel does not depend on libgomp: {raw_needed}")
    resolved_libgomp = re.findall(
        r"^\s*libgomp\.so(?:\.[0-9]+)*\s+=>\s+(\S+)", raw_ldd, flags=re.MULTILINE
    )
    if len(resolved_libgomp) != 1:
        raise ProvenanceError(f"ldd did not resolve exactly one raw-wheel libgomp: {raw_ldd}")
    if Path(resolved_libgomp[0]).resolve(strict=True) != provider:
        raise ProvenanceError(
            "raw-wheel libgomp provider differs from the compiler provider: "
            f"{resolved_libgomp[0]} != {provider}"
        )
    if private_basename not in repaired_needed:
        raise ProvenanceError(
            f"repaired extension does not reference its grafted libgomp member: {repaired_needed}"
        )
    remaining_plain = [name for name in repaired_needed if name.startswith("libgomp.so")]
    if remaining_plain:
        raise ProvenanceError(f"repaired extension retains an external libgomp dependency: {remaining_plain}")

    private_record = _zip_member_record(repaired_wheel, private_member)
    private_record.update(
        {
            "needed": private_needed,
            "provider_source_path": str(provider),
            "provider_resolution_method": "ldd on raw wheel extension",
            "provider_sha256": _sha256_file(provider),
            "provider_size": provider.stat().st_size,
            "provider_rpm": provider_rpm,
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "kind": "manylinux-wheel-private-library-provenance",
        "raw_wheel": {
            "path": str(raw_wheel),
            "sha256": _sha256_file(raw_wheel),
            "extension": {
                **_zip_member_record(raw_wheel, raw_extensions[0]),
                "needed": raw_needed,
                "ldd_output": raw_ldd.strip(),
            },
        },
        "repaired_wheel": {
            "path": str(repaired_wheel),
            "sha256": _sha256_file(repaired_wheel),
            "extension": {
                **_zip_member_record(repaired_wheel, repaired_extensions[0]),
                "needed": repaired_needed,
            },
            "private_libraries": [private_record],
        },
        "corresponding_source": {
            "path": str(source_rpm),
            "sha256": _sha256_file(source_rpm),
            "size": source_rpm.stat().st_size,
            "rpm_name": source_query[0],
            "epoch": source_query[1],
            "version": source_query[2],
            "release": source_query[3],
            "arch": source_query[4],
            "nevra": source_nevra,
            "expected_filename": provider_rpm["source_rpm"],
        },
    }
    _write_json(Path(args.output), payload)
    print(f"WHEEL_PROVENANCE_PASS output={args.output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    wrapper = subparsers.add_parser("nvcc-wrapper")
    wrapper.add_argument("arguments", nargs=argparse.REMAINDER)

    cuda = subparsers.add_parser("cuda-ledger")
    cuda.add_argument("--nvcc", required=True)
    cuda.add_argument("--trace-jsonl", required=True)
    cuda.add_argument("--link-dryrun", required=True)
    cuda.add_argument("--plugin", action="append", required=True)
    cuda.add_argument("--license-dir", required=True)
    cuda.add_argument("--output", required=True)

    wheel = subparsers.add_parser("wheel-ledger")
    wheel.add_argument("--raw-wheel", required=True)
    wheel.add_argument("--repaired-wheel", required=True)
    wheel.add_argument("--libgomp-provider", required=True)
    wheel.add_argument("--libgomp-source-rpm", required=True)
    wheel.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "nvcc-wrapper":
        return _nvcc_wrapper(args.arguments)
    if args.command == "cuda-ledger":
        return _cuda_ledger(args)
    if args.command == "wheel-ledger":
        return _wheel_ledger(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ProvenanceError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"PROVENANCE_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
