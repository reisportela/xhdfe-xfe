"""Falsification tests for the offline corresponding-source custody tools."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

from tools import build_corresponding_source_bundle as builder
from tools import validate_corresponding_source_bundle as validator


REPO_ROOT = Path(__file__).resolve().parents[1]


class CorrespondingSourceBundleTests(unittest.TestCase):
    def _file(self, root: Path, name: str, data: bytes) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def _inputs(self, root: Path, *, contains_cuda: bool = False) -> argparse.Namespace:
        component_paths = {}
        for object_id in sorted(builder.REQUIRED_COMPONENTS):
            component_paths[object_id] = self._file(
                root, f"inputs/{object_id}.src", f"source:{object_id}\n".encode()
            )
        provider_paths = {}
        for object_id in sorted(builder.REQUIRED_PROVIDERS):
            provider_paths[object_id] = self._file(
                root, f"providers/{object_id}.src", f"provider:{object_id}\n".encode()
            )

        components = [f"{name}={path}" for name, path in component_paths.items()]
        providers = [f"{name}={path}" for name, path in provider_paths.items()]
        runtimes = []
        runtime_providers = []
        provider_binaries = []
        metadata = [
            "gcc.version=13.2.0",
            "gcc.upstream_url=https://example.invalid/gcc-13.2.0",
            "mingw-w64.version=11.0.1",
            "mingw-w64.upstream_url=https://example.invalid/mingw-w64-11.0.1",
            "dlfcn-win32.version=1.4.1",
            "dlfcn-win32.upstream_url=https://example.invalid/dlfcn-win32-1.4.1",
            "winlibs-recipes.commit=0123456789abcdef",
            "winlibs-recipes.upstream_url=https://example.invalid/winlibs-recipes",
            "winlibs-tools.commit=fedcba9876543210",
            "winlibs-tools.upstream_url=https://example.invalid/winlibs-tools",
            "ubuntu-mingw-gcc.source_package=gcc-mingw-w64",
            "ubuntu-mingw-gcc.source_version=13.2.0-test",
            "ubuntu-mingw-gcc.upstream_url=https://example.invalid/ubuntu-gcc-source",
            "ubuntu-mingw-w64.source_package=mingw-w64",
            "ubuntu-mingw-w64.source_version=11.0.1-test",
            "ubuntu-mingw-w64.upstream_url=https://example.invalid/ubuntu-mingw-source",
            "manylinux-libgomp.nevra=libgomp-test.x86_64",
            "manylinux-libgomp.sourcerpm=gcc-test.src.rpm",
            "manylinux-libgomp.provider_path=/opt/test/libgomp.so.1",
            "manylinux-libgomp.upstream_url=https://example.invalid/manylinux-source-rpm",
        ]
        for runtime_id, provider_id in sorted(builder.RUNTIME_PROVIDERS.items()):
            data = f"runtime:{runtime_id}\n".encode()
            packaged = self._file(root, f"runtime/{runtime_id}.bin", data)
            provider = self._file(root, f"provider-bin/{runtime_id}.bin", data)
            runtimes.append(f"{runtime_id}={packaged}")
            runtime_providers.append(f"{runtime_id}={provider_id}")
            provider_binaries.append(f"{runtime_id}={provider}")
            metadata.append(f"{runtime_id}.release_path=release/{packaged.name}")
            metadata.append(f"{runtime_id}.provider_path=/toolchain/{runtime_id}.bin")

        return argparse.Namespace(
            output=str(root / "corresponding-source.zip"),
            version="2.24.0.20260815",
            component=components,
            provider_source=providers,
            runtime_binary=runtimes,
            runtime_provider=runtime_providers,
            provider_binary=provider_binaries,
            metadata=metadata,
            contains_cuda=contains_cuda,
            license_dir=str(REPO_ROOT / "third_party" / "licenses"),
            force=False,
        )

    def _stata_runtime_ledgers(self, root: Path) -> tuple[Path, Path, Path]:
        runtime_dir = root / "stata-runtimes"
        runtime_dir.mkdir()
        claims = {
            "libgomp-1.dll": "GCC-13.2.0-COPYING.RUNTIME",
            "libstdc++-6.dll": "GCC-13.2.0-COPYING.RUNTIME",
            "libwinpthread-1.dll": "mingw-w64-11.0.1-winpthreads-COPYING",
        }
        entries = []
        for name, license_file in claims.items():
            payload = f"test runtime {name}\n".encode()
            (runtime_dir / name).write_bytes(payload)
            entries.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                    "source_path": f"/usr/test/{name}",
                    "runtime_package": "test-runtime:amd64",
                    "runtime_version": "1-test",
                    "source_package": "test-source",
                    "source_version": "1-test",
                    "built_using": "",
                    "license_file": license_file,
                }
            )
        provider_ledger = root / "windows-stata-provider-ledger.json"
        provider_ledger.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact": "xhdfe-xfe-stata-windows-cpu",
                    "compiler": {
                        "target": "x86_64-w64-mingw32",
                        "version": "13.2.0",
                    },
                    "entries": sorted(entries, key=lambda item: item["name"].lower()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        closure_ledger = root / "windows-stata-runtime-ledger.json"
        closure_ledger.write_text(
            json.dumps(
                {
                    "format": "xhdfe-windows-runtime-closure-v1",
                    "roots": [
                        {
                            "architecture": "pei-x86-64",
                            "dependencies": sorted(claims),
                            "member": "xhdfe.plugin",
                            "member_sha256": "0" * 64,
                            "size": 1,
                        }
                    ],
                    "runtimes": [
                        {
                            "architecture": "pei-x86-64",
                            "dependencies": [],
                            "license": "test",
                            "member": entry["name"],
                            "member_sha256": entry["sha256"],
                            "resolution_method": "test",
                            "size": entry["size"],
                            "source_path": entry["source_path"],
                            "source_sha256": entry["sha256"],
                        }
                        for entry in entries
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return runtime_dir, provider_ledger, closure_ledger

    def _stage_windows_site(
        self,
        root: Path,
        runtime_dir: Path,
        provider_ledger: Path,
        closure_ledger: Path,
        site_name: str,
    ) -> subprocess.CompletedProcess[str]:
        xhdfe_plugin = self._file(root, "plugins/xhdfe.plugin", b"xhdfe PE fixture\n")
        xfe_plugin = self._file(root, "plugins/xfe.plugin", b"xfe PE fixture\n")
        return subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "tools" / "stage_stata_netinstall_site.sh"),
                str(root / site_name),
                "--windows-xhdfe",
                str(xhdfe_plugin),
                "--windows-xfe",
                str(xfe_plugin),
                "--windows-runtime-dir",
                str(runtime_dir),
                "--windows-runtime-provider-ledger",
                str(provider_ledger),
                "--windows-runtime-closure-ledger",
                str(closure_ledger),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )

    def _rewrite_provenance(
        self, source: Path, destination: Path, transform
    ) -> None:
        with zipfile.ZipFile(source) as archive:
            payloads = {info.filename: archive.read(info.filename) for info in archive.infolist()}
        root = next(iter(payloads)).split("/", 1)[0]
        provenance_name = f"{root}/PROVENANCE.json"
        provenance = json.loads(payloads[provenance_name])
        transform(provenance)
        payloads[provenance_name] = (
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        ).encode()
        manifest_name = f"{root}/MANIFEST.sha256"
        payloads[manifest_name] = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name[len(root) + 1:]}\n"
            for name, data in sorted(payloads.items())
            if name != manifest_name
        ).encode()
        with zipfile.ZipFile(destination, "w") as archive:
            for name, data in sorted(payloads.items()):
                archive.writestr(name, data)

    def test_roundtrip_closes_every_required_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._inputs(Path(temporary))
            builder.build_archive(args)
            provenance = validator.validate(Path(args.output))
            self.assertEqual(
                set(builder.RUNTIME_PROVIDERS),
                {entry["id"] for entry in provenance["runtime_binaries"]},
            )

    def test_archive_is_deterministic_across_staging_paths(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_args = self._inputs(Path(first))
            second_args = self._inputs(Path(second))
            builder.build_archive(first_args)
            builder.build_archive(second_args)
            self.assertEqual(
                hashlib.sha256(Path(first_args.output).read_bytes()).hexdigest(),
                hashlib.sha256(Path(second_args.output).read_bytes()).hexdigest(),
            )

    def test_missing_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._inputs(Path(temporary))
            args.component = [
                value for value in args.component if not value.startswith("gcc=")
            ]
            with self.assertRaisesRegex(ValueError, "missing source components: gcc"):
                builder.build_archive(args)

    def test_extra_runtime_requires_full_mapping_and_is_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._inputs(root)
            packaged = self._file(root, "runtime/extra-runtime.dll", b"extra runtime\n")
            provider = self._file(root, "provider-bin/extra-runtime.dll", b"extra runtime\n")
            args.runtime_binary.append(f"windows-python-extra={packaged}")
            args.provider_binary.append(f"windows-python-extra={provider}")
            args.metadata.extend(
                [
                    "windows-python-extra.release_path=wheel/xhdfe/extra-runtime.dll",
                    "windows-python-extra.provider_path=C:/toolchain/extra-runtime.dll",
                ]
            )
            with self.assertRaisesRegex(ValueError, "runtime-provider"):
                builder.build_archive(args)

            args.runtime_provider.append("windows-python-extra=gcc")
            builder.build_archive(args)
            provenance = validator.validate(Path(args.output))
            runtime_ids = {entry["id"] for entry in provenance["runtime_binaries"]}
            self.assertIn("windows-python-extra", runtime_ids)

    def test_missing_license_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._inputs(Path(temporary))
            empty = Path(temporary) / "empty-licenses"
            empty.mkdir()
            args.license_dir = str(empty)
            with self.assertRaisesRegex(ValueError, "missing input"):
                builder.build_archive(args)

    def test_payload_hash_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._inputs(root)
            builder.build_archive(args)
            original = Path(args.output)
            mutated = root / "mutated.zip"
            with zipfile.ZipFile(original) as source, zipfile.ZipFile(mutated, "w") as target:
                for info in source.infolist():
                    data = source.read(info.filename)
                    if info.filename.endswith("/README.md"):
                        data += b"mutation\n"
                    target.writestr(info, data)
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                validator.validate(mutated)

    def test_duplicate_runtime_id_is_rejected_beyond_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._inputs(root)
            builder.build_archive(args)
            mutated = root / "duplicate-runtime.zip"

            def duplicate(provenance):
                provenance["runtime_binaries"].append(
                    copy.deepcopy(provenance["runtime_binaries"][0])
                )

            self._rewrite_provenance(Path(args.output), mutated, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate runtime evidence IDs"):
                validator.validate(mutated)

    def test_false_byte_identity_claim_is_rejected_beyond_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._inputs(root)
            builder.build_archive(args)
            mutated = root / "false-byte-identity.zip"

            def contradict(provenance):
                runtime = provenance["runtime_binaries"][0]
                runtime["provider_binary_sha256"] = "0" * 64
                runtime["byte_identical_to_provider"] = True

            self._rewrite_provenance(Path(args.output), mutated, contradict)
            with self.assertRaisesRegex(ValueError, "claim contradicts recorded hashes"):
                validator.validate(mutated)

    def test_license_ledger_omission_is_rejected_beyond_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._inputs(root)
            builder.build_archive(args)
            mutated = root / "missing-license-record.zip"

            def omit(provenance):
                provenance["license_files"] = provenance["license_files"][:-1]

            self._rewrite_provenance(Path(args.output), mutated, omit)
            with self.assertRaisesRegex(ValueError, "license ledger"):
                validator.validate(mutated)

    def test_unsafe_zip_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._inputs(root)
            builder.build_archive(args)
            unsafe = root / "unsafe.zip"
            shutil.copyfile(args.output, unsafe)
            with zipfile.ZipFile(unsafe, "a") as archive:
                archive.writestr("../escape", b"not extracted")
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member path"):
                validator.validate(unsafe)

    def test_cuda_requires_and_validates_both_exact_license_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            license_dir = root / "licenses"
            license_dir.mkdir()
            for name in builder.LICENSE_HASHES:
                shutil.copyfile(REPO_ROOT / "third_party" / "licenses" / name, license_dir / name)

            fake_cuda = {
                "NVIDIA-CUDA-12.6-EULA.pdf": b"test CUDA EULA bytes\n",
                "NVIDIA-CCCL-2.5.0-LICENSE": b"test CCCL license bytes\n",
            }
            test_hashes = {
                name: hashlib.sha256(data).hexdigest()
                for name, data in fake_cuda.items()
            }
            (license_dir / "NVIDIA-CUDA-12.6-EULA.pdf").write_bytes(
                fake_cuda["NVIDIA-CUDA-12.6-EULA.pdf"]
            )

            args = self._inputs(root, contains_cuda=True)
            args.license_dir = str(license_dir)
            with mock.patch.dict(builder.CUDA_LICENSE_HASHES, test_hashes, clear=True):
                with self.assertRaisesRegex(ValueError, "missing input"):
                    builder.build_archive(args)

            (license_dir / "NVIDIA-CCCL-2.5.0-LICENSE").write_bytes(
                fake_cuda["NVIDIA-CCCL-2.5.0-LICENSE"]
            )
            with mock.patch.dict(builder.CUDA_LICENSE_HASHES, test_hashes, clear=True), \
                 mock.patch.object(builder.subprocess, "run"):
                builder.build_archive(args)
            with mock.patch.dict(validator.CUDA_LICENSE_HASHES, test_hashes, clear=True):
                provenance = validator.validate(Path(args.output))
            self.assertTrue(provenance["cuda"]["included"])
            self.assertEqual(
                provenance["cuda"]["redistributable_static_components"],
                ["libcudadevrt", "libcudart_static"],
            )

    def test_stata_stage_uses_complete_generic_runtime_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_dir, provider_ledger, closure_ledger = self._stata_runtime_ledgers(root)
            staged = self._stage_windows_site(
                root, runtime_dir, provider_ledger, closure_ledger, "site"
            )
            self.assertEqual(staged.returncode, 0, staged.stderr)
            site = root / "site"
            validated = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "tools" / "validate_stata_package_site.sh"),
                    str(site),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            package = (site / "xhdfe.pkg").read_text(encoding="utf-8")
            for runtime in sorted(path.name for path in runtime_dir.iterdir()):
                self.assertIn(f"g WIN64 {runtime} {runtime}\n", package)
            self.assertIn(
                "g WIN64 windows-stata-provider-ledger.json "
                "windows-stata-provider-ledger.json\n",
                package,
            )
            self.assertIn(
                "g WIN64 windows-stata-runtime-ledger.json "
                "windows-stata-runtime-ledger.json\n",
                package,
            )

            (site / "unledgered.dll").write_bytes(b"orphan\n")
            rejected = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "tools" / "validate_stata_package_site.sh"),
                    str(site),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("runtime ledger/site DLL mismatch", rejected.stderr)

    def test_stata_stage_rejects_runtime_not_in_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_dir, provider_ledger, closure_ledger = self._stata_runtime_ledgers(root)
            (runtime_dir / "unledgered.dll").write_bytes(b"orphan\n")
            rejected = self._stage_windows_site(
                root,
                runtime_dir,
                provider_ledger,
                closure_ledger,
                "bad-site",
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("ledger/directory mismatch", rejected.stderr)

    def test_stata_stage_rejects_provider_closure_ledger_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_dir, provider_ledger, closure_ledger = self._stata_runtime_ledgers(root)
            closure = json.loads(closure_ledger.read_text(encoding="utf-8"))
            closure["runtimes"][0]["member_sha256"] = "0" * 64
            closure_ledger.write_text(
                json.dumps(closure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rejected = self._stage_windows_site(
                root,
                runtime_dir,
                provider_ledger,
                closure_ledger,
                "bad-closure-site",
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("ledgers disagree", rejected.stderr)

    def test_static_windows_package_installs_entire_runtime_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_dir, provider_ledger, closure_ledger = self._stata_runtime_ledgers(root)
            staged = self._stage_windows_site(
                root, runtime_dir, provider_ledger, closure_ledger, "static-site"
            )
            self.assertEqual(staged.returncode, 0, staged.stderr)
            site = root / "static-site"
            for cmd in ("xhdfe", "xfe"):
                (site / f"{cmd}.win64.plugin").replace(site / f"{cmd}.plugin")
            runtime_names = sorted(path.name for path in runtime_dir.iterdir())
            additions = runtime_names + [
                "windows-stata-provider-ledger.json",
                "windows-stata-runtime-ledger.json",
            ]
            for pkg_name in ("xhdfe.pkg", "xfe.pkg"):
                pkg = site / pkg_name
                lines = [
                    line
                    for line in pkg.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("g WIN64 ")
                ]
                plugin_names = (
                    ("xhdfe.plugin", "xfe.plugin")
                    if pkg_name == "xhdfe.pkg"
                    else ("xfe.plugin",)
                )
                lines.extend(f"f {name}" for name in plugin_names)
                lines.extend(f"f {name}" for name in additions)
                pkg.write_text("\n".join(lines) + "\n", encoding="utf-8")

            validated = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "tools" / "validate_stata_package_site.sh"),
                    str(site),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)

            xfe_pkg = site / "xfe.pkg"
            omitted = f"f {runtime_names[0]}\n"
            xfe_pkg.write_text(
                xfe_pkg.read_text(encoding="utf-8").replace(omitted, ""),
                encoding="utf-8",
            )
            rejected = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "tools" / "validate_stata_package_site.sh"),
                    str(site),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("runtime DLL lacks one exact platform mapping", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
