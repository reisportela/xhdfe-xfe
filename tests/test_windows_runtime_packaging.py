"""Tests for fail-closed MinGW runtime bundling in the Python wheel."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xhdfe


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "xhdfe_setup_windows_runtime_test", ROOT / "setup.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load setup.py")
    module = importlib.util.module_from_spec(spec)
    with patch("setuptools.setup"):
        spec.loader.exec_module(module)
    return module


SETUP_MODULE = _load_setup_module()


def _load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "xhdfe_python_artifact_validator_test",
        ROOT / "tools" / "validate_python_release_artifacts.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Python artifact validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR_MODULE = _load_validator_module()


def _write_cmake_toolchain(build_dir: Path, compiler: Path, compiler_id: str) -> None:
    build_dir.mkdir(parents=True)
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n", encoding="utf-8"
    )
    compiler_dir = build_dir / "CMakeFiles" / "3.30.0"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "CMakeCXXCompiler.cmake").write_text(
        f'set(CMAKE_CXX_COMPILER_ID "{compiler_id}")\n', encoding="utf-8"
    )


def _synthetic_pe_ledger(
    root_members: dict[str, tuple[bytes, list[str]]],
    runtime_members: dict[str, tuple[bytes, list[str]]],
) -> dict[str, object]:
    roots = []
    for name in sorted(root_members, key=str.casefold):
        data, dependencies = root_members[name]
        roots.append(
            {
                "architecture": "pei-x86-64",
                "dependencies": sorted(dependencies, key=str.casefold),
                "member": name,
                "member_sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    runtimes = []
    for name in sorted(runtime_members, key=str.casefold):
        data, dependencies = runtime_members[name]
        digest = hashlib.sha256(data).hexdigest()
        runtimes.append(
            {
                "architecture": "pei-x86-64",
                "dependencies": sorted(dependencies, key=str.casefold),
                "license": VALIDATOR_MODULE._windows_runtime_license(name),
                "member": name,
                "member_sha256": digest,
                "resolution_method": "compiler-print-file-name",
                "size": len(data),
                "source_path": rf"C:\Strawberry\c\bin\{name}",
                "source_sha256": digest,
            }
        )
    return {
        "format": VALIDATOR_MODULE._WINDOWS_RUNTIME_FORMAT,
        "roots": roots,
        "runtimes": runtimes,
    }


def _write_synthetic_wheel(
    wheel: Path,
    roots: dict[str, tuple[bytes, list[str]]],
    runtimes: dict[str, tuple[bytes, list[str]]],
    ledger: dict[str, object],
    extra_members: tuple[tuple[str, bytes], ...] = (),
) -> None:
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, (data, _) in roots.items():
            archive.writestr(f"xhdfe/{name}", data)
        for name, (data, _) in runtimes.items():
            archive.writestr(f"xhdfe/{name}", data)
        archive.writestr(
            f"xhdfe/{VALIDATOR_MODULE._WINDOWS_RUNTIME_MANIFEST}",
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        )
        for member, data in extra_members:
            archive.writestr(member, data)


class WindowsRuntimePackagingTest(unittest.TestCase):
    def test_windows_native_tuning_is_portable_by_default(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("if(APPLE OR WIN32)", cmake)
        self.assertRegex(
            cmake,
            r"if\(APPLE OR WIN32\)[\s\S]*?set\(XHDFE_ENABLE_MARCH_NATIVE_DEFAULT OFF\)[\s\S]*?endif\(\)",
        )

    def test_windows_host_dependency_policy_is_fail_closed(self):
        self.assertTrue(SETUP_MODULE._is_windows_host_dll("KERNEL32.dll"))
        self.assertTrue(
            SETUP_MODULE._is_windows_host_dll(
                "api-ms-win-crt-runtime-l1-1-0.dll"
            )
        )
        self.assertTrue(SETUP_MODULE._is_windows_host_dll("python312.dll"))
        self.assertFalse(SETUP_MODULE._is_windows_host_dll("python999.dll"))
        self.assertFalse(SETUP_MODULE._is_windows_host_dll("libdl.dll"))
        self.assertFalse(SETUP_MODULE._is_windows_host_dll("vcruntime999.dll"))
        self.assertFalse(SETUP_MODULE._is_windows_host_dll("vendor-runtime.dll"))
        self.assertTrue(
            VALIDATOR_MODULE._is_licensed_bundled_dll("libgomp-1.dll")
        )
        self.assertTrue(
            VALIDATOR_MODULE._is_licensed_bundled_dll("libwinpthread-1.dll")
        )
        self.assertTrue(
            VALIDATOR_MODULE._is_licensed_bundled_dll("libdl.dll")
        )
        self.assertFalse(
            VALIDATOR_MODULE._is_licensed_bundled_dll("vendor-runtime.dll")
        )
        self.assertEqual(
            SETUP_MODULE._WINDOWS_HOST_DLL_NAMES,
            VALIDATOR_MODULE._WINDOWS_HOST_DLL_NAMES,
        )
        self.assertEqual(
            SETUP_MODULE._WINDOWS_HOST_DLL_RE.pattern,
            VALIDATOR_MODULE._WINDOWS_HOST_DLL_RE.pattern,
        )

    def test_directory_without_packaged_dll_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            (package_dir / "unrelated.txt").write_bytes(b"not a DLL")

            def unexpected_register(_path):
                raise AssertionError("non-MinGW directory must not be registered")

            self.assertIsNone(
                xhdfe._register_packaged_dll_directory(
                    package_dir, add_dll_directory=unexpected_register
                )
            )

    def test_packaged_runtime_directory_is_registered_and_handle_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            (package_dir / "vendor-runtime.dll").write_bytes(b"runtime")
            handle = object()
            calls = []

            def register(path):
                calls.append(path)
                return handle

            saved_handles = xhdfe._PACKAGED_DLL_DIRECTORY_HANDLES
            xhdfe._PACKAGED_DLL_DIRECTORY_HANDLES = {}
            try:
                first = xhdfe._register_packaged_dll_directory(
                    package_dir, add_dll_directory=register
                )
                second = xhdfe._register_packaged_dll_directory(
                    package_dir, add_dll_directory=register
                )
                self.assertIs(first, handle)
                self.assertIs(second, handle)
                self.assertEqual(calls, [str(package_dir.resolve())])
                self.assertIs(
                    xhdfe._PACKAGED_DLL_DIRECTORY_HANDLES[
                        str(package_dir.resolve())
                    ],
                    handle,
                )
            finally:
                xhdfe._PACKAGED_DLL_DIRECTORY_HANDLES = saved_handles

    def test_strawberry_style_adjacent_objdump_is_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "Strawberry" / "c" / "bin"
            bin_dir.mkdir(parents=True)
            compiler = bin_dir / "g++.exe"
            objdump = bin_dir / "objdump.exe"
            compiler.write_bytes(b"compiler")
            objdump.write_bytes(b"objdump")
            self.assertEqual(
                SETUP_MODULE._find_mingw_objdump(compiler), objdump
            )

    def test_objdump_dependency_parser(self):
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "py_hdfe_v11.pyd: file format pei-x86-64\n"
                "architecture: i386:x86-64, flags 0x0000012f:\n"
                "The Import Tables\n"
                "\tDLL Name: KERNEL32.dll\n"
                "\tDLL Name: libgomp-1.dll\n"
                "\tDLL Name: libstdc++-6.dll\n"
            ),
            stderr="",
        )
        with patch.object(SETUP_MODULE.subprocess, "run", return_value=result):
            dependencies = SETUP_MODULE._pe_dll_dependencies(
                Path("objdump.exe"), Path("py_hdfe_v11.pyd")
            )
            architecture = SETUP_MODULE._pe_architecture(
                Path("objdump.exe"), Path("py_hdfe_v11.pyd")
            )
        self.assertEqual(
            dependencies,
            {"KERNEL32.dll", "libgomp-1.dll", "libstdc++-6.dll"},
        )
        self.assertEqual(architecture, "pei-x86-64")

    def test_compiler_reported_runtime_path_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "libgomp-1.dll"
            runtime.write_bytes(b"runtime")
            result = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=f"{runtime}\n", stderr=""
            )
            with patch.object(
                SETUP_MODULE.subprocess, "run", return_value=result
            ):
                self.assertEqual(
                    SETUP_MODULE._compiler_runtime_path(
                        Path(tmp) / "g++.exe", "libgomp-1.dll"
                    ),
                    runtime,
                )

    def test_adjacent_runtime_is_used_when_compiler_reports_only_its_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            compiler = bin_dir / "g++.exe"
            compiler.write_bytes(b"compiler")
            runtime = bin_dir / "libstdc++-6.dll"
            runtime.write_bytes(b"runtime")
            result = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="libstdc++-6.dll\n", stderr=""
            )
            with patch.object(
                SETUP_MODULE.subprocess, "run", return_value=result
            ):
                self.assertEqual(
                    SETUP_MODULE._compiler_runtime_path(
                        compiler, "libstdc++-6.dll"
                    ),
                    runtime,
                )

    def test_recursive_runtime_closure_is_copied_beside_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            compiler = root / "toolchain" / "bin" / "g++.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            _write_cmake_toolchain(build_dir, compiler, "GNU")

            extension = root / "wheel" / "xhdfe" / "py_hdfe_v11.pyd"
            extension.parent.mkdir(parents=True)
            extension.write_bytes(b"extension")

            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            runtime_names = (
                "libgcc_s_seh-1.dll",
                "libstdc++-6.dll",
                "libgomp-1.dll",
                "libwinpthread-1.dll",
                "libatomic-1.dll",
                "libdl.dll",
                "vendor-runtime.dll",
            )
            runtimes = {}
            for name in runtime_names:
                path = runtime_dir / name
                path.write_bytes(f"runtime:{name}".encode())
                runtimes[name.lower()] = path

            dependencies = {
                extension.name.lower(): {
                    "KERNEL32.dll",
                    "libstdc++-6.dll",
                    "libgomp-1.dll",
                },
                "libstdc++-6.dll": {"libgcc_s_seh-1.dll", "libatomic-1.dll"},
                "libgomp-1.dll": {
                    "libgcc_s_seh-1.dll",
                    "libwinpthread-1.dll",
                    "libdl.dll",
                },
                "libwinpthread-1.dll": {"libgcc_s_seh-1.dll"},
                "libgcc_s_seh-1.dll": set(),
                "libatomic-1.dll": {"libgcc_s_seh-1.dll"},
                "libdl.dll": {
                    "KERNEL32.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "vendor-runtime.dll",
                },
                "vendor-runtime.dll": {"KERNEL32.dll"},
            }

            def inspect_dependencies(_objdump, binary):
                return dependencies[binary.name.lower()]

            def find_runtime(_compiler, name, _search_dirs=()):
                path = runtimes.get(name.lower())
                return path, "compiler-print-file-name" if path else None

            with (
                patch.object(
                    SETUP_MODULE,
                    "_find_mingw_objdump",
                    return_value=root / "objdump.exe",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_architecture",
                    return_value="i386:x86-64",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_dll_dependencies",
                    side_effect=inspect_dependencies,
                ),
                patch.object(
                    SETUP_MODULE,
                    "_compiler_runtime_resolution",
                    side_effect=find_runtime,
                ),
            ):
                bundled = SETUP_MODULE._bundle_mingw_runtime_dlls(
                    extension, build_dir
                )

            self.assertEqual({path.name for path in bundled}, set(runtime_names))
            for name in runtime_names:
                self.assertEqual(
                    (extension.parent / name).read_bytes(),
                    runtimes[name.lower()].read_bytes(),
                )
            manifest_path = extension.parent / SETUP_MODULE._WINDOWS_RUNTIME_MANIFEST
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            self.assertEqual(
                manifest_bytes,
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            )
            self.assertEqual(
                [record["member"] for record in manifest["runtimes"]],
                sorted(runtime_names, key=str.casefold),
            )
            for record in manifest["runtimes"]:
                expected_hash = hashlib.sha256(
                    runtimes[record["member"].lower()].read_bytes()
                ).hexdigest()
                self.assertEqual(record["source_sha256"], expected_hash)
                self.assertEqual(record["member_sha256"], expected_hash)
                self.assertTrue(Path(record["source_path"]).is_absolute())
                expected_license = SETUP_MODULE._windows_runtime_license(
                    record["member"]
                )
                self.assertEqual(record["license"], expected_license or "UNMAPPED")

    def test_wheel_runtime_ledger_validates_direct_zip_bytes(self):
        roots = {
            "py_hdfe_v11.cp312-win_amd64.pyd": (
                b"synthetic-pyd",
                ["KERNEL32.dll", "libgomp-1.dll"],
            )
        }
        runtimes = {"libgomp-1.dll": (b"synthetic-libgomp", ["KERNEL32.dll"])}
        ledger = _synthetic_pe_ledger(roots, runtimes)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / "synthetic-cp312-cp312-win_amd64.whl"
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            _write_synthetic_wheel(wheel, roots, runtimes, ledger)

            dependencies = {
                name.casefold(): set(items[1])
                for name, items in {**roots, **runtimes}.items()
            }
            with (
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_dependencies",
                    side_effect=lambda _objdump, path: dependencies[
                        path.name.casefold()
                    ],
                ),
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_architecture",
                    return_value="pei-x86-64",
                ),
            ):
                VALIDATOR_MODULE._validate_mingw_closure(wheel, objdump)

    def test_wheel_runtime_ledger_rejects_mutations(self):
        roots = {
            "py_hdfe_v11.cp312-win_amd64.pyd": (
                b"synthetic-pyd",
                ["KERNEL32.dll", "libgomp-1.dll"],
            )
        }
        runtimes = {"libgomp-1.dll": (b"synthetic-libgomp", ["KERNEL32.dll"])}
        base_ledger = _synthetic_pe_ledger(roots, runtimes)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            cases = (
                (
                    "tampered",
                    {"libgomp-1.dll": (b"tampered", ["KERNEL32.dll"])},
                    base_ledger,
                    (),
                    "packaged bytes do not match member_sha256",
                ),
                (
                    "missing",
                    {},
                    base_ledger,
                    (),
                    "ledger DLLs differ from packaged DLLs",
                ),
                (
                    "extra",
                    runtimes,
                    base_ledger,
                    (("xhdfe/libgcc_s_seh-1.dll", b"extra"),),
                    "ledger DLLs differ from packaged DLLs",
                ),
                (
                    "duplicate",
                    runtimes,
                    base_ledger,
                    (("xhdfe/LIBGOMP-1.DLL", b"synthetic-libgomp"),),
                    "duplicate case-insensitive",
                ),
            )
            dependencies = {
                name.casefold(): set(items[1])
                for name, items in {**roots, **runtimes}.items()
            }
            for case, packaged_runtimes, ledger, extra_members, message in cases:
                with self.subTest(case=case):
                    wheel = root / f"{case}.whl"
                    _write_synthetic_wheel(
                        wheel,
                        roots,
                        packaged_runtimes,
                        ledger,
                        extra_members,
                    )
                    with (
                        patch.object(
                            VALIDATOR_MODULE,
                            "_pe_dependencies",
                            side_effect=lambda _objdump, path: dependencies[
                                path.name.casefold()
                            ],
                        ),
                        patch.object(
                            VALIDATOR_MODULE,
                            "_pe_architecture",
                            return_value="pei-x86-64",
                        ),
                        self.assertRaisesRegex(RuntimeError, message),
                    ):
                        VALIDATOR_MODULE._validate_mingw_closure(wheel, objdump)

    def test_wheel_runtime_ledger_rejects_architecture_and_license_claims(self):
        roots = {
            "py_hdfe_v11.cp312-win_amd64.pyd": (
                b"synthetic-pyd",
                ["KERNEL32.dll", "libgomp-1.dll"],
            )
        }
        runtimes = {"libgomp-1.dll": (b"synthetic-libgomp", ["KERNEL32.dll"])}
        dependencies = {
            name.casefold(): set(items[1])
            for name, items in {**roots, **runtimes}.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            for field, value, message in (
                ("architecture", "pei-i386", "architecture does not match ledger"),
                ("license", "MIT", "incorrect license mapping"),
            ):
                with self.subTest(field=field):
                    ledger = _synthetic_pe_ledger(roots, runtimes)
                    ledger["runtimes"][0][field] = value
                    wheel = root / f"wrong-{field}.whl"
                    _write_synthetic_wheel(wheel, roots, runtimes, ledger)
                    with (
                        patch.object(
                            VALIDATOR_MODULE,
                            "_pe_dependencies",
                            side_effect=lambda _objdump, path: dependencies[
                                path.name.casefold()
                            ],
                        ),
                        patch.object(
                            VALIDATOR_MODULE,
                            "_pe_architecture",
                            return_value="pei-x86-64",
                        ),
                        self.assertRaisesRegex(RuntimeError, message),
                    ):
                        VALIDATOR_MODULE._validate_mingw_closure(wheel, objdump)

    def test_generic_pe_ledger_builder_copies_and_revalidates_stata_bundle(self):
        roots = {
            "xfe.plugin": (b"xfe-plugin", ["KERNEL32.dll"]),
            "xhdfe.plugin": (
                b"xhdfe-plugin",
                ["KERNEL32.dll", "libgomp-1.dll"],
            ),
        }
        runtimes = {"libgomp-1.dll": (b"synthetic-libgomp", ["KERNEL32.dll"])}
        dependencies = {
            name.casefold(): set(items[1])
            for name, items in {**roots, **runtimes}.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            source_dir = root / "toolchain"
            source_dir.mkdir()
            binaries = []
            for name, (data, _) in roots.items():
                path = root / name
                path.write_bytes(data)
                binaries.append(path)
            runtime_sources = []
            for name, (data, _) in runtimes.items():
                source = source_dir / name
                source.write_bytes(data)
                runtime_sources.append(source)
            ledger = root / "runtime-ledger.json"
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            with (
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_dependencies",
                    side_effect=lambda _objdump, path: dependencies[
                        path.name.casefold()
                    ],
                ),
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_architecture",
                    return_value="pei-x86-64",
                ),
            ):
                VALIDATOR_MODULE.build_pe_runtime_ledger(
                    binaries,
                    runtime_sources,
                    runtime_dir,
                    ledger,
                    objdump,
                )
                VALIDATOR_MODULE.validate_pe_runtime_directory(
                    binaries, runtime_dir, ledger, objdump
                )
                ledger_data = json.loads(ledger.read_bytes())
                self.assertEqual(
                    ledger.read_bytes(),
                    (json.dumps(ledger_data, indent=2, sort_keys=True) + "\n").encode(),
                )
                self.assertEqual(
                    ledger_data["runtimes"][0]["resolution_method"],
                    "explicit-runtime-source",
                )
                self.assertEqual(
                    ledger_data["runtimes"][0]["source_path"],
                    str(runtime_sources[0].resolve()),
                )
                (runtime_dir / "libgomp-1.dll").write_bytes(b"tampered-after-copy")
                with self.assertRaisesRegex(
                    RuntimeError, "packaged bytes do not match member_sha256"
                ):
                    VALIDATOR_MODULE.validate_pe_runtime_directory(
                        binaries, runtime_dir, ledger, objdump
                    )

    def test_runtime_search_directories_resolve_complete_transitive_graph(self):
        roots = {
            "xhdfe.plugin": (
                b"xhdfe-plugin",
                ["KERNEL32.dll", "libgomp-1.dll"],
            )
        }
        runtimes = {
            "libgomp-1.dll": (
                b"libgomp",
                ["libgcc_s_seh-1.dll", "libwinpthread-1.dll"],
            ),
            "libgcc_s_seh-1.dll": (b"libgcc", ["KERNEL32.dll"]),
            "libwinpthread-1.dll": (b"libwinpthread", ["KERNEL32.dll"]),
        }
        dependencies = {
            name.casefold(): set(items[1])
            for name, items in {**roots, **runtimes}.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            search_a = root / "toolchain-a"
            search_b = root / "toolchain-b"
            runtime_dir.mkdir()
            search_a.mkdir()
            search_b.mkdir()
            binary = root / "xhdfe.plugin"
            binary.write_bytes(roots[binary.name][0])
            for name, (data, _) in runtimes.items():
                target_dir = search_b if name == "libwinpthread-1.dll" else search_a
                (target_dir / name).write_bytes(data)
            (search_b / "libgomp-1.dll").write_bytes(runtimes["libgomp-1.dll"][0])
            ledger = root / "runtime-ledger.json"
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            with (
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_dependencies",
                    side_effect=lambda _objdump, path: dependencies[
                        path.name.casefold()
                    ],
                ),
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_architecture",
                    return_value="pei-x86-64",
                ),
            ):
                VALIDATOR_MODULE.build_pe_runtime_ledger(
                    [binary],
                    [],
                    runtime_dir,
                    ledger,
                    objdump,
                    [search_a, search_b],
                )

            manifest = json.loads(ledger.read_bytes())
            self.assertEqual(
                {record["member"] for record in manifest["runtimes"]},
                set(runtimes),
            )
            self.assertTrue(
                all(
                    record["resolution_method"] == "runtime-search-directory"
                    for record in manifest["runtimes"]
                )
            )
            libgomp = next(
                record
                for record in manifest["runtimes"]
                if record["member"] == "libgomp-1.dll"
            )
            self.assertEqual(
                libgomp["source_path"], str((search_a / "libgomp-1.dll").resolve())
            )
            self.assertEqual(
                {path.name for path in runtime_dir.iterdir()}, set(runtimes)
            )

    def test_runtime_search_missing_transitive_dependency_fails_before_copy(self):
        dependencies = {
            "xhdfe.plugin": {"libgomp-1.dll"},
            "libgomp-1.dll": {"libgcc_s_seh-1.dll"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            search_dir = root / "toolchain"
            runtime_dir.mkdir()
            search_dir.mkdir()
            binary = root / "xhdfe.plugin"
            binary.write_bytes(b"plugin")
            (search_dir / "libgomp-1.dll").write_bytes(b"libgomp")
            ledger = root / "runtime-ledger.json"
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            with (
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_dependencies",
                    side_effect=lambda _objdump, path: dependencies[
                        path.name.casefold()
                    ],
                ),
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_architecture",
                    return_value="pei-x86-64",
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "could not resolve non-system PE dependency libgcc_s_seh-1.dll",
                ),
            ):
                VALIDATOR_MODULE.build_pe_runtime_ledger(
                    [binary], [], runtime_dir, ledger, objdump, [search_dir]
                )
            self.assertEqual(list(runtime_dir.iterdir()), [])
            self.assertFalse(ledger.exists())

    def test_runtime_search_ambiguous_different_bytes_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            search_a = root / "toolchain-a"
            search_b = root / "toolchain-b"
            runtime_dir.mkdir()
            search_a.mkdir()
            search_b.mkdir()
            binary = root / "xhdfe.plugin"
            binary.write_bytes(b"plugin")
            (search_a / "libgomp-1.dll").write_bytes(b"first")
            (search_b / "libgomp-1.dll").write_bytes(b"second")
            ledger = root / "runtime-ledger.json"
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            with (
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_dependencies",
                    return_value={"libgomp-1.dll"},
                ),
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_architecture",
                    return_value="pei-x86-64",
                ),
                self.assertRaisesRegex(
                    RuntimeError, "ambiguous runtime sources with different bytes"
                ),
            ):
                VALIDATOR_MODULE.build_pe_runtime_ledger(
                    [binary], [], runtime_dir, ledger, objdump, [search_a, search_b]
                )
            self.assertEqual(list(runtime_dir.iterdir()), [])
            self.assertFalse(ledger.exists())

    def test_runtime_search_wrong_architecture_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            search_dir = root / "toolchain"
            runtime_dir.mkdir()
            search_dir.mkdir()
            binary = root / "xhdfe.plugin"
            binary.write_bytes(b"plugin")
            (search_dir / "libgomp-1.dll").write_bytes(b"libgomp")
            ledger = root / "runtime-ledger.json"
            objdump = root / "objdump.exe"
            objdump.write_bytes(b"objdump")
            with (
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_dependencies",
                    return_value={"libgomp-1.dll"},
                ),
                patch.object(
                    VALIDATOR_MODULE,
                    "_pe_architecture",
                    side_effect=lambda _objdump, path: (
                        "pei-i386"
                        if path.name.casefold() == "libgomp-1.dll"
                        else "pei-x86-64"
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "has wrong architecture"),
            ):
                VALIDATOR_MODULE.build_pe_runtime_ledger(
                    [binary], [], runtime_dir, ledger, objdump, [search_dir]
                )
            self.assertEqual(list(runtime_dir.iterdir()), [])
            self.assertFalse(ledger.exists())

    def test_missing_referenced_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            compiler = root / "toolchain" / "bin" / "g++.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            _write_cmake_toolchain(build_dir, compiler, "GNU")
            extension = root / "wheel" / "xhdfe" / "py_hdfe_v11.pyd"
            extension.parent.mkdir(parents=True)
            extension.write_bytes(b"extension")

            with (
                patch.object(
                    SETUP_MODULE,
                    "_find_mingw_objdump",
                    return_value=root / "objdump.exe",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_architecture",
                    return_value="i386:x86-64",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_dll_dependencies",
                    return_value={"libgomp-1.dll"},
                ),
                patch.object(
                    SETUP_MODULE,
                    "_compiler_runtime_resolution",
                    return_value=(None, None),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "depends on libgomp-1.dll"
                ):
                    SETUP_MODULE._bundle_mingw_runtime_dlls(
                        extension, build_dir
                    )

    def test_conflicting_transitive_runtime_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            compiler = root / "toolchain" / "bin" / "g++.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            _write_cmake_toolchain(build_dir, compiler, "GNU")

            extension = root / "wheel" / "xhdfe" / "py_hdfe_v11.pyd"
            extension.parent.mkdir(parents=True)
            extension.write_bytes(b"extension")

            runtime_a = root / "runtime-a"
            runtime_b = root / "runtime-b"
            runtime_a.mkdir()
            runtime_b.mkdir()
            parent_a = runtime_a / "parent-a.dll"
            parent_b = runtime_b / "parent-b.dll"
            shared_a = runtime_a / "shared.dll"
            shared_b = runtime_b / "shared.dll"
            parent_a.write_bytes(b"parent-a")
            parent_b.write_bytes(b"parent-b")
            shared_a.write_bytes(b"shared-a")
            shared_b.write_bytes(b"shared-b")

            def inspect_dependencies(_objdump, binary):
                if binary == extension:
                    return {"parent-a.dll", "parent-b.dll"}
                if binary.name.lower().startswith("parent-"):
                    return {"shared.dll"}
                return set()

            def find_runtime(_compiler, name, search_dirs=()):
                if search_dirs:
                    candidate = search_dirs[0] / name
                    if candidate.is_file():
                        return candidate, "requester-directory"
                path = {
                    "parent-a.dll": parent_a,
                    "parent-b.dll": parent_b,
                }.get(name.lower())
                return path, "compiler-print-file-name" if path else None

            with (
                patch.object(
                    SETUP_MODULE,
                    "_find_mingw_objdump",
                    return_value=root / "objdump.exe",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_architecture",
                    return_value="i386:x86-64",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_dll_dependencies",
                    side_effect=inspect_dependencies,
                ),
                patch.object(
                    SETUP_MODULE,
                    "_compiler_runtime_resolution",
                    side_effect=find_runtime,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Conflicting MinGW runtime DLL sources"
                ):
                    SETUP_MODULE._bundle_mingw_runtime_dlls(
                        extension, build_dir
                    )

    def test_missing_compiler_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            build_dir.mkdir()
            compiler = root / "toolchain" / "bin" / "g++.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            (build_dir / "CMakeCache.txt").write_text(
                f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n", encoding="utf-8"
            )
            extension = root / "wheel" / "xhdfe" / "py_hdfe_v11.pyd"
            extension.parent.mkdir(parents=True)
            extension.write_bytes(b"extension")

            with self.assertRaisesRegex(RuntimeError, "compiler identity"):
                SETUP_MODULE._bundle_mingw_runtime_dlls(extension, build_dir)

    def test_conflicting_stale_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            compiler = root / "toolchain" / "bin" / "g++.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            _write_cmake_toolchain(build_dir, compiler, "GNU")
            extension = root / "wheel" / "xhdfe" / "py_hdfe_v11.pyd"
            extension.parent.mkdir(parents=True)
            extension.write_bytes(b"extension")
            (extension.parent / "libstdc++-6.dll").write_bytes(b"stale")
            runtime = root / "libstdc++-6.dll"
            runtime.write_bytes(b"current")

            with (
                patch.object(
                    SETUP_MODULE,
                    "_find_mingw_objdump",
                    return_value=root / "objdump.exe",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_architecture",
                    return_value="i386:x86-64",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_dll_dependencies",
                    return_value={"libstdc++-6.dll"},
                ),
                patch.object(
                    SETUP_MODULE,
                    "_compiler_runtime_resolution",
                    return_value=(runtime, "compiler-print-file-name"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Conflicting MinGW"):
                    SETUP_MODULE._bundle_mingw_runtime_dlls(
                        extension, build_dir
                    )

    def test_non_gnu_windows_compiler_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            compiler = root / "cl.exe"
            compiler.write_bytes(b"compiler")
            _write_cmake_toolchain(build_dir, compiler, "MSVC")
            extension = root / "py_hdfe_v11.pyd"
            extension.write_bytes(b"extension")

            with patch.object(
                SETUP_MODULE,
                "_find_mingw_objdump",
                side_effect=AssertionError("MSVC must not use MinGW packaging"),
            ):
                self.assertEqual(
                    SETUP_MODULE._bundle_mingw_runtime_dlls(
                        extension, build_dir
                    ),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
