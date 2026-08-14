"""Tests for fail-closed MinGW runtime bundling in the Python wheel."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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


class WindowsRuntimePackagingTest(unittest.TestCase):
    def test_directory_without_packaged_gnu_runtime_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            (package_dir / "unrelated.dll").write_bytes(b"runtime")

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
            (package_dir / "libgomp-1.dll").write_bytes(b"runtime")
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
        self.assertEqual(
            dependencies,
            {"KERNEL32.dll", "libgomp-1.dll", "libstdc++-6.dll"},
        )

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
                },
                "libwinpthread-1.dll": {"libgcc_s_seh-1.dll"},
                "libgcc_s_seh-1.dll": set(),
                "libatomic-1.dll": {"libgcc_s_seh-1.dll"},
            }

            def inspect_dependencies(_objdump, binary):
                return dependencies[binary.name.lower()]

            def find_runtime(_compiler, name):
                return runtimes.get(name.lower())

            with (
                patch.object(
                    SETUP_MODULE,
                    "_find_mingw_objdump",
                    return_value=root / "objdump.exe",
                ),
                patch.object(
                    SETUP_MODULE,
                    "_pe_dll_dependencies",
                    side_effect=inspect_dependencies,
                ),
                patch.object(
                    SETUP_MODULE,
                    "_compiler_runtime_path",
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
                    "_pe_dll_dependencies",
                    return_value={"libgomp-1.dll"},
                ),
                patch.object(
                    SETUP_MODULE, "_compiler_runtime_path", return_value=None
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "depends on libgomp-1.dll"
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
                    "_pe_dll_dependencies",
                    return_value={"libstdc++-6.dll"},
                ),
                patch.object(
                    SETUP_MODULE,
                    "_compiler_runtime_path",
                    return_value=runtime,
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
