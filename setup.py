from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ("cmake-args=", None, "extra arguments passed to CMake configure"),
    ]

    def initialize_options(self) -> None:
        super().initialize_options()
        self.cmake_args = None

    def run(self) -> None:
        if shutil.which("cmake") is None:
            raise RuntimeError("CMake is required to build the xhdfe Python extension")
        super().run()

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        cfg = "Debug" if self.debug else os.environ.get("CMAKE_BUILD_TYPE", "Release")
        build_temp = Path(self.build_temp).resolve() / ext.name.replace(".", "_")
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={build_temp / 'lib'}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DXHDFE_BUILD_PYTHON=ON",
        ]
        if sys.platform == "darwin" and not os.environ.get("XHDFE_ENABLE_MARCH_NATIVE"):
            cmake_args.append("-DXHDFE_ENABLE_MARCH_NATIVE=OFF")
        if sys.platform.startswith("win"):
            cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}")

        for name in (
            "XHDFE_ENABLE_CUDA",
            "XHDFE_ENABLE_METAL",
            "XHDFE_ENABLE_MARCH_NATIVE",
            "XHDFE_GPU_BACKEND_DEFAULT",
            "CMAKE_CUDA_ARCHITECTURES",
            "CMAKE_OSX_ARCHITECTURES",
        ):
            if os.environ.get(name):
                define = f"-D{name}="
                if not any(arg.startswith(define) for arg in cmake_args):
                    cmake_args.append(f"{define}{os.environ[name]}")

        extra_args = self.cmake_args or os.environ.get("CMAKE_ARGS", "")
        if extra_args:
            cmake_args.extend(shlex.split(extra_args))

        build_args = ["--target", "py_hdfe_v11", "--config", cfg]
        parallel = self.parallel or os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        if parallel:
            build_args.append(f"-j{parallel}")

        subprocess.check_call(["cmake", "-S", ext.sourcedir, "-B", str(build_temp), *cmake_args])
        subprocess.check_call(["cmake", "--build", str(build_temp), *build_args])

        if not ext_fullpath.exists():
            candidates = []
            for suffix in ("*.so", "*.pyd", "*.dylib"):
                candidates.extend(extdir.glob(f"py_hdfe_v11{suffix}"))
            if candidates:
                shutil.copy2(candidates[0], ext_fullpath)
        if not ext_fullpath.exists():
            raise RuntimeError(f"CMake did not produce expected extension: {ext_fullpath}")


setup(
    ext_modules=[CMakeExtension("xhdfe.py_hdfe_v11")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
