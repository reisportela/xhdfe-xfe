from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


MIN_CUDA_ARCH = 75


def _lower(value: str | None) -> str:
    return (value or "").strip().lower()


def _is_cuda_truthy(value: str | None) -> bool:
    return _lower(value) in {"1", "on", "true", "yes"}


def _is_cuda_falsey(value: str | None) -> bool:
    return _lower(value) in {"0", "off", "false", "no"}


def _find_nvcc() -> str | None:
    candidates: list[str] = []
    for name in ("CUDACXX", "NVCC"):
        if os.environ.get(name):
            candidates.append(os.environ[name])
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        exe = "nvcc.exe" if os.name == "nt" else "nvcc"
        candidates.append(str(Path(cuda_home) / "bin" / exe))
    if shutil.which("nvcc"):
        candidates.append("nvcc")

    for candidate in candidates:
        resolved = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if resolved and Path(resolved).exists():
            return str(Path(resolved).resolve())
    return None


def _find_nvidia_smi() -> str | None:
    resolved = shutil.which("nvidia-smi")
    if resolved:
        return resolved
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidate = candidate / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if candidate.exists():
            return str(candidate)
    return None


def _normalize_cuda_architectures(value: str) -> list[str]:
    arches: set[str] = set()
    for part in re.split(r"[,;\s]+", value):
        token = part.strip().lower().removeprefix("sm_")
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:\.(\d+))?", token)
        if not match:
            raise RuntimeError(
                f"Unsupported CUDA architecture value: {part!r}. "
                "Use values such as 75, 8.6, sm_90, or 75,80,86,89,90."
            )
        arch = int(match.group(1) + (match.group(2) or ""))
        if arch < MIN_CUDA_ARCH:
            raise RuntimeError(
                f"CUDA architecture {arch} is below xhdfe's minimum "
                f"supported architecture {MIN_CUDA_ARCH}."
            )
        arches.add(str(arch))
    if not arches:
        raise RuntimeError("No CUDA architecture was specified.")
    return sorted(arches, key=int)


def _detect_cuda_architectures() -> list[str]:
    nvidia_smi = _find_nvidia_smi()
    if not nvidia_smi:
        raise RuntimeError(
            "CUDA auto-detection requested, but nvidia-smi was not found. "
            "Set XHDFE_CUDA_ARCH explicitly, for example XHDFE_CUDA_ARCH=90."
        )

    cmd = [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        cmd[-1] = "csv,noheader"
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    detected: list[str] = []
    for line in result.stdout.splitlines():
        match = re.search(r"(\d+)\.(\d+)", line)
        if match:
            detected.append(f"{match.group(1)}{match.group(2)}")
    if not detected:
        raise RuntimeError("nvidia-smi did not report a usable CUDA compute capability.")
    return _normalize_cuda_architectures(" ".join(detected))


def _resolved_cmake_env() -> dict[str, str]:
    names = (
        "XHDFE_ENABLE_CUDA",
        "XHDFE_ENABLE_METAL",
        "XHDFE_ENABLE_MARCH_NATIVE",
        "XHDFE_GPU_BACKEND_DEFAULT",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_OSX_ARCHITECTURES",
    )
    env_values = {name: os.environ[name] for name in names if os.environ.get(name)}

    enable_cuda = os.environ.get("XHDFE_ENABLE_CUDA")
    if _is_cuda_falsey(enable_cuda):
        env_values["XHDFE_ENABLE_CUDA"] = "OFF"
        return env_values

    arch_source = (
        os.environ.get("CMAKE_CUDA_ARCHITECTURES")
        or os.environ.get("XHDFE_CUDA_ARCHS")
        or os.environ.get("XHDFE_CUDA_ARCH")
    )
    cuda_requested = (
        _lower(enable_cuda) == "auto"
        or _is_cuda_truthy(enable_cuda)
        or bool(arch_source)
    )
    if not cuda_requested:
        return env_values

    env_values["XHDFE_ENABLE_CUDA"] = "ON"
    should_detect_arch = (
        _lower(enable_cuda) == "auto"
        or _lower(arch_source) == "auto"
        or (_is_cuda_truthy(enable_cuda) and not arch_source)
    )
    if should_detect_arch:
        nvcc = _find_nvcc()
        if not nvcc:
            raise RuntimeError(
                "CUDA auto-detection requested, but nvcc was not found. "
                "Install the CUDA toolkit or set CUDACXX/NVCC/CUDA_HOME."
            )
        os.environ.setdefault("CUDACXX", nvcc)
        arches = _detect_cuda_architectures()
        env_values["CMAKE_CUDA_ARCHITECTURES"] = ";".join(arches)
        print(f"CUDA auto-detected for SM target(s): {', '.join(arches)}")
    elif arch_source:
        env_values["CMAKE_CUDA_ARCHITECTURES"] = ";".join(
            _normalize_cuda_architectures(arch_source)
        )

    return env_values


def _python_header_candidates() -> list[Path]:
    raw_paths: list[str] = []
    paths = sysconfig.get_paths()
    for key in ("include", "platinclude"):
        value = paths.get(key)
        if value:
            raw_paths.append(value)
    for key in ("INCLUDEPY", "CONFINCLUDEPY"):
        value = sysconfig.get_config_var(key)
        if value:
            raw_paths.append(str(value))

    candidates: list[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        key = str(path)
        if key not in seen:
            candidates.append(path)
            seen.add(key)
    return candidates


def _check_python_headers() -> None:
    candidates = _python_header_candidates()
    for include_dir in candidates:
        if (include_dir / "Python.h").is_file():
            return

    searched = "\n  - ".join(str(path) for path in candidates) or "(no include paths reported)"
    version = ".".join(str(part) for part in sys.version_info[:3])
    raise RuntimeError(
        "Python development headers are required to build xhdfe from source, "
        "but Python.h was not found for the active Python interpreter.\n\n"
        f"Python executable: {sys.executable}\n"
        f"Python version: {version}\n"
        "Searched include paths:\n"
        f"  - {searched}\n\n"
        "Install the headers for this exact Python, then rerun pip install. "
        "Examples: Debian/Ubuntu: sudo apt install python3-dev; "
        "Fedora/RHEL/Rocky: sudo dnf install python3-devel. "
        "On a cluster without sudo, use a conda/mamba environment or a Python "
        "module that includes development headers."
    )


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
        _check_python_headers()
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

        for name, value in _resolved_cmake_env().items():
            define = f"-D{name}="
            if not any(arg.startswith(define) for arg in cmake_args):
                cmake_args.append(f"{define}{value}")

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
