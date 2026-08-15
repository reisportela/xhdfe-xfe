from __future__ import annotations

import filecmp
import hashlib
import json
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

# CPython 3.8+ does not use PATH as a general search path for extension-module
# dependencies. Keep every non-system MinGW dependency beside the built .pyd.
_WINDOWS_HOST_DLL_RE = re.compile(
    r"^(?:api|ext)-ms-win-.+\.dll$",
    re.IGNORECASE,
)
_PYTHON_HOST_DLL_NAMES = frozenset(
    {
        "python3.dll",
        f"python{sys.version_info.major}{sys.version_info.minor}.dll",
        f"python{sys.version_info.major}{sys.version_info.minor}_d.dll",
    }
)
# Conservative Windows 10/11 system-provider policy, informed by delvewheel's
# versioned baseline lists. Unknown names are deliberately treated as package
# dependencies and must be resolved or the build fails closed.
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

_WINDOWS_RUNTIME_MANIFEST = "_windows_runtime_manifest.json"
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


def _is_windows_host_dll(dll_name: str) -> bool:
    """Return whether Windows or the active CPython provides this dependency."""

    name = Path(dll_name).name.lower()
    return (
        name in _WINDOWS_HOST_DLL_NAMES
        or name in _PYTHON_HOST_DLL_NAMES
        or bool(_WINDOWS_HOST_DLL_RE.fullmatch(name))
    )


def _windows_runtime_license(dll_name: str) -> str | None:
    name = Path(dll_name).name
    for pattern, license_expression in _WINDOWS_RUNTIME_LICENSES:
        if pattern.fullmatch(name):
            return license_expression
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _cmake_cache_value(build_dir: Path, name: str) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None

    prefix = f"{name}:"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix) and "=" in line:
            value = line.split("=", 1)[1].strip().strip('"')
            return value or None
    return None


def _cmake_cxx_compiler_id(build_dir: Path) -> str | None:
    compiler_files = sorted(
        (build_dir / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake")
    )
    pattern = re.compile(r'^set\(CMAKE_CXX_COMPILER_ID\s+"([^"]+)"\)')
    for compiler_file in compiler_files:
        for line in compiler_file.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            match = pattern.match(line.strip())
            if match:
                return match.group(1)
    return None


def _find_mingw_objdump(compiler: Path) -> Path | None:
    names: list[str] = []
    compiler_name = compiler.name
    compiler_name_lower = compiler_name.lower()
    if compiler_name_lower.endswith("g++.exe"):
        names.append(f"{compiler_name[:-len('g++.exe')]}objdump.exe")
    elif compiler_name_lower.endswith("g++"):
        names.append(f"{compiler_name[:-len('g++')]}objdump")
    names.extend(("objdump.exe", "objdump"))

    for name in names:
        adjacent = compiler.parent / name
        if adjacent.is_file():
            return adjacent
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def _pe_dll_dependencies(objdump: Path, binary: Path) -> set[str]:
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
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(
            f"Could not inspect Windows DLL dependencies for {binary.name}: {detail}"
        )

    dependencies: set[str] = set()
    pattern = re.compile(r"^\s*DLL Name:\s*(\S+)\s*$", re.IGNORECASE)
    for line in result.stdout.splitlines():
        match = pattern.match(line)
        if match:
            dependencies.add(match.group(1))
    return dependencies


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
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(
            f"Could not inspect Windows architecture for {binary.name}: {detail}"
        )
    match = re.search(
        r"\b(pe(?:i)?-[a-z0-9_-]+)\s*$",
        result.stdout,
        re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(
            f"Could not identify Windows architecture for {binary.name}."
        )
    return match.group(1).lower()


def _compiler_runtime_path(
    compiler: Path, dll_name: str, search_dirs: tuple[Path, ...] = ()
) -> Path | None:
    path, _ = _compiler_runtime_resolution(compiler, dll_name, search_dirs)
    return path


def _compiler_runtime_resolution(
    compiler: Path, dll_name: str, search_dirs: tuple[Path, ...] = ()
) -> tuple[Path | None, str | None]:
    for directory in search_dirs:
        candidate = directory / dll_name
        if candidate.is_file():
            return candidate.resolve(), "requester-directory"

    result = subprocess.run(
        [str(compiler), f"-print-file-name={dll_name}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        errors="replace",
    )
    if result.returncode == 0:
        reported = result.stdout.strip().strip('"')
        if reported and reported.lower() != dll_name.lower():
            candidate = Path(reported)
            if candidate.is_file():
                return candidate.resolve(), "compiler-print-file-name"

    adjacent = compiler.parent / dll_name
    if adjacent.is_file():
        return adjacent.resolve(), "compiler-adjacent"
    return None, None


def _bundle_mingw_runtime_dlls(extension: Path, build_dir: Path) -> list[Path]:
    compiler_id = _cmake_cxx_compiler_id(build_dir)
    if compiler_id is None:
        raise RuntimeError(
            "CMake did not record the Windows C++ compiler identity; refusing "
            "to create a wheel with unverifiable runtime dependencies."
        )
    if compiler_id != "GNU":
        return []

    compiler_value = _cmake_cache_value(build_dir, "CMAKE_CXX_COMPILER")
    if not compiler_value:
        raise RuntimeError(
            "CMake selected GNU on Windows but did not record CMAKE_CXX_COMPILER; "
            "refusing to create a wheel with unverifiable runtime dependencies."
        )
    compiler = Path(compiler_value)
    objdump = _find_mingw_objdump(compiler)
    if objdump is None:
        raise RuntimeError(
            "A MinGW-built Python extension requires objdump to identify its "
            "runtime DLL closure, but objdump was not found beside the active "
            f"compiler ({compiler}) or on PATH."
        )

    extension_architecture = _pe_architecture(objdump, extension)
    pending = sorted(
        (name, None)
        for name in _pe_dll_dependencies(objdump, extension)
        if not _is_windows_host_dll(name)
    )
    bundled: list[Path] = []
    resolved_sources: dict[str, Path] = {}
    runtime_records: dict[str, dict[str, object]] = {}

    while pending:
        dll_name, requester_dir = pending.pop(0)
        key = dll_name.lower()

        search_dirs = () if requester_dir is None else (requester_dir,)
        source, resolution_method = _compiler_runtime_resolution(
            compiler, dll_name, search_dirs
        )
        previous_source = resolved_sources.get(key)
        if previous_source is not None:
            if source is not None and not filecmp.cmp(
                previous_source, source, shallow=False
            ):
                raise RuntimeError(
                    f"Conflicting MinGW runtime DLL sources for {dll_name}: "
                    f"{previous_source} and {source}. Windows loads DLLs by "
                    "basename, so the wheel cannot safely contain both."
                )
            continue
        if source is None:
            raise RuntimeError(
                f"The MinGW-built extension depends on {dll_name}, but the active "
                f"compiler ({compiler}) could not locate that runtime DLL. "
                "Refusing to create a wheel that would fail at import time."
            )
        source_architecture = _pe_architecture(objdump, source)
        if source_architecture != extension_architecture:
            raise RuntimeError(
                f"The MinGW-built extension is {extension_architecture}, but "
                f"{source.name} is {source_architecture}. Refusing to bundle a "
                "runtime DLL for a different architecture."
            )
        license_expression = _windows_runtime_license(dll_name) or "UNMAPPED"
        resolved_sources[key] = source

        destination = extension.parent / dll_name
        if destination.exists():
            if not filecmp.cmp(source, destination, shallow=False):
                raise RuntimeError(
                    f"Conflicting MinGW runtime DLL already exists: {destination}. "
                    "Remove the stale build output and rebuild."
                )
        else:
            shutil.copy2(source, destination)
        source_sha256 = _sha256(source)
        member_sha256 = _sha256(destination)
        if member_sha256 != source_sha256:
            raise RuntimeError(
                f"Bundled MinGW runtime DLL {dll_name} does not match its source."
            )
        bundled.append(destination)

        dependencies = sorted(
            _pe_dll_dependencies(objdump, source), key=str.casefold
        )
        runtime_records[key] = {
            "architecture": source_architecture,
            "dependencies": dependencies,
            "license": license_expression,
            "member": destination.name,
            "member_sha256": member_sha256,
            "resolution_method": resolution_method,
            "size": destination.stat().st_size,
            "source_path": str(source),
            "source_sha256": source_sha256,
        }
        for dependency in dependencies:
            if not _is_windows_host_dll(dependency):
                pending.append((dependency, source.parent))

    extension_dependencies = sorted(
        _pe_dll_dependencies(objdump, extension), key=str.casefold
    )
    manifest = {
        "format": "xhdfe-windows-runtime-closure-v1",
        "roots": [
            {
                "architecture": extension_architecture,
                "dependencies": extension_dependencies,
                "member": extension.name,
                "member_sha256": _sha256(extension),
                "size": extension.stat().st_size,
            }
        ],
        "runtimes": [runtime_records[key] for key in sorted(runtime_records)],
    }
    manifest_path = extension.parent / _WINDOWS_RUNTIME_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return bundled


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

        if sys.platform.startswith("win"):
            bundled = _bundle_mingw_runtime_dlls(ext_fullpath, build_temp)
            if bundled:
                names = ", ".join(sorted(path.name for path in bundled))
                print(f"Bundled MinGW runtime DLLs beside the extension: {names}")


setup(
    ext_modules=[CMakeExtension("xhdfe.py_hdfe_v11")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
