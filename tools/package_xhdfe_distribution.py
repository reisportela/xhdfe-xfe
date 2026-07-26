#!/usr/bin/env python3
"""Create an autonomous xhdfe/xfe distribution bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, ignore=None) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copytree(src, dst, ignore=ignore)


def ignore_names(_dir_name: str, names: list[str]) -> set[str]:
    ignored = {".DS_Store", "__pycache__"}
    return ignored.intersection(names)


def ignore_stata_tools(_dir_name: str, names: list[str]) -> set[str]:
    ignored = {".DS_Store", "__pycache__", "_build"}
    return ignored.intersection(names)


def ignore_r_source(_dir_name: str, names: list[str]) -> set[str]:
    ignored = {".DS_Store", "__pycache__", "_problems"}
    ignored.update(
        name
        for name in names
        if name.endswith((".o", ".so")) or name.startswith("00LOCK")
    )
    return ignored.intersection(names)


def read_version(path: Path) -> str:
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    return first.replace("*! version ", "")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_python_module(build_dir: Path) -> Path:
    candidates = sorted(build_dir.glob("py_hdfe_v11*.so"))
    if not candidates:
        raise FileNotFoundError(f"py_hdfe_v11 module not found in {build_dir}")
    return candidates[0]


def load_python_validation(log_prefix: str) -> dict | None:
    path = REPO / "benchmarks" / "_out" / f"{log_prefix}_validate_python_backends.stdout"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    return json.loads(text)


def parse_v08_cpu(log_name: str) -> tuple[str, str]:
    path = REPO / "benchmarks" / "_out" / log_name
    if not path.exists():
        return ("not recorded", "not recorded")
    text = path.read_text(encoding="utf-8", errors="replace")
    elapsed = "not found"
    iterations = "not found"
    m = re.search(r"MWFE estimator converged in\s+([0-9]+)\s+iterations", text)
    if m:
        iterations = m.group(1)
    m = re.search(r"V08_QP_XHDFE_RESULT engine=xhdfe_cpu.*?elapsed=\s*([0-9.]+)", text, re.S)
    if m:
        elapsed = m.group(1)
    return (elapsed, iterations)


def copy_validation_logs(stage: Path, log_prefix: str) -> None:
    outdir = REPO / "benchmarks" / "_out"
    logs_dir = stage / "build_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(outdir.glob(f"{log_prefix}*")):
        if path.is_file():
            copy_file(path, logs_dir / path.name)
    for path in [
        REPO / "VALIDATE_STATA_BUNDLE.log",
        REPO / "VALIDATE_STATA_BUNDLE_CUDA.log",
    ]:
        if path.exists():
            copy_file(path, logs_dir / path.name)


def write_manifest(stage: Path) -> Path:
    manifest = stage / "MANIFEST.sha256"
    rows: list[str] = []
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        if path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(stage).as_posix()}")
    write_text(manifest, "\n".join(rows) + "\n")
    return manifest


def zip_stage(stage: Path) -> Path:
    zip_path = stage.with_suffix(".zip")
    if zip_path.exists():
        raise FileExistsError(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(stage.parent))
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    write_text(zip_path.with_suffix(zip_path.suffix + ".sha256"), f"{digest}  {zip_path.name}\n")
    write_text(REPO / "dist" / "latest_distribution_stage.txt", f"{stage.relative_to(REPO)}\n")
    write_text(REPO / "dist" / "latest_distribution_zip.txt", f"{zip_path.relative_to(REPO)}\n")
    write_text(REPO / "dist" / "latest_distribution_sha256.txt", f"{zip_path.with_suffix(zip_path.suffix + '.sha256').relative_to(REPO)}\n")
    return zip_path


def stage_bundle(args: argparse.Namespace) -> Path:
    bundle = f"xhdfe_xfe_distribution_{args.timestamp}_autonomous"
    stage = REPO / "dist" / bundle
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)

    for name in [
        "CMakeLists.txt",
        "README.md",
        "COMPILATION_NOTES.md",
        "QUICK_START.md",
        "LICENSE",
        "DISCLAIMER.md",
        "CITATION.cff",
        "xhdfe_py_hdfe_v11_help.html",
        "RELEASE_NOTES_2.21.0.20260725.md",
        "XHDFEGELBACH_EMPIRICAL_APPLICATION_COVERAGE_20260720.md",
        "XHDFEGELBACH_CODEX_REMEDIATION_REPORT_20260725.md",
        "XHDFEGELBACH_FUNCTIONAL_CLOSURE_20260725.md",
        "XHDFEGELBACH_RELEASE_CERTIFICATION_REPORT_20260725.md",
        "VALIDATE_GELBACH.py",
        "VALIDATE_GELBACH_ADVERSARIAL.py",
        "VALIDATE_GELBACH_FRONTENDS.py",
        "VALIDATE_GELBACH_HELP.py",
        "VALIDATE_GELBACH_PYFIXEST_FEATURES.py",
        "VALIDATE_GELBACH_SAMPLE_PROVENANCE.py",
        "VALIDATE_GELBACH_REMEDIATION_COVERAGE.py",
        "VALIDATE_AKM_KSS.py",
    ]:
        copy_file(REPO / name, stage / name)

    for dirname in ["include", "src", "python", "share", "third_party"]:
        copy_tree(REPO / dirname, stage / dirname, ignore=ignore_names)

    copy_tree(REPO / "tools", stage / "tools", ignore=ignore_names)
    copy_tree(REPO / "examples", stage / "examples", ignore=ignore_names)
    copy_tree(REPO / "r" / "xhdfe", stage / "r" / "xhdfe", ignore=ignore_r_source)
    copy_file(REPO / "r" / "README.md", stage / "r" / "README.md")

    evidence_src = REPO / "Remediation" / "xhdfegelbach_postaudit_20260723"
    evidence_dst = stage / "certification_evidence"
    for name in [
        "MC_BASE_SHARE_COVERAGE_DIAGNOSIS.md",
        "RCPP_OFFLINE_SOURCE_VALIDATION.md",
        "RCPP_PROVENANCE.md",
        "R_INTEGRATION_EVIDENCE.md",
    ]:
        copy_file(evidence_src / name, evidence_dst / name)
    for relative in [
        Path("evidence") / "absorbed_share_coverage_final.json",
        Path("evidence") / "base_share_coverage_final.json",
        Path("performance_30pct_20260723") / "cpu_pooled_complete_runs.json",
        Path("performance_30pct_20260723") / "cuda8_confirmatory" / "summary.json",
        Path("repros") / "f01_near_fe_band_mc.json",
        Path("repros") / "f01_near_fe_band_mc.py",
        Path("repros") / "performance_ab_30pct.py",
    ]:
        copy_file(evidence_src / relative, evidence_dst / relative)

    for dirname in ["include", "src"]:
        copy_tree(REPO / "stata" / dirname, stage / "stata" / dirname, ignore=ignore_names)
    copy_tree(REPO / "stata" / "tools", stage / "stata" / "tools", ignore=ignore_stata_tools)

    for name in [
        "README.md",
        "BUILD_CUDA.md",
        "stata.toc",
        "xhdfe.ado",
        "xhdfe.sthlp",
        "xhdfe.pkg",
        "xhdfe_p.ado",
        "xhdfe_estat.ado",
        "xhdfeakm.ado",
        "xhdfeakm.sthlp",
        "xhdfeconnected.ado",
        "xhdfeconnected.sthlp",
        "xhdfegelbach.ado",
        "xhdfegelbach.sthlp",
        "xhdfegelbachbootstrap.ado",
        "xhdfegelbachbootstrap.sthlp",
        "xhdfegelbachetable.ado",
        "xhdfegelbachetable.sthlp",
        "xhdfegelbachcoefplot.ado",
        "xhdfegelbachcoefplot.sthlp",
        "xhdfegpu.ado",
        "xhdfegpu.sthlp",
        "xhdfe_mobility_profile_main_95_21_ready.txt",
        "xfe.ado",
        "xfe.sthlp",
        "xfe.pkg",
    ]:
        copy_file(REPO / "stata" / name, stage / "stata" / name)

    prebuilt = args.prebuilt_dir
    copy_file(prebuilt / "xhdfe_cpu.plugin", stage / "stata" / "xhdfe.plugin")
    copy_file(prebuilt / "xfe_cpu.plugin", stage / "stata" / "xfe.plugin")
    stata_prebuilt = stage / "stata" / "prebuilt" / "linux_x86_64"
    for name in [
        "xhdfe_cpu.plugin",
        "xfe_cpu.plugin",
        "xhdfe_cuda_sm90.plugin",
        "xfe_cuda_sm90.plugin",
    ]:
        copy_file(prebuilt / name, stata_prebuilt / name)

    cpp_prebuilt = stage / "prebuilt" / "linux_x86_64"
    copy_file(args.cpu_build / "libxhdfe.a", cpp_prebuilt / "cpp" / "libxhdfe_cpu.a")
    copy_file(args.cuda_build / "libxhdfe.a", cpp_prebuilt / "cpp" / "libxhdfe_cuda_sm90.a")
    cpu_so = find_python_module(args.cpu_build)
    cuda_so = find_python_module(args.cuda_build)
    copy_file(cpu_so, cpp_prebuilt / "python" / ("cpu_" + cpu_so.name))
    copy_file(cuda_so, cpp_prebuilt / "python" / ("cuda_sm90_" + cuda_so.name))

    package_artifacts = stage / "packages"
    if args.python_wheel is not None:
        copy_file(
            args.python_wheel,
            package_artifacts / "python" / args.python_wheel.name,
        )
    if args.python_sdist is not None:
        copy_file(
            args.python_sdist,
            package_artifacts / "python" / args.python_sdist.name,
        )
    if args.r_tarball is not None:
        copy_file(
            args.r_tarball,
            package_artifacts / "R" / args.r_tarball.name,
        )

    for name in [
        "VALIDATE_STATA_BUNDLE.do",
        "VALIDATE_STATA_BUNDLE_CUDA.do",
        "VALIDATE_PYTHON_BACKENDS.py",
    ]:
        copy_file(REPO / name, stage / name)

    write_text(
        stata_prebuilt / "README.md",
        """# Linux x86_64 Stata prebuilt plugins

The default `stata/xhdfe.plugin` and `stata/xfe.plugin` files are CPU/OpenMP
builds. CUDA alternates target the local H100 server (`sm_90`).

To switch to CUDA:

```bash
cd stata
cp prebuilt/linux_x86_64/xhdfe_cuda_sm90.plugin xhdfe.plugin
cp prebuilt/linux_x86_64/xfe_cuda_sm90.plugin xfe.plugin
```

After switching plugins, run `discard` in Stata and require
`e(gpu_used)==1`, `e(gpu_backend)=="cuda"`, and `e(gpu_status)=="used"` on a
CUDA validation run.
""",
    )
    write_text(
        cpp_prebuilt / "README.md",
        """# Linux x86_64 C++/Python prebuilt artifacts

These convenience artifacts were built from the bundled sources on the release
host. Rebuild locally for another Python ABI or operating system.

- `cpp/libxhdfe_cpu.a`: CPU/OpenMP C++ library.
- `cpp/libxhdfe_cuda_sm90.a`: CUDA `sm_90` C++ library.
- `python/*.so`: Python 3.12 Linux x86_64 `py_hdfe_v11` modules.
""",
    )

    validation = load_python_validation(args.log_prefix)
    max_coef = "not recorded"
    max_se = "not recorded"
    py_gpu = "not recorded"
    if validation:
        max_coef = f"{validation['max_coef_diff']:.3g}"
        max_se = f"{validation['max_se_diff']:.3g}"
        py_gpu = str(validation["cuda"]["gpu_used"]).lower()

    v08_a, v08_iter_a = parse_v08_cpu(f"{args.log_prefix}_v08_perf_cpu_default.log")
    v08_b, v08_iter_b = parse_v08_cpu(f"{args.log_prefix}_v08_perf_cpu_default_repeat2.log")
    v08_head, v08_head_iter = parse_v08_cpu(f"{args.log_prefix}_v08_perf_cpu_headplugin_control.log")
    v08_old, v08_old_iter = parse_v08_cpu(f"{args.log_prefix}_v08_perf_cpu_previousdist_control.log")

    git_head = run_text(["git", "rev-parse", "--short", "HEAD"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_text(
        stage / "DIST_README.md",
        f"""# xhdfe + xfe autonomous distribution

Bundle: `{bundle}`
Date: {now}
Workspace commit: `{git_head}`

Versions:
- `xhdfe.ado`: `{read_version(REPO / 'stata' / 'xhdfe.ado')}`
- `xfe.ado`: `{read_version(REPO / 'stata' / 'xfe.ado')}`

This bundle is designed for offline sharing. It includes installable Stata
package material, C++/Python/Stata/R sources, build scripts, vendored Eigen,
pybind11, and Rcpp source inputs, Stata `stplugin` inputs, validation scripts,
Linux x86_64 prebuilt artifacts, release Python/R package archives when supplied,
and validation logs from the release host.

## Stata Use

```stata
adopath ++ "/path/to/{bundle}/stata"
discard
which xhdfe
which xfe
xhdfe y x1 x2, absorb(id1 id2) vce(cluster id1)
```

The default plugins are Linux x86_64 CPU/OpenMP builds. CUDA `sm_90` alternates
are under `stata/prebuilt/linux_x86_64/`.

## Offline Rebuild

```bash
# CPU/OpenMP
bash stata/tools/build-plugin.sh --linux --openmp
bash stata/tools/build-xfe-plugin.sh --linux --openmp

# CUDA for H100/sm_90
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash stata/tools/build-plugin.sh --linux --openmp
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash stata/tools/build-xfe-plugin.sh --linux --openmp

# C++/Python CPU
cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native" \
  -DXHDFE_ENABLE_CUDA=OFF -DXHDFE_BUILD_PYTHON=ON
cmake --build build_cpu --parallel

# C++/Python CUDA sm90
cmake -S . -B build_cuda_sm90 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native" \
  -DXHDFE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DXHDFE_BUILD_PYTHON=ON
cmake --build build_cuda_sm90 --parallel

# R CPU, fully offline
mkdir -p r/Rlib
R_PROFILE_USER=/dev/null R_LIBS_USER="$PWD/r/Rlib" \
  R CMD INSTALL --library="$PWD/r/Rlib" third_party/Rcpp_1.1.2.tar.gz
R_PROFILE_USER=/dev/null R_LIBS_USER="$PWD/r/Rlib" \
  XHDFE_ENABLE_CUDA=OFF R CMD INSTALL --library="$PWD/r/Rlib" r/xhdfe
```

Required local tools not bundled: Stata, a C++ compiler, CMake, Python, R, and
the CUDA toolkit/driver for CUDA builds. Windows 11 and macOS ARM users should
rebuild locally from these bundled sources and scripts; no Windows/macOS
prebuilt binaries were produced on this AlmaLinux host.

## Validation

Run:

```bash
python VALIDATE_PYTHON_BACKENDS.py --cpu-dir prebuilt/linux_x86_64/python --cuda-dir prebuilt/linux_x86_64/python
stata-mp -q -b do VALIDATE_STATA_BUNDLE.do
# after copying CUDA alternates over stata/*.plugin:
stata-mp -q -b do VALIDATE_STATA_BUNDLE_CUDA.do
```
""",
    )

    write_text(
        stage / "VALIDATION_SUMMARY.md",
        f"""# Validation Summary

Bundle: `{bundle}`
Date: {now}
Host: {platform.platform()}
Workspace commit: `{git_head}`

## Build Artifacts

- CMake CPU/Python build: `{args.cpu_build.name}`
- CMake CUDA/Python build: `{args.cuda_build.name}` with `sm_90`
- Stata CPU/OpenMP plugins: `xhdfe_cpu.plugin`, `xfe_cpu.plugin`
- Stata CUDA sm90 plugins: `xhdfe_cuda_sm90.plugin`, `xfe_cuda_sm90.plugin`

## Validation Results

- Python/C++ CPU and CUDA validation: passed.
- Python/C++ CUDA reported `gpu_used={py_gpu}`.
- CPU/CUDA coefficient max difference: `{max_coef}`.
- CPU/CUDA standard-error max difference: `{max_se}`.
- Stata CPU bundle validation: passed (`VALIDATE_STATA_BUNDLE.log`).
- Stata CUDA bundle validation: passed, including real GPU assertions
  (`e(gpu_used)==1`, `e(gpu_backend)=="cuda"`, `e(gpu_status)=="used"`).
- Final default Linux plugins link `libgomp` (see `build_logs/*ldd*final_cpu.log`).

## Large CPU Smoke

The V08/QP CPU smoke with the freshly rebuilt CPU plugin converged on both
release-host runs:

- run 1: `{v08_a}s`, MWFE iterations `{v08_iter_a}`
- run 2: `{v08_b}s`, MWFE iterations `{v08_iter_b}`

The host was heavily loaded during these runs. Under the same load, the
`HEAD` plugin control was `{v08_head}s` with MWFE iterations `{v08_head_iter}`.
An older 20260616 distribution plugin control was `{v08_old}s` but used MWFE
iterations `{v08_old_iter}`, so it is not a clean same-code baseline. These
results do not reproduce the historical no-OpenMP failure mode; the rejected
case was around 201s and missing OpenMP linkage.

The V08 script also attempts an `xhdfe_gpu` run. In the CPU-default smoke this
may fail because the default plugin is intentionally CPU/OpenMP. CUDA is
validated separately by `VALIDATE_STATA_BUNDLE_CUDA.do` with the CUDA sm90
alternates active.

## Platform Scope

Prebuilt binary artifacts in this ZIP are Linux x86_64 only. The ZIP includes
the source trees and scripts needed for offline rebuilds on Ubuntu, Windows 11,
and macOS ARM when the corresponding local toolchain is available.
""",
    )

    copy_validation_logs(stage, args.log_prefix)
    write_manifest(stage)
    return stage


def main() -> int:
    default_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=os.environ.get("XHDFE_DIST_TIMESTAMP", default_timestamp))
    parser.add_argument("--cpu-build", type=Path, required=True)
    parser.add_argument("--cuda-build", type=Path, required=True)
    parser.add_argument("--prebuilt-dir", type=Path, required=True)
    parser.add_argument("--log-prefix", required=True)
    parser.add_argument("--python-wheel", type=Path)
    parser.add_argument("--python-sdist", type=Path)
    parser.add_argument("--r-tarball", type=Path)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    args.cpu_build = args.cpu_build.resolve()
    args.cuda_build = args.cuda_build.resolve()
    args.prebuilt_dir = args.prebuilt_dir.resolve()
    for name in ("python_wheel", "python_sdist", "r_tarball"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    stage = stage_bundle(args)
    print(stage)
    if not args.no_zip:
        print(zip_stage(stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
