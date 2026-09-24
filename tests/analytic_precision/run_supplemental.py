#!/usr/bin/env python3
"""Orchestrate the frozen T01 supplemental contracts without weakening refusals."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run import clean_json, write_json
from run_r import library_record, validate_tree


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUPPLEMENTAL = HERE / "supplemental"
SOURCE_MANIFEST = HERE / "supplemental_cases.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_record(path):
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(ROOT):
        raise RuntimeError(f"native module is absent: {path}")
    return dict(path=str(path), sha256=sha(path))


def validate_sources():
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    for record in manifest["files"]:
        target = HERE / record["target"]
        if not target.is_file() or sha(target) != record["target_sha256"]:
            raise RuntimeError(f"supplemental source custody mismatch: {target}")
    return manifest


def prepare(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise RuntimeError("supplemental output must be a new directory inside the selective tree")
    sources = validate_sources()
    modules = {"cpu": module_record(args.module_cpu),
               "cuda": module_record(args.module_cuda)}
    libraries = {"cpu": library_record(args.library_cpu),
                 "cuda": library_record(args.library_cuda)}
    dependencies = [str(path.resolve()) for path in args.dependency_library]
    if len(dependencies) != len(set(dependencies)) or any(not Path(path).is_dir() for path in dependencies):
        raise RuntimeError("dependency libraries must be distinct existing directories")
    stata_root = args.stata_package_root.resolve()
    if not stata_root.is_relative_to(ROOT):
        raise RuntimeError("Stata package root must be inside the selective tree")
    stata_ado = stata_root / "stata/xhdfe.ado"
    stata_plugin = stata_root / "stata/xhdfe.plugin"
    if not stata_ado.is_file() or not stata_plugin.is_file():
        raise RuntimeError(f"isolated Stata package is incomplete: {stata_root}")
    rscript = args.rscript.resolve()
    stata = args.stata.resolve()
    if not rscript.is_file() or not stata.is_file():
        raise RuntimeError("Rscript/Stata executable is absent")

    python_suites = (
        "savefe_no_intercept_contract.py",
        "savefe_extended_normalization_contract.py",
        "ordinary_weak_weights_contract.py",
        "ordinary_leverage_noise_contract.py",
        "ordinary_redundant_fe_contract.py",
    )
    jobs = []
    for backend in ("cpu", "cuda"):
        for script in python_suites:
            jobs.append(dict(id=f"python__{Path(script).stem}__{backend}",
                             kind="python", backend=backend, script=script))
        jobs.append(dict(id=f"r__savefe_extended_normalization_contract__{backend}",
                         kind="r", backend=backend,
                         script="savefe_extended_normalization_contract.R"))
    jobs.append(dict(id="stata__pure_slope_stata_semantics__cpu", kind="stata",
                     backend="cpu", script="pure_slope_stata_semantics.do"))

    output.mkdir()
    (output / "attempts").mkdir()
    (output / "tmp").mkdir()
    runner_files = [Path(__file__), HERE / "run_r.py", HERE / "run.py",
                    SOURCE_MANIFEST, *[SUPPLEMENTAL / job["script"] for job in jobs]]
    manifest = dict(
        schema=1,
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        certificate_mode=args.certificate_mode,
        jobs=jobs,
        modules=modules,
        libraries=libraries,
        dependency_libraries=dependencies,
        stata_package=dict(root=str(stata_root), ado=str(stata_ado),
                           ado_sha256=sha(stata_ado), plugin=str(stata_plugin),
                           plugin_sha256=sha(stata_plugin)),
        executables={"rscript": {"path": str(rscript), "sha256": sha(rscript)},
                     "stata": {"path": str(stata), "sha256": sha(stata)}},
        files={str(path.resolve()): sha(path) for path in dict.fromkeys(runner_files)},
        source_manifest_sha256=sha(SOURCE_MANIFEST),
        source_manifest=sources,
        valid_refusal_is_failure=True,
        historical_evidence_is_not_an_oracle=True,
        subset_is_full_G1=False,
        joint_fit_metric_is_gate=False,
        required_unexecuted_gates=["main T01 default/audit", "R main matrix/formula",
                                   "core24 x 8", "group/individual 3M performance"],
    )
    write_json(output / "manifest.json", manifest)
    print("T01_SUPPLEMENTAL_PREPARED", len(jobs), "jobs")


def validate_manifest(manifest):
    validate_sources()
    for path, expected in manifest["files"].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError(f"supplemental campaign source changed: {path}")
    for record in manifest["modules"].values():
        if sha(record["path"]) != record["sha256"]:
            raise RuntimeError(f"native module changed: {record['path']}")
    for record in manifest["libraries"].values():
        validate_tree(record["package"], record["package_files"])
        if sha(record["dll"]) != record["dll_sha256"]:
            raise RuntimeError(f"R DLL changed: {record['dll']}")
    stata = manifest["stata_package"]
    if sha(stata["ado"]) != stata["ado_sha256"] or sha(stata["plugin"]) != stata["plugin_sha256"]:
        raise RuntimeError("Stata package changed")
    for record in manifest["executables"].values():
        if sha(record["path"]) != record["sha256"]:
            raise RuntimeError(f"executable changed: {record['path']}")


def run(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    validate_manifest(manifest)
    jobs = [job for job in manifest["jobs"]
            if (args.backend is None or job["backend"] == args.backend)
            and (not args.filter or args.filter in job["id"])]
    for job in jobs:
        target = output / "attempts" / job["id"]
        if target.exists():
            raise RuntimeError(f"attempt already exists: {target}")
        target.mkdir()
        scratch = output / "tmp" / job["id"]
        scratch.mkdir()
        for name in ("cuda-cache", "xdg-cache", "r-user-cache"):
            (scratch / name).mkdir()
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("XHDFE_")}
        if manifest.get("certificate_mode") == "audit":
            env["XHDFE_CERTIFY"] = "1"
        env.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
                   CUDA_CACHE_PATH=str(scratch / "cuda-cache"),
                   XDG_CACHE_HOME=str(scratch / "xdg-cache"),
                   R_USER_CACHE_DIR=str(scratch / "r-user-cache"),
                   OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        script = SUPPLEMENTAL / job["script"]
        result_path = target / "result.json"
        if job["kind"] == "python":
            module = manifest["modules"][job["backend"]]
            command = [sys.executable, "-B", str(script), "--module", module["path"],
                       "--backend", job["backend"], "--out", str(result_path)]
        elif job["kind"] == "r":
            library = manifest["libraries"][job["backend"]]
            env["R_LIBS_USER"] = library["library"]
            command = [manifest["executables"]["rscript"]["path"], "--vanilla", str(script),
                       library["library"], os.pathsep.join(manifest["dependency_libraries"]),
                       job["backend"], str(result_path), library["package"], library["dll"]]
        else:
            command = [manifest["executables"]["stata"]["path"], "-q", "-b", "do",
                       str(script), manifest["stata_package"]["root"]]
        log_path = target / "process.log"
        with log_path.open("x") as log:
            try:
                completed = subprocess.run(command, cwd=target, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=args.timeout)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
        validate_manifest(manifest)
        marker_ok = True
        custody_marker = True
        if job["kind"] == "stata":
            logs = list(target.glob("*.log"))
            log_text = "\n".join(path.read_text(errors="replace") for path in logs)
            marker_ok = "T01_PURE_SLOPE_PASS" in log_text
            custody_marker = "T01_PURE_SLOPE_START" in log_text
        output_ok = result_path.is_file() if job["kind"] != "stata" else marker_ok
        receipt = dict(job=job, command=command, returncode=returncode,
                       output_present=output_ok, marker_ok=marker_ok,
                       custody_marker=custody_marker,
                       success=returncode == 0 and output_ok,
                       process_log_sha256=sha(log_path),
                       result_path=str(result_path) if result_path.is_file() else None,
                       result_sha256=sha(result_path) if result_path.is_file() else None)
        write_json(target / "receipt.json", receipt)
        print("T01_SUPPLEMENTAL", job["id"], "PASS" if receipt["success"] else "FAIL")
    return 0


def report(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    validate_manifest(manifest)
    rows = []
    for job in manifest["jobs"]:
        receipt_path = output / "attempts" / job["id"] / "receipt.json"
        if not receipt_path.is_file():
            rows.append(dict(job=job, verdict="COVERAGE_MISSING"))
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("success"):
            verdict = "PASS"
        elif receipt.get("returncode") == 124:
            verdict = "COVERAGE_MISSING"
        elif receipt.get("result_sha256"):
            result_path = Path(receipt["result_path"])
            if sha(result_path) != receipt["result_sha256"]:
                verdict = "HARNESS_FAILURE"
            else:
                result = json.loads(result_path.read_text())
                text = json.dumps(result)
                verdict = ("VERIFIER_FAILURE" if "FAIL_VALID_REFUSAL" in text
                           and '"verdict": "FAIL"' not in text
                           else "REAL_ESTIMATOR_FAILURE")
        elif receipt.get("custody_marker"):
            attempt = output / "attempts" / job["id"]
            log_text = "\n".join(path.read_text(errors="replace")
                                 for path in attempt.glob("*.log"))
            verdict = ("VERIFIER_FAILURE" if any(token in log_text.lower()
                       for token in ("precision could not be established", "independent check", "certificate"))
                       else "REAL_ESTIMATOR_FAILURE")
        else:
            verdict = "HARNESS_FAILURE"
        rows.append(dict(job=job, verdict=verdict, receipt=receipt))
    counts = Counter(row["verdict"] for row in rows)
    scoped_pass = bool(rows) and set(counts) == {"PASS"}
    summary = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                   certificate_mode=manifest.get("certificate_mode", "default"),
                   planned_rows=len(manifest["jobs"]),
                   completed_rows=sum(row["verdict"] != "COVERAGE_MISSING" for row in rows),
                   counts=dict(counts), scoped_supplemental_pass=scoped_pass,
                   subset_is_full_G1=False, core24_x8_still_required=True,
                   rows=rows)
    target = output / f'report_{len(list(output.glob("report_*.json"))) + 1:03d}.json'
    write_json(target, clean_json(summary))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print("T01_SUPPLEMENTAL_REPORT", target)
    return 0 if scoped_pass else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-cpu", type=Path)
    parser.add_argument("--module-cuda", type=Path)
    parser.add_argument("--library-cpu", type=Path)
    parser.add_argument("--library-cuda", type=Path)
    parser.add_argument("--dependency-library", type=Path, action="append", default=[])
    parser.add_argument("--stata-package-root", type=Path)
    parser.add_argument("--rscript", type=Path, default=Path("/bin/Rscript"))
    parser.add_argument("--stata", type=Path, default=Path("/usr/local/stata/stata-mp"))
    parser.add_argument("--certificate-mode", choices=("default", "audit"), default="default")
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--filter")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    prepared = (args.module_cpu, args.module_cuda, args.library_cpu, args.library_cuda,
                args.stata_package_root)
    if args.action == "prepare" and any(value is None for value in prepared):
        parser.error("prepare requires CPU/CUDA modules, R libraries and Stata package root")
    if args.action != "prepare" and (any(value is not None for value in prepared)
                                      or args.dependency_library
                                      or args.certificate_mode != "default"):
        parser.error("run/report use paths and certificate mode frozen by prepare")
    if args.action != "run" and (args.backend is not None or args.filter):
        parser.error("--backend/--filter apply only to run")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    return {"prepare": prepare, "run": run, "report": report}[args.action](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
