#!/usr/bin/env python3
"""Prepare, run and report the T01 R matrix/formula campaign.

The xhdfe package and DLL are pinned. Explicit dependency libraries are added
ahead of the read-only system libraries; system/base R libraries remain usable.
"""
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

import pandas as pd

from evaluate_fail_closed import assess as historical_assess
from evaluate_t01 import assess_t01
from run import clean_json, write_json


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
R_WORKER = HERE / "r_cases.R"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(directory):
    directory = Path(directory).resolve()
    return {str(path.relative_to(directory)): sha(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def validate_tree(directory, expected):
    current = tree_hashes(directory)
    if current != expected:
        missing = sorted(set(expected) - set(current))
        extra = sorted(set(current) - set(expected))
        changed = sorted(key for key in set(current) & set(expected)
                         if current[key] != expected[key])
        raise RuntimeError(f"R xhdfe package custody mismatch; missing={missing}, extra={extra}, changed={changed}")


def library_record(path):
    library = Path(path).resolve()
    if not library.is_relative_to(ROOT):
        raise RuntimeError(f"xhdfe library must be inside the selective tree: {library}")
    package = library / "xhdfe"
    if not package.is_dir():
        raise RuntimeError(f"isolated xhdfe package is absent under {library}")
    dlls = list((package / "libs").glob("xhdfe.*"))
    dlls = [path.resolve() for path in dlls if path.is_file()]
    if len(dlls) != 1:
        raise RuntimeError(f"expected one installed xhdfe DLL under {package/'libs'}")
    return dict(library=str(library), package=str(package.resolve()),
                dll=str(dlls[0]), dll_sha256=sha(dlls[0]),
                package_files=tree_hashes(package))


def prepare(args):
    campaign = args.campaign.resolve()
    if not campaign.is_relative_to(ROOT):
        raise RuntimeError("source campaign must be inside the selective tree")
    source_manifest = json.loads((campaign / "manifest.json").read_text())
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise RuntimeError("R T01 output must remain inside the selective tree")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    records = {"cpu": library_record(args.library_cpu)}
    if args.library_cuda:
        records["cuda"] = library_record(args.library_cuda)
    dependency_libraries = [str(path.resolve()) for path in args.dependency_library]
    if len(dependency_libraries) != len(set(dependency_libraries)):
        raise RuntimeError("duplicate dependency library")
    for path in map(Path, dependency_libraries):
        if not path.is_dir():
            raise RuntimeError(f"dependency library is absent: {path}")
    rscript = args.rscript.resolve()
    if not rscript.is_file():
        raise RuntimeError(f"Rscript is absent: {rscript}")

    native_jobs = [job for job in source_manifest["jobs"]
                   if job.get("engine") == "xhdfe" and job.get("interface") == "native"]
    jobs = []
    for old in native_jobs:
        if old["backend"] not in records:
            continue
        case_kind = json.loads(
            (campaign / "fixtures" / old["case"] / "case.json").read_text()
        )["kind"]
        for interface in ("r-matrix", "r-formula"):
            job = dict(old, interface=interface,
                       fixture=str((campaign / "fixtures" / old["case"]).resolve()))
            job["id"] = old["id"].replace("__native__", f"__{interface}__")
            if job.get("expected_rejection"):
                job["reject_pattern"] += r"|absorptionmethod\(lsmr/mlsmr\) is CPU-only"
                if (job["backend"] == "cpu" and job["method"] == "schwarz"
                        and job.get("rejection_status") and case_kind == "standard"):
                    job["reject_pattern"] = r"absorptionmethod\(schwarz\).*not supported"
            jobs.append(job)
    if not jobs:
        raise RuntimeError("source campaign has no eligible native xhdfe jobs")

    output.mkdir()
    (output / "csv").mkdir()
    (output / "attempts").mkdir()
    (output / "tmp").mkdir()
    fixture_hashes = {}
    for case, files in source_manifest["fixtures"].items():
        fixture = campaign / "fixtures" / case
        current = {name: sha(fixture / name) for name in files}
        if current != files:
            raise RuntimeError(f"fixture custody mismatch: {case}")
        target = output / "csv" / f"{case}.csv"
        pd.read_stata(fixture / "long.dta").to_csv(
            target, index=False, float_format="%.17g")
        fixture_hashes[case] = dict(source=files, csv_sha256=sha(target))

    files = [R_WORKER, HERE / "evaluate.py", HERE / "evaluate_fail_closed.py",
             HERE / "evaluate_t01.py", HERE / "precision_contract.py", Path(__file__),
             campaign / "manifest.json", rscript]
    manifest = dict(
        schema=1,
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_campaign=str(campaign),
        source_manifest_sha256=sha(campaign / "manifest.json"),
        certificate_mode=source_manifest.get("certificate_mode", "default"),
        libraries=records,
        dependency_libraries=dependency_libraries,
        rscript=dict(path=str(rscript), sha256=sha(rscript)),
        jobs=jobs,
        csv=str((output / "csv").resolve()),
        output=str((output / "attempts").resolve()),
        fixtures=fixture_hashes,
        files={str(path.resolve()): sha(path) for path in files},
        historical_verdict_preserved=True,
        joint_fit_metric_is_gate=False,
        required_unexecuted_gates=["supplemental controls", "core24 x 8",
                                   "group/individual 3M performance"],
    )
    write_json(output / "manifest.json", manifest)
    print("T01_R_PREPARED", len(jobs), "jobs")


def validate_manifest(output, manifest):
    for path, expected in manifest["files"].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError(f"R T01 source custody mismatch: {path}")
    campaign = Path(manifest["source_campaign"])
    if sha(campaign / "manifest.json") != manifest["source_manifest_sha256"]:
        raise RuntimeError("source campaign manifest changed")
    for case, record in manifest["fixtures"].items():
        fixture = campaign / "fixtures" / case
        if any(sha(fixture / name) != expected
               for name, expected in record["source"].items()):
            raise RuntimeError(f"source fixture changed: {case}")
        if sha(output / "csv" / f"{case}.csv") != record["csv_sha256"]:
            raise RuntimeError(f"R CSV changed: {case}")
    for record in manifest["libraries"].values():
        validate_tree(record["package"], record["package_files"])
        if sha(record["dll"]) != record["dll_sha256"]:
            raise RuntimeError(f"R DLL changed: {record['dll']}")


def run(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    validate_manifest(output, manifest)
    backend = args.backend
    if backend not in manifest["libraries"]:
        raise RuntimeError(f"backend was not prepared: {backend}")
    record = manifest["libraries"][backend]
    scratch = output / "tmp" / backend
    scratch.mkdir(exist_ok=False)
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
               R_LIBS_USER=record["library"], OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    dependencies = os.pathsep.join(manifest["dependency_libraries"])
    command = [manifest["rscript"]["path"], "--vanilla", str(R_WORKER),
               record["library"], dependencies, str(output / "manifest.json"),
               backend, record["package"], record["dll"]]
    log_path = output / f"run_{backend}.log"
    with log_path.open("x") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=args.timeout)
    validate_manifest(output, manifest)
    expected = [job for job in manifest["jobs"] if job["backend"] == backend]
    present = [job for job in expected
               if (output / "attempts" / job["id"] / "raw.json").is_file()]
    receipt = dict(backend=backend, command=command, returncode=completed.returncode,
                   expected_rows=len(expected), raw_rows=len(present),
                   package=record["package"], dll=record["dll"],
                   dll_sha256=sha(record["dll"]), log_sha256=sha(log_path))
    write_json(output / f"run_{backend}_receipt.json", receipt)
    if completed.returncode != 0 or len(present) != len(expected):
        raise RuntimeError(f"R worker incomplete: {receipt}")
    print("T01_R_RUN_COMPLETE", backend, len(present))


def report(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    validate_manifest(output, manifest)
    campaign = Path(manifest["source_campaign"])
    rows = []
    raw_hashes = {}
    for job in manifest["jobs"]:
        raw_path = output / "attempts" / job["id"] / "raw.json"
        if not raw_path.is_file():
            rows.append(dict(job, verdict="NOT_RUN",
                             t01=dict(applicable=True, g1_verdict="COVERAGE_MISSING",
                                      passed=False, reason="R raw absent")))
            continue
        raw = json.loads(raw_path.read_text())
        raw_hashes[job["id"]] = sha(raw_path)
        library = manifest["libraries"][job["backend"]]
        try:
            if (Path(str(raw.get("module", ""))).resolve() != Path(library["dll"])
                    or Path(str(raw.get("package", ""))).resolve() != Path(library["package"])):
                historical = dict(job, verdict="HARNESS_ERROR",
                                  grading_error="R loaded package/DLL path differs from manifest")
            else:
                historical = historical_assess(job, raw, Path(job["fixture"]))
        except Exception as error:
            historical = dict(job, verdict="HARNESS_ERROR", grading_error=str(error))
        try:
            t01 = assess_t01(job, raw, Path(job["fixture"]), historical)
        except Exception as error:
            t01 = dict(applicable=True, g1_verdict="HARNESS_FAILURE",
                       passed=False, reason="R T01 grading error",
                       grading_error=str(error))
        historical["result"] = {key: value for key, value in raw.items()
                                if key not in ("residuals", "groups", "recovered_fe")}
        historical["t01"] = t01
        rows.append(historical)
    counts = Counter(row["verdict"] for row in rows)
    t01_counts = Counter(row["t01"]["g1_verdict"] for row in rows)
    t01_applicable = [row["t01"] for row in rows if row["t01"].get("applicable")]
    historical_pass = bool(rows) and set(counts) <= {"PASS", "UNSUPPORTED_EXPECTED"}
    t01_pass = bool(t01_applicable) and all(
        row["g1_verdict"] in {"PASS", "UNSUPPORTED_EXPECTED"}
        for row in t01_applicable)
    summary = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                   source_manifest_sha256=manifest["source_manifest_sha256"],
                   certificate_mode=manifest.get("certificate_mode", "default"),
                   planned_rows=len(manifest["jobs"]), completed_rows=sum(r["verdict"] != "NOT_RUN" for r in rows),
                   historical_counts=dict(counts), t01_counts=dict(t01_counts),
                   historical_pass=historical_pass, t01_pass=t01_pass,
                   scoped_R_pass=historical_pass and t01_pass,
                   raw_sha256=raw_hashes,
                   subset_is_full_G1=False, core24_x8_still_required=True,
                   rows=rows)
    target = output / f'report_{len(list(output.glob("report_*.json"))) + 1:03d}.json'
    write_json(target, clean_json(summary))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print("T01_R_REPORT", target)
    return 0 if summary["scoped_R_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--library-cpu", type=Path)
    parser.add_argument("--library-cuda", type=Path)
    parser.add_argument("--dependency-library", type=Path, action="append", default=[])
    parser.add_argument("--rscript", type=Path, default=Path("/bin/Rscript"))
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.campaign is None or args.library_cpu is None:
            parser.error("prepare requires --campaign and --library-cpu")
    elif args.action == "run":
        if args.backend is None:
            parser.error("run requires --backend")
        if any(value is not None for value in
               (args.campaign, args.library_cpu, args.library_cuda)) or args.dependency_library:
            parser.error("run uses paths frozen by prepare")
    elif any(value is not None for value in
             (args.campaign, args.backend, args.library_cpu, args.library_cuda)) or args.dependency_library:
        parser.error("report uses paths frozen by prepare")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    return {"prepare": prepare, "run": run, "report": report}[args.action](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
