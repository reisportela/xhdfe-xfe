"""Persistent-cache invalidation when absorption solve semantics change."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from n05_receipt import N05ReceiptError, parse_n05_receipt


OLD_MODULE_SHA256 = "58860720525e5a38be51efce0984ef2de35949588fc2b2e9228dac0800522e39"
TIMEOUT_SECONDS = 420
FIXTURES = ("redundant_constant_exp40", "leverage_noise_exp40_offset1")
JOB_KEYS = frozenset(
    ("name", "role", "module", "module_sha256", "fixture", "mutation",
     "cache_mode", "cache_path")
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def digest_arrays(values: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        header = json.dumps(
            [name, value.dtype.str, list(value.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = memoryview(value).cast("B")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def redundant_fixture(mutation: str) -> dict:
    index = np.arange(64)
    cell = index // 16
    first, second = cell // 2, cell % 2
    x = (cell == 1).astype(float)
    noise = 0.125 * (2 * (index % 2) - 1)
    y = 0.75 * x + second - first + noise
    epsilon = Fraction(1, 2**40)
    weights = np.where(first == second, 1.0, float(epsilon))
    if mutation == "weight":
        weights = 2.0 * weights
    elif mutation != "none":
        raise ValueError(f"unsupported redundant-fixture mutation {mutation}")
    expected_variance = float((1 + epsilon) ** 2 / (960 * epsilon))
    return {
        "y": y,
        "X": x[:, None],
        "fes": [first, second, np.zeros(64, dtype=int)],
        "weights": weights,
        "se_type": "unadjusted",
        "expected_beta": 0.75,
        "expected_variance": expected_variance,
        "expected_residual": noise,
        "expected_rss": 1.0,
    }


def leverage_fixture(mutation: str) -> dict:
    index = np.arange(4096)
    cell = index // 256
    first, second = cell // 4, cell % 4
    sign_a = 2 * (index % 2) - 1
    sign_b = 2 * ((index // 2) % 2) - 1
    leverage = (first == 0) & (second == 1)
    x = sign_a * (1 + sign_b) / 2 * leverage
    small = 2.0**-40
    noise = np.where(leverage, small, 1.0) * sign_b
    expected_beta = 0.75
    y = expected_beta * x + (second - first) + noise
    if mutation == "y":
        y = y + 0.125 * x
        expected_beta += 0.125
    elif mutation != "none":
        raise ValueError(f"unsupported leverage-fixture mutation {mutation}")
    weights = (1 + (3 * first + second) % 7).astype(float)
    normalized = weights * (len(index) / weights.sum())
    return {
        "y": y,
        "X": x[:, None],
        "fes": [first, second],
        "weights": weights,
        "se_type": "robust",
        "expected_beta": expected_beta,
        "expected_variance": float(Fraction(32, 4088 * 2**80)),
        "expected_residual": noise,
        "expected_rss": float(normalized @ (noise * noise)),
    }


def fixture(name: str, mutation: str) -> dict:
    if name == "redundant_constant_exp40":
        return redundant_fixture(mutation)
    if name == "leverage_noise_exp40_offset1":
        return leverage_fixture(mutation)
    raise ValueError(f"unknown fixture {name}")


def configure_environment(cache_mode: str, cache_path: str) -> None:
    for name in tuple(os.environ):
        if name.startswith("XHDFE_"):
            os.environ.pop(name)
    os.environ.update(
        XHDFE_GPU_BACKEND="cpu",
        XHDFE_CERTIFY="1",
        XHDFE_ABSORPTION_CACHE_MODE=cache_mode,
        XHDFE_ABSORPTION_CACHE=cache_path,
        XHDFE_MOBILITY_MODE="off",
        XHDFE_FE_NORMALIZE="component",
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create native-module specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_digest(model) -> str:
    values = {
        "beta": np.asarray(model.coef_),
        "V": np.asarray(model.covariance_),
        "residual": np.asarray(model.residuals_),
        "scalars": np.asarray(
            [model.rss_, model.num_iterations_, model.absorption_method_used,
             model.df_resid_, model.gpu_status_code_], dtype=float
        ),
    }
    return digest_arrays(values)


def worker(job_path: Path) -> int:
    job = json.loads(job_path.read_text())
    if not isinstance(job, dict) or frozenset(job) != JOB_KEYS:
        raise ValueError("worker job keys differ from the frozen contract")
    if job["role"] not in ("old", "candidate"):
        raise ValueError("worker role must be old or candidate")
    if job["cache_mode"] not in ("read", "write"):
        raise ValueError("cache mode must be read or write")
    module_path = Path(job["module"]).resolve()
    if sha256(module_path) != job["module_sha256"]:
        raise RuntimeError("native module custody mismatch")
    configure_environment(job["cache_mode"], job["cache_path"])
    data = fixture(job["fixture"], job["mutation"])
    inputs = {"y": data["y"], "X": data["X"], "weights": data["weights"]}
    inputs.update({f"fe{j}": value for j, value in enumerate(data["fes"])})
    input_before = digest_arrays(inputs)
    cpp = load_module(module_path)
    model = cpp.HdfeRegressor(
        num_threads=2, fit_intercept=False, drop_singletons=False,
        max_iter=1000, tol=1e-8, tolerance_mode="reghdfe-comparable",
        se_type=data["se_type"],
    )
    result = {
        "name": job["name"], "role": job["role"], "fixture": job["fixture"],
        "mutation": job["mutation"], "module": str(module_path),
        "module_sha256": job["module_sha256"], "input_sha_before": input_before,
    }
    try:
        model.fit(
            data["y"], data["X"], fes=data["fes"], weights=data["weights"]
        )
        beta = float(model.coef_[0])
        variance = float(model.covariance_[0, 0])
        residual = np.asarray(model.residuals_)
        result.update(
            rc=0, beta=beta, variance=variance, rss=float(model.rss_),
            beta_error=abs(beta - data["expected_beta"]),
            variance_relative_error=abs(variance / data["expected_variance"] - 1.0),
            residual_error=float(np.max(np.abs(residual - data["expected_residual"]))),
            rss_error=abs(float(model.rss_) - data["expected_rss"]),
            iterations=int(model.num_iterations_), method=int(model.absorption_method_used),
            converged=bool(model.converged_), certified=bool(model.precision_certified_),
            gpu_used=bool(model.gpu_used_), output_sha256=output_digest(model),
        )
    except Exception as error:
        result.update(rc=1, error=str(error), error_type=type(error).__name__)
    cache_path = Path(job["cache_path"])
    result.update(
        input_sha_after=digest_arrays(inputs),
        module_sha256_after=sha256(module_path),
        cache_exists=cache_path.is_file(),
        cache_sha256=sha256(cache_path) if cache_path.is_file() else None,
    )
    print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    return 0


def clean_environment() -> dict:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("XHDFE_")
    }
    environment.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1"
    )
    return environment


def run_job(script: Path, directory: Path, job: dict) -> tuple[dict, str, dict]:
    directory.mkdir(exist_ok=False)
    job_path = directory / "job.json"
    write_new(job_path, job)
    started = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-B", str(script), "--worker-job", str(job_path)],
        text=True, capture_output=True, env=clean_environment(), check=False,
        timeout=TIMEOUT_SECONDS,
    )
    elapsed = time.monotonic() - started
    (directory / "stdout.log").write_text(process.stdout)
    (directory / "stderr.log").write_text(process.stderr)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if process.returncode or len(lines) != 1:
        raise RuntimeError(
            f"worker {job['name']} rc={process.returncode}, stdout_lines={len(lines)}"
        )
    result = json.loads(lines[0])
    write_new(directory / "result.json", result)
    process_record = {
        "returncode": process.returncode, "elapsed_seconds": elapsed,
        "timeout_seconds": TIMEOUT_SECONDS,
    }
    write_new(directory / "process.json", process_record)
    return result, process.stderr, process_record


def make_job(name: str, role: str, module: Path, module_sha: str,
             fixture_name: str, mutation: str, cache_mode: str,
             cache_path: Path) -> dict:
    return {
        "name": name, "role": role, "module": str(module),
        "module_sha256": module_sha, "fixture": fixture_name,
        "mutation": mutation, "cache_mode": cache_mode,
        "cache_path": str(cache_path.resolve()),
    }


def old_error_verified(result: dict) -> bool:
    if result.get("rc") != 0 or not result.get("converged") or not result.get("certified"):
        return False
    if result["fixture"] == "redundant_constant_exp40":
        return result["beta_error"] > 0.5 and result["residual_error"] > 0.5
    return result["variance_relative_error"] > 1.0


def candidate_correct(result: dict) -> bool:
    return bool(
        result.get("rc") == 0 and result.get("converged") and result.get("certified")
        and not result.get("gpu_used") and result["beta_error"] <= 1e-9
        and result["variance_relative_error"] <= 1e-8
        and result["residual_error"] <= 1e-8 and result["rss_error"] <= 1e-8
    )


def validate_result(job: dict, result: dict, stderr: str,
                    expected_cache_hit: bool | None) -> tuple[str, list, object]:
    errors = []
    if result.get("module_sha256") != job["module_sha256"] or \
       result.get("module_sha256_after") != job["module_sha256"]:
        errors.append("module identity changed")
    if result.get("input_sha_before") != result.get("input_sha_after"):
        errors.append("fit mutated its inputs")
    if not result.get("cache_exists") or not result.get("cache_sha256"):
        errors.append("expected cache file is absent")
    receipt = None
    if job["role"] == "old":
        try:
            parse_n05_receipt(stderr, required=False)
        except N05ReceiptError as error:
            errors.append(str(error))
        if not old_error_verified(result):
            errors.append("old producer did not reproduce the pre-fix estimator error")
        status = "EXPECTED_OLD_ERROR" if not errors else "FAIL"
    else:
        try:
            receipt = parse_n05_receipt(
                stderr, required=True, expected_entrypoint="fit",
                expected_mode="reghdfe-comparable", expected_backend="cpu",
            )
        except N05ReceiptError as error:
            errors.append(str(error))
        if receipt is not None:
            if receipt["cache_hit"] is not expected_cache_hit:
                errors.append("receipt cache_hit differs from the planned operation")
            if receipt["identity_match"] is not True:
                errors.append("candidate receipt lacks final result identity")
        if not candidate_correct(result):
            errors.append("candidate result differs from the frozen analytic reference")
        status = "PASS" if not errors else "FAIL"
    return status, errors, receipt


def plans(old: Path, old_sha: str, candidate: Path | None,
          candidate_sha: str | None, output: Path, preflight: bool) -> list:
    entries = []
    for fixture_name in FIXTURES:
        old_cache = output / "caches" / f"{fixture_name}.old.bin"
        entries.append((make_job(
            f"{fixture_name}__old_write", "old", old, old_sha,
            fixture_name, "none", "write", old_cache), None))
        if preflight:
            continue
        new_cache = output / "caches" / f"{fixture_name}.candidate.bin"
        entries.extend([
            (make_job(f"{fixture_name}__candidate_read_old", "candidate",
                      candidate, candidate_sha, fixture_name, "none", "read", old_cache), False),
            (make_job(f"{fixture_name}__candidate_write", "candidate",
                      candidate, candidate_sha, fixture_name, "none", "write", new_cache), False),
            (make_job(f"{fixture_name}__candidate_read", "candidate",
                      candidate, candidate_sha, fixture_name, "none", "read", new_cache), True),
            (make_job(f"{fixture_name}__candidate_mutated", "candidate",
                      candidate, candidate_sha, fixture_name,
                      "weight" if fixture_name.startswith("redundant") else "y",
                      "read", new_cache), False),
        ])
    return entries


def controller(old: Path, candidate: Path | None, output: Path,
               preflight: bool) -> int:
    old = old.resolve()
    if not old.is_file() or sha256(old) != OLD_MODULE_SHA256:
        raise ValueError("--old-module is not the frozen N1 CPU module")
    candidate_sha = None
    if not preflight:
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise ValueError("--candidate-module is not a file")
        candidate_sha = sha256(candidate)
        if candidate_sha == OLD_MODULE_SHA256:
            raise ValueError("candidate module must differ from the frozen old producer")
    output = output.resolve()
    if output.exists():
        raise ValueError("--out must be a new directory")
    output.mkdir(parents=True)
    (output / "cases").mkdir()
    (output / "caches").mkdir()
    script = Path(__file__).resolve()
    plan = plans(old, OLD_MODULE_SHA256, candidate, candidate_sha, output, preflight)
    reference_dir = output.parent.parent / "baseline_n1_supplemental"
    manifest = {
        "schema": "xhdfe-solve-generation-cache-v1",
        "mode": "old_preflight" if preflight else "full_contract",
        "fit_quota": len(plan), "old_module": str(old),
        "old_module_sha256": OLD_MODULE_SHA256,
        "candidate_module": str(candidate) if candidate else None,
        "candidate_module_sha256": candidate_sha,
        "script_sha256": sha256(script),
        "references": {
            name: sha256(reference_dir / name)
            for name in ("redundant_exact_n1.json", "LOW_NOISE_EXACT_REFERENCE.json",
                         "LOW_NOISE_T01_ADJUDICATION.json")
        },
        "old_boundary": "XHDFE_CERTIFY participates in the cache key; N05 receipt/counter absent",
        "jobs": [job for job, _hit in plan],
    }
    write_new(output / "manifest.json", manifest)
    rows, results = [], {}
    for job, expected_hit in plan:
        try:
            result, stderr, process = run_job(script, output / "cases" / job["name"], job)
            status, errors, receipt = validate_result(job, result, stderr, expected_hit)
            results[job["name"]] = result
        except Exception as error:
            result, process, receipt = {}, {}, None
            status, errors = "FAIL", [f"{type(error).__name__}: {error}"]
        rows.append({
            "name": job["name"], "role": job["role"], "fixture": job["fixture"],
            "mutation": job["mutation"], "status": status, "errors": errors,
            "expected_cache_hit": expected_hit, "receipt": receipt,
            "metrics": {key: result.get(key) for key in
                        ("beta", "variance", "beta_error", "variance_relative_error",
                         "residual_error", "rss_error", "cache_sha256", "output_sha256")},
            "process": process,
        })
    cross_checks = []
    if not preflight and len(results) == len(plan):
        for fixture_name in FIXTURES:
            prefix = fixture_name + "__"
            old_write = results[prefix + "old_write"]
            read_old = results[prefix + "candidate_read_old"]
            write = results[prefix + "candidate_write"]
            read = results[prefix + "candidate_read"]
            mutated = results[prefix + "candidate_mutated"]
            checks = {
                "old_cache_not_overwritten": old_write["cache_sha256"] == read_old["cache_sha256"],
                "candidate_cache_not_overwritten": len({write["cache_sha256"], read["cache_sha256"],
                                                         mutated["cache_sha256"]}) == 1,
                "generation_separates_cache_bytes": old_write["cache_sha256"] != write["cache_sha256"],
                "cold_candidate_results_agree": read_old["output_sha256"] == write["output_sha256"],
                "candidate_hit_results_agree": write["output_sha256"] == read["output_sha256"],
                "old_wrong_result_not_reused": old_write["output_sha256"] != read_old["output_sha256"],
            }
            cross_checks.append({
                "fixture": fixture_name,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
            })
    valid_rows = all(row["status"] in (("EXPECTED_OLD_ERROR",) if preflight else
                     ("EXPECTED_OLD_ERROR", "PASS")) for row in rows)
    valid_cross = preflight or (
        len(cross_checks) == len(FIXTURES) and all(row["status"] == "PASS" for row in cross_checks)
    )
    verdict = "PASS_EXPECTED_OLD_ERROR" if preflight and valid_rows else \
              "PASS" if valid_rows and valid_cross else "FAIL"
    report = {
        "schema": "xhdfe-solve-generation-cache-report-v1",
        "mode": manifest["mode"], "verdict": verdict,
        "counts": {
            "fits": len(rows),
            "expected_old_errors": sum(row["status"] == "EXPECTED_OLD_ERROR" for row in rows),
            "candidate_pass": sum(row["status"] == "PASS" for row in rows),
            "fail": sum(row["status"] == "FAIL" for row in rows),
        },
        "rows": rows, "cross_checks": cross_checks,
        "interpretation": "Old wrong outputs are negative controls; only candidate analytic correctness plus receipt cache_hit proves the new generation contract.",
    }
    write_new(output / "report.json", report)
    print(json.dumps({"verdict": verdict, **report["counts"]}, sort_keys=True))
    return int(verdict == "FAIL")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-module", type=Path)
    parser.add_argument("--candidate-module", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--old-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job:
        return worker(args.worker_job)
    if args.old_module is None or args.out is None:
        parser.error("--old-module and --out are required")
    if not args.old_preflight and args.candidate_module is None:
        parser.error("--candidate-module is required for the full contract")
    return controller(args.old_module, args.candidate_module, args.out, args.old_preflight)


if __name__ == "__main__":
    raise SystemExit(main())
