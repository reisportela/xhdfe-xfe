"""Bounded N05 gate, receipt, cache, and immutable-result contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Tuple

import numpy as np

from n05_receipt import N05ReceiptError, parse_n05_receipt, require_family_passes


WORKER_TIMEOUT_SECONDS = 420
PAIR_EPS_MULTIPLIER = 64
PAIR_ATOL = PAIR_EPS_MULTIPLIER * np.finfo(float).eps
PAIR_RTOL = PAIR_ATOL
FLOAT_FIELDS = ("beta", "V", "residuals", "rss")
EXACT_FIELDS = (
    "iterations",
    "method",
    "backend_actual",
    "gpu_used",
    "omitted",
    "sample_index",
    "nobs",
    "df_resid",
    "input_sha_before",
)
PAIR_FIXTURES = (
    ("ordinary_homo_constant", "reghdfe-comparable"),
    ("ordinary_homo_noconstant", "xhdfe-fast"),
    ("ordinary_robust", "reghdfe-comparable"),
    ("ordinary_robust_aweight", "reghdfe-comparable"),
    ("ordinary_oneway", "reghdfe-comparable"),
    ("ordinary_multiway", "xhdfe-fast"),
    ("ordinary_collinear", "reghdfe-comparable"),
    ("ordinary_fweight_robust", "xhdfe-fast"),
    ("iv_homo", "reghdfe-comparable"),
    ("iv_robust", "xhdfe-fast"),
    ("ordinary_slope", "reghdfe-comparable"),
    ("group_sum", "reghdfe-comparable"),
    ("group_mean", "xhdfe-fast"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: object, *, allow_nan: bool = False) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=allow_nan)
        output.write("\n")


def _worker_job(
    *,
    name: str,
    module: Path,
    module_sha256: str,
    backend: str,
    fixture: str,
    mode: str,
    certify: str,
    old_flags: bool = False,
    cache_mode: str = "off",
    cache_path: str = "",
    mutation: str = "none",
) -> dict:
    return dict(
        name=name,
        module=str(module),
        module_sha256=module_sha256,
        backend=backend,
        fixture=fixture,
        mode=mode,
        certify=certify,
        old_flags=old_flags,
        cache_mode=cache_mode,
        cache_path=cache_path,
        mutation=mutation,
    )


def _plan(module: Path, module_sha: str, backend: str, output: Path) -> list:
    plan = []

    def add(job: dict, *, cache_hit=None) -> None:
        plan.append(dict(job=job, expected_cache_hit=cache_hit))

    for fixture, mode in PAIR_FIXTURES:
        add(
            _worker_job(
                name=f"{fixture}__off",
                module=module,
                module_sha256=module_sha,
                backend=backend,
                fixture=fixture,
                mode=mode,
                certify="unset",
            )
        )
        add(
            _worker_job(
                name=f"{fixture}__audit",
                module=module,
                module_sha256=module_sha,
                backend=backend,
                fixture=fixture,
                mode=mode,
                certify="1",
            ),
            cache_hit=False,
        )

    selector = dict(
        module=module,
        module_sha256=module_sha,
        backend=backend,
        fixture="ordinary_homo_constant",
        mode="reghdfe-comparable",
    )
    add(_worker_job(name="selector_zero", certify="0", **selector))
    add(_worker_job(name="legacy_flags_without_selector", certify="unset", old_flags=True, **selector))
    add(
        _worker_job(name="legacy_flags_with_audit", certify="1", old_flags=True, **selector),
        cache_hit=False,
    )

    if backend == "cpu":
        cache_dir = output / "caches"
        default_cache = str((cache_dir / "default.bin").resolve())
        audit_cache = str((cache_dir / "audit.bin").resolve())
        cache_common = dict(
            module=module,
            module_sha256=module_sha,
            backend=backend,
            fixture="ordinary_cache",
            mode="reghdfe-comparable",
        )
        add(
            _worker_job(
                name="cache_default_write",
                certify="unset",
                cache_mode="write",
                cache_path=default_cache,
                **cache_common,
            )
        )
        add(
            _worker_job(
                name="cache_default_read",
                certify="unset",
                cache_mode="read",
                cache_path=default_cache,
                **cache_common,
            )
        )
        add(
            _worker_job(
                name="cache_audit_read_default",
                certify="1",
                cache_mode="read",
                cache_path=default_cache,
                **cache_common,
            ),
            cache_hit=False,
        )
        add(
            _worker_job(
                name="cache_audit_write",
                certify="1",
                cache_mode="write",
                cache_path=audit_cache,
                **cache_common,
            ),
            cache_hit=False,
        )
        add(
            _worker_job(
                name="cache_audit_read",
                certify="1",
                cache_mode="read",
                cache_path=audit_cache,
                **cache_common,
            ),
            cache_hit=True,
        )
        add(
            _worker_job(
                name="cache_audit_changed_y",
                certify="1",
                cache_mode="read",
                cache_path=audit_cache,
                mutation="y",
                **cache_common,
            ),
            cache_hit=False,
        )
        add(
            _worker_job(
                name="cache_audit_changed_weight",
                certify="1",
                cache_mode="read",
                cache_path=audit_cache,
                mutation="weight",
                **cache_common,
            ),
            cache_hit=False,
        )
    return plan


def _family_policy(fixture: str) -> Tuple[Tuple[str, ...], dict, dict]:
    bounded = ("PASS", "BOUND_INCONCLUSIVE")
    if fixture == "ordinary_slope":
        statuses = {
            name: ("UNSUPPORTED",)
            for name in ("projection", "homoskedastic", "sandwich", "full_v")
        }
        reasons = {name: "heterogeneous_slopes" for name in statuses}
        return (), statuses, reasons
    if fixture.startswith("group_"):
        statuses = {
            name: ("UNSUPPORTED",)
            for name in ("projection", "homoskedastic", "sandwich", "full_v")
        }
        reasons = {name: "group_individual" for name in statuses}
        return (), statuses, reasons
    if fixture.startswith("iv_"):
        statuses = {
            "projection": bounded,
            "homoskedastic": ("UNSUPPORTED",),
            "sandwich": ("UNSUPPORTED",),
            "full_v": ("UNSUPPORTED",),
        }
        reasons = {name: "iv_score_design" for name in ("homoskedastic", "sandwich", "full_v")}
        return (), statuses, reasons
    if fixture == "ordinary_multiway":
        return (
            (),
            {
                "projection": bounded,
                "homoskedastic": ("UNSUPPORTED",),
                "sandwich": ("UNSUPPORTED",),
                "full_v": ("UNSUPPORTED",),
            },
            {
                "homoskedastic": "vce_not_homoskedastic",
                "sandwich": "multiway",
                "full_v": "multiway",
            },
        )
    if fixture == "ordinary_oneway":
        return (
            ("projection", "sandwich", "full_v"),
            {
                "projection": ("PASS",),
                "homoskedastic": ("UNSUPPORTED",),
                "sandwich": ("PASS",),
                "full_v": ("PASS",),
            },
            {"homoskedastic": "vce_not_homoskedastic"},
        )
    if fixture == "ordinary_robust":
        return (
            ("projection", "sandwich"),
            {
                "projection": ("PASS",),
                "homoskedastic": ("UNSUPPORTED",),
                "sandwich": ("PASS",),
                "full_v": ("UNSUPPORTED",),
            },
            {"homoskedastic": "vce_not_homoskedastic"},
        )
    if fixture in ("ordinary_robust_aweight", "ordinary_fweight_robust"):
        return (
            (),
            {
                "projection": bounded,
                "homoskedastic": ("UNSUPPORTED",),
                "sandwich": bounded,
                "full_v": ("UNSUPPORTED",),
            },
            {"homoskedastic": "vce_not_homoskedastic"},
        )
    exact_homoskedastic = fixture == "ordinary_homo_constant"
    return (
        ("projection", "homoskedastic") if exact_homoskedastic else (),
        {
            "projection": ("PASS",) if exact_homoskedastic else bounded,
            "homoskedastic": ("PASS",) if exact_homoskedastic else bounded,
            "sandwich": ("UNSUPPORTED",),
            "full_v": ("UNSUPPORTED",),
        },
        {},
    )


def _clean_environment() -> dict:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("XHDFE_")}
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "2"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _parse_worker_stdout(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"worker emitted {len(lines)} nonempty stdout lines")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise ValueError("worker result must be a JSON object")
    return value


def _run_worker(worker: Path, case_dir: Path, job: dict) -> Tuple[dict, dict]:
    case_dir.mkdir(exist_ok=False)
    job_path = case_dir / "job.json"
    _write_new(job_path, job)
    started = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-B", str(worker), str(job_path)],
        text=True,
        capture_output=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        env=_clean_environment(),
        check=False,
    )
    elapsed = time.monotonic() - started
    (case_dir / "stdout.log").write_text(process.stdout)
    (case_dir / "stderr.log").write_text(process.stderr)
    process_record = {
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "timeout_seconds": WORKER_TIMEOUT_SECONDS,
    }
    _write_new(case_dir / "process.json", process_record)
    result = _parse_worker_stdout(process.stdout)
    _write_new(case_dir / "result.json", result, allow_nan=True)
    return result, dict(process_record, stderr=process.stderr)


def _validate_functional(result: dict, process: dict, backend: str) -> list:
    errors = []
    if process["returncode"] != 0:
        errors.append(f"worker return code {process['returncode']}")
    if result.get("rc") != 0:
        errors.append(f"fit failed: {result.get('error', 'no error text')}")
        return errors
    if result["input_sha_before"] != result["input_sha_after"]:
        errors.append("worker inputs changed during the native call")
    if not result["converged"] or not result["precision_certified"]:
        errors.append("fit did not return converged and N1-certified results")
    if not np.isfinite(np.asarray(result["beta"], dtype=float)).all():
        errors.append("returned coefficients are not finite")
    if not np.isfinite(np.asarray(result["V"], dtype=float)).all():
        errors.append("returned covariance is not finite")
    if not np.isfinite(np.asarray(result["residuals"], dtype=float)).all():
        errors.append("returned residuals are not finite")
    if not np.isfinite(result["rss"]):
        errors.append("returned RSS is not finite")
    expected_length = result["structural_columns"] + int(result["fit_intercept"])
    if len(result["beta"]) != expected_length:
        errors.append("constant/no-constant coefficient shape changed")
    if result["omitted_count"] != result["expected_omitted"]:
        errors.append("retained/dropped-column decision changed")
    beta_limit = 1e-8 if result["fixture"].startswith("group_") else 1e-9
    if result["beta_error"] is not None and (
        not np.isfinite(result["beta_error"]) or result["beta_error"] > beta_limit
    ):
        errors.append(f"analytic beta error {result['beta_error']:.3e} exceeds {beta_limit:.1e}")
    residual_limit = 1e-7 if result["fixture"].startswith("group_") else 1e-8
    if not np.isfinite(result["residual_error"]) or result["residual_error"] > residual_limit:
        errors.append(
            f"analytic residual error {result['residual_error']:.3e} exceeds {residual_limit:.1e}"
        )
    rss_limit = 1e-8 + 1e-10 * abs(result["rss"])
    if not np.isfinite(result["rss_error"]) or result["rss_error"] > rss_limit:
        errors.append(f"analytic RSS error {result['rss_error']:.3e} exceeds {rss_limit:.3e}")
    if backend == "cuda":
        if not result["gpu_used"] or result["gpu_status"] != 1 or result["backend_actual"] != "cuda":
            errors.append("CUDA case did not report real GPU use")
    elif result["gpu_used"] or result["backend_actual"] != "cpu":
        errors.append("CPU case unexpectedly reported GPU use")
    return errors


def _validate_receipt(result: dict, process: dict, job: dict, expected_cache_hit) -> Tuple[object, list, str]:
    enabled = job["certify"] == "1"
    errors = []
    receipt = None
    expected_entrypoint = "group_fit" if job["fixture"].startswith("group_") else "fit"
    try:
        receipt = parse_n05_receipt(
            process["stderr"],
            required=enabled,
            expected_entrypoint=expected_entrypoint if enabled else None,
            expected_mode=job["mode"] if enabled else None,
            expected_backend=job["backend"] if enabled else None,
        )
    except N05ReceiptError as error:
        errors.append(str(error))
    if not enabled:
        if result.get("n05_work_count") != 0:
            errors.append("private N05 work counter advanced with the exact gate disabled")
        return receipt, errors, "NOT_REQUESTED"
    if receipt is None:
        return receipt, errors, "FAIL"
    if receipt["iterations"] != result.get("iterations"):
        errors.append("receipt iterations do not match the returned final result")
    if expected_cache_hit is None or receipt["cache_hit"] is not expected_cache_hit:
        errors.append("receipt cache_hit differs from the planned cache operation")
    expected_identity = not (
        job["fixture"].startswith("iv_")
        or job["fixture"].startswith("group_")
        or job["fixture"] == "ordinary_slope"
    )
    if receipt["identity_match"] is not expected_identity:
        errors.append(
            f"identity_match={receipt['identity_match']} differs from expected {expected_identity}"
        )
    required, allowed_statuses, reasons = _family_policy(job["fixture"])
    for family, allowed in allowed_statuses.items():
        if receipt["families"][family]["status"] not in allowed:
            errors.append(
                f"{family} status {receipt['families'][family]['status']} not in {allowed}"
            )
    for family, reason in reasons.items():
        if receipt["families"][family]["reason"] != reason:
            errors.append(
                f"{family} reason {receipt['families'][family]['reason']!r} != {reason!r}"
            )
    try:
        require_family_passes(receipt, required)
    except N05ReceiptError as error:
        errors.append(str(error))
    applicable = [
        family
        for family, allowed in allowed_statuses.items()
        if allowed != ("UNSUPPORTED",)
    ]
    if applicable and result.get("n05_work_count", 0) <= 0:
        errors.append("enabled supported fit did not advance the private N05 work counter")
    if not applicable:
        proof_status = "NOT_APPLICABLE"
    elif all(receipt["families"][family]["status"] == "PASS" for family in applicable):
        proof_status = "PASS"
    elif any(
        receipt["families"][family]["status"] == "BOUND_INCONCLUSIVE"
        for family in applicable
    ):
        proof_status = "INCONCLUSIVE"
    else:
        proof_status = "FAIL"
    return receipt, errors, proof_status


def _same_float(left: object, right: object) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return a.shape == b.shape and np.allclose(
        a, b, atol=PAIR_ATOL, rtol=PAIR_RTOL, equal_nan=True
    )


def _maximum_float_difference(left: object, right: object) -> object:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return None
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0


def _report_scalar(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _compare_results(name: str, left: dict, right: dict) -> dict:
    differences = []
    maximum_float_differences = {}
    for field in FLOAT_FIELDS:
        maximum_float_differences[field] = _maximum_float_difference(
            left[field], right[field]
        )
        if not _same_float(left[field], right[field]):
            differences.append(field)
    for field in EXACT_FIELDS:
        if left[field] != right[field]:
            differences.append(field)
    return {
        "name": name,
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
        "maximum_float_differences": maximum_float_differences,
    }


def _cache_checks(results: Dict[str, dict]) -> list:
    names = (
        "cache_default_write",
        "cache_default_read",
        "cache_audit_read_default",
        "cache_audit_write",
        "cache_audit_read",
        "cache_audit_changed_y",
        "cache_audit_changed_weight",
    )
    if not any(name in results for name in names):
        return []
    missing = [name for name in names if name not in results]
    if missing:
        return [{"name": "cache_evidence_complete", "status": "FAIL", "differences": missing}]
    failed = [name for name in names if results[name].get("rc") != 0]
    if failed:
        return [{"name": "cache_fits_successful", "status": "FAIL", "differences": failed}]
    checks = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    default_write = results["cache_default_write"]
    default_read = results["cache_default_read"]
    audit_read_default = results["cache_audit_read_default"]
    audit_write = results["cache_audit_write"]
    audit_read = results["cache_audit_read"]
    changed_y = results["cache_audit_changed_y"]
    changed_weight = results["cache_audit_changed_weight"]
    default_sha = default_write.get("cache_sha256")
    audit_sha = audit_write.get("cache_sha256")
    add("default_cache_created", bool(default_sha), str(default_sha))
    add("audit_cache_created", bool(audit_sha), str(audit_sha))
    add(
        "default_and_audit_signatures_separated",
        bool(default_sha and audit_sha and default_sha != audit_sha),
        f"default={default_sha}, audit={audit_sha}",
    )
    add(
        "default_cache_read_only",
        default_read.get("cache_sha256") == default_sha == audit_read_default.get("cache_sha256"),
        "default write/read/audit-miss hashes",
    )
    add(
        "audit_cache_read_only",
        all(
            row.get("cache_sha256") == audit_sha
            for row in (audit_read, changed_y, changed_weight)
        ),
        "audit write/read/mutated-read hashes",
    )
    checks.append(_compare_results("default_cache_write_read_results", default_write, default_read))
    checks.append(_compare_results("audit_cache_write_read_results", audit_write, audit_read))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="new evidence directory")
    args = parser.parse_args()
    module = args.module.resolve()
    if not module.is_file():
        parser.error(f"module is not a file: {module}")
    output = args.out.resolve()
    if output.exists():
        parser.error(f"--out must be a new path: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cases_dir = output / "cases"
    cases_dir.mkdir()
    if args.backend == "cpu":
        (output / "caches").mkdir()

    source_dir = Path(__file__).resolve().parent
    worker = source_dir / "n05_audit_worker.py"
    receipt_source = source_dir / "n05_receipt.py"
    dependencies = (Path(__file__).resolve(), worker, receipt_source, source_dir / "cases.py", source_dir / "precision_contract.py")
    module_sha = _sha256(module)
    plan = _plan(module, module_sha, args.backend, output)
    manifest = {
        "schema": "xhdfe-n05-controls-v1",
        "module": str(module),
        "module_sha256": module_sha,
        "backend": args.backend,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "planned_calls": len(plan),
        "cache_scope": "CPU only; explicit CUDA requests intentionally bypass persistent absorption cache",
        "comparison": {
            "floating": "componentwise atol=rtol=64*binary64 epsilon",
            "atol": PAIR_ATOL,
            "rtol": PAIR_RTOL,
            "exact": "iterations, method, backend, omission, sample and input bytes",
        },
        "sources": {str(path): _sha256(path) for path in dependencies},
        "jobs": [entry["job"] for entry in plan],
    }
    _write_new(output / "manifest.json", manifest)

    parser_process = subprocess.run(
        [sys.executable, "-B", str(receipt_source)],
        text=True,
        capture_output=True,
        env=_clean_environment(),
        check=False,
    )
    (output / "parser_self_test.stdout.log").write_text(parser_process.stdout)
    (output / "parser_self_test.stderr.log").write_text(parser_process.stderr)
    parser_ok = parser_process.returncode == 0

    rows = []
    results = {}
    receipts = {}
    stderrs = {}
    for entry in plan:
        job = entry["job"]
        functional_errors = []
        audit_errors = []
        controller_errors = []
        result = {}
        process = {"returncode": -1, "stderr": ""}
        receipt = None
        proof_status = "FAIL" if job["certify"] == "1" else "NOT_REQUESTED"
        try:
            result, process = _run_worker(worker, cases_dir / job["name"], job)
            functional_errors.extend(_validate_functional(result, process, args.backend))
            receipt, receipt_errors, proof_status = _validate_receipt(
                result, process, job, entry["expected_cache_hit"]
            )
            audit_errors.extend(receipt_errors)
            results[job["name"]] = result
            receipts[job["name"]] = receipt
            stderrs[job["name"]] = process["stderr"]
        except Exception as error:
            controller_errors.append(f"controller error: {type(error).__name__}: {error}")
        errors = functional_errors + audit_errors + controller_errors
        rows.append(
            {
                "name": job["name"],
                "fixture": job["fixture"],
                "certify": job["certify"],
                "functional_status": "PASS" if not functional_errors and not controller_errors else "FAIL",
                "receipt_contract_status": "PASS" if not audit_errors and not controller_errors else "FAIL",
                "proof_status": proof_status,
                "proof_families_passed": (
                    [name for name, family in receipt["families"].items() if family["status"] == "PASS"]
                    if receipt is not None
                    else []
                ),
                "errors": errors,
                "metrics": (
                    {
                        "beta_error": _report_scalar(result.get("beta_error")),
                        "residual_error": _report_scalar(result.get("residual_error")),
                        "rss_error": _report_scalar(result.get("rss_error")),
                        "iterations": result.get("iterations"),
                        "gpu_used": result.get("gpu_used"),
                        "n05_work_count": result.get("n05_work_count"),
                    }
                    if result
                    else {}
                ),
                "receipt": receipt,
                "evidence": f"cases/{job['name']}",
            }
        )

    pair_checks = []
    for fixture, _mode in PAIR_FIXTURES:
        off = results.get(f"{fixture}__off")
        audit = results.get(f"{fixture}__audit")
        if off is None or audit is None or off.get("rc") != 0 or audit.get("rc") != 0:
            pair_checks.append({"name": fixture, "status": "FAIL", "differences": ["missing successful pair"]})
        else:
            pair_checks.append(_compare_results(fixture, off, audit))
    canonical_off = results.get("ordinary_homo_constant__off")
    canonical_audit = results.get("ordinary_homo_constant__audit")
    for name, reference in (
        ("selector_zero", canonical_off),
        ("legacy_flags_without_selector", canonical_off),
        ("legacy_flags_with_audit", canonical_audit),
    ):
        candidate = results.get(name)
        if reference is None or candidate is None or reference.get("rc") != 0 or candidate.get("rc") != 0:
            pair_checks.append({"name": name, "status": "FAIL", "differences": ["missing successful comparison"]})
        else:
            pair_checks.append(_compare_results(name, reference, candidate))
    canonical_receipt = receipts.get("ordinary_homo_constant__audit")
    legacy_receipt = receipts.get("legacy_flags_with_audit")
    pair_checks.append(
        {
            "name": "legacy_flags_audit_receipt",
            "status": "PASS" if canonical_receipt == legacy_receipt and canonical_receipt is not None else "FAIL",
            "differences": [] if canonical_receipt == legacy_receipt and canonical_receipt is not None else ["receipt"],
        }
    )
    for name, reference in (
        ("selector_zero_stderr", "selector_zero"),
        ("legacy_flags_without_selector_stderr", "legacy_flags_without_selector"),
    ):
        same = reference in stderrs and stderrs.get(reference) == stderrs.get(
            "ordinary_homo_constant__off"
        )
        pair_checks.append(
            {
                "name": name,
                "status": "PASS" if same else "FAIL",
                "differences": [] if same else ["stderr"],
            }
        )
    same_audit_stderr = stderrs.get("legacy_flags_with_audit") == stderrs.get(
        "ordinary_homo_constant__audit"
    ) and "legacy_flags_with_audit" in stderrs
    pair_checks.append(
        {
            "name": "legacy_flags_with_audit_stderr",
            "status": "PASS" if same_audit_stderr else "FAIL",
            "differences": [] if same_audit_stderr else ["stderr"],
        }
    )
    cache_checks = _cache_checks(results)

    failed_functional = sum(row["functional_status"] != "PASS" for row in rows)
    failed_rows = sum(row["receipt_contract_status"] != "PASS" for row in rows)
    failed_pairs = sum(row["status"] != "PASS" for row in pair_checks)
    failed_cache = sum(row["status"] != "PASS" for row in cache_checks)
    report = {
        "schema": "xhdfe-n05-controls-report-v1",
        "module": str(module),
        "module_sha256": module_sha,
        "backend": args.backend,
        "verdict": "PASS" if parser_ok and not (failed_functional or failed_rows or failed_pairs or failed_cache) else "FAIL",
        "counts": {
            "calls": len(rows),
            "functional_pass": sum(row["functional_status"] == "PASS" for row in rows),
            "receipt_contract_pass": len(rows) - failed_rows,
            "proof_pass": sum(row["proof_status"] == "PASS" for row in rows),
            "proof_inconclusive": sum(row["proof_status"] == "INCONCLUSIVE" for row in rows),
            "proof_fail": sum(row["proof_status"] == "FAIL" for row in rows),
            "proof_not_applicable": sum(row["proof_status"] == "NOT_APPLICABLE" for row in rows),
            "proof_not_requested": sum(row["proof_status"] == "NOT_REQUESTED" for row in rows),
            "pair_pass": len(pair_checks) - failed_pairs,
            "cache_pass": len(cache_checks) - failed_cache,
        },
        "parser_self_test": {
            "status": "PASS" if parser_ok else "FAIL",
            "returncode": parser_process.returncode,
            "stdout": parser_process.stdout.strip(),
            "stderr": parser_process.stderr.strip(),
        },
        "rows": rows,
        "pair_checks": pair_checks,
        "cache_checks": cache_checks,
        "interpretation": {
            "functional_pass": "analytic estimates and immutable outputs passed independently of audit proof status",
            "proof_pass": "only a literal PASS for every predeclared supported family counts",
            "unsupported_or_bound": "never counted as N05 proof PASS",
            "gate_liveness": "disabled zero-work and enabled positive-work claims use the private ELF counter, not stderr silence",
        },
    }
    _write_new(output / "report.json", report)
    print(json.dumps({"verdict": report["verdict"], "backend": args.backend, **report["counts"]}, sort_keys=True))
    return int(report["verdict"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
