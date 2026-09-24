#!/usr/bin/env python3
"""Aggregate complete default/audit T01 reports without promoting a subset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from run import write_json


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected_mode):
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(ROOT):
        raise RuntimeError(f"report must be an existing file inside the selective tree: {path}")
    value = json.loads(path.read_text())
    if value.get("certificate_mode") != expected_mode:
        raise RuntimeError(f"certificate mode mismatch in {path}")
    return path, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ("default", "audit"):
        parser.add_argument(f"--main-{mode}", type=Path, required=True)
        parser.add_argument(f"--r-{mode}", type=Path, required=True)
        parser.add_argument(f"--supplemental-{mode}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("output must be a new directory inside the selective tree")

    reports = {}
    for mode in ("default", "audit"):
        for component in ("main", "r", "supplemental"):
            path, value = load(getattr(args, f"{component}_{mode}"), mode)
            reports[f"{component}_{mode}"] = dict(path=str(path), sha256=sha(path), report=value)

    default_main = reports["main_default"]["report"]
    default_r = reports["r_default"]["report"]
    default_supplemental = reports["supplemental_default"]["report"]
    default_pass = bool(default_main.get("xhdfe_scoped_certified")) and bool(default_main.get("t01_scoped_pass"))
    default_pass = default_pass and bool(default_r.get("scoped_R_pass"))
    default_pass = default_pass and bool(default_supplemental.get("scoped_supplemental_pass"))

    audit_incomplete = False
    audit_real_failures = 0
    audit_verifier_findings = 0
    audit_unclassified_failures = 0
    for component in ("main", "r"):
        value = reports[f"{component}_audit"]["report"]
        for row in value.get("rows", []):
            if component == "main" and row.get("engine") != "xhdfe":
                continue
            historical = row.get("verdict")
            t01 = row.get("t01", {})
            classification = t01.get("g1_verdict") if t01.get("applicable") else None
            if classification == "REAL_ESTIMATOR_FAILURE":
                audit_real_failures += 1
            elif classification == "VERIFIER_FAILURE":
                audit_verifier_findings += 1
            elif classification in ("COVERAGE_MISSING", "HARNESS_FAILURE"):
                audit_incomplete = True
            elif historical not in ("PASS", "UNSUPPORTED_EXPECTED"):
                audit_unclassified_failures += 1
                audit_incomplete = True
    for row in reports["supplemental_audit"]["report"].get("rows", []):
        classification = row.get("verdict")
        if classification == "REAL_ESTIMATOR_FAILURE":
            audit_real_failures += 1
        elif classification == "VERIFIER_FAILURE":
            audit_verifier_findings += 1
        elif classification in ("COVERAGE_MISSING", "HARNESS_FAILURE"):
            audit_incomplete = True

    pending = [
        "formal covariance-oracle interval where gamma_m*kappa(G)>=1",
        "IV/2SLS N1 normal defect on X-hat/instrument score design",
        "core24 x 8 performance and correctness matrix",
        "group/individual 3M performance gate",
    ]
    summary = dict(
        schema=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        default_scoped_functional_pass=default_pass,
        audit_observation_complete=not audit_incomplete,
        audit_real_estimator_failures=audit_real_failures,
        audit_verifier_findings=audit_verifier_findings,
        audit_unclassified_failures=audit_unclassified_failures,
        audit_verifier_findings_block_default=False,
        joint_fit_metric_is_gate=False,
        full_G1_claim=False,
        release_or_G2_claim=False,
        core24_x8_still_required=True,
        pending=pending,
        inputs={key: {"path": value["path"], "sha256": value["sha256"]}
                for key, value in reports.items()},
    )
    output.mkdir()
    write_json(output / "T01_AGGREGATE.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if default_pass and not audit_incomplete and audit_real_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
