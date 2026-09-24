#!/usr/bin/env python3
"""Add T01 mode adjudication to immutable completed campaign raws."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluate_fail_closed import assess as historical_assess
from evaluate_t01 import assess_t01
from run import clean_json, write_json


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("campaign", type=Path)
args = parser.parse_args()
campaign = args.campaign.resolve()
manifest = json.loads((campaign / "manifest.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


changed = [path for path, expected in manifest["files"].items()
           if not Path(path).is_file() or sha(path) != expected]
if changed:
    raise RuntimeError(f"custody mismatch: {changed}")
rows = []
for job in manifest["jobs"]:
    attempt = campaign / "attempts" / job["id"]
    raw_path = attempt / "raw.json"
    if not raw_path.is_file():
        rows.append(dict(job, t01=dict(applicable=True,
                                      g1_verdict="COVERAGE_MISSING",
                                      passed=False,
                                      reason="raw result absent")))
        continue
    raw = json.loads(raw_path.read_text())
    fixture = campaign / "fixtures" / job["case"]
    historical = historical_assess(job, raw, fixture)
    rows.append(dict(job, historical_verdict=historical["verdict"],
                     t01=assess_t01(job, raw, fixture, historical)))
counts = Counter(row["t01"]["g1_verdict"] for row in rows
                 if row["t01"].get("applicable"))
report = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
              campaign=str(campaign), campaign_manifest_sha256=sha(campaign / "manifest.json"),
              evaluator_sha256=sha(Path(__file__).with_name("evaluate_t01.py")),
              contract_sha256=sha(Path(__file__).with_name("precision_contract.py")),
              counts=dict(counts), passed=bool(counts) and set(counts) <= {"PASS", "UNSUPPORTED_EXPECTED"},
              joint_fit_metric_is_gate=False, rows=rows)
target = campaign / f't01_regrade_{len(list(campaign.glob("t01_regrade_*.json"))) + 1:03d}.json'
write_json(target, clean_json(report))
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
print("T01_REPORT", target)
raise SystemExit(0 if report["passed"] else 1)
