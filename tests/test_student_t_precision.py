#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess


LIMIT = 5e-12


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--high-precision-provenance", required=True)
    args = parser.parse_args()

    with Path(args.fixtures).open(newline="") as handle:
        fixtures = list(csv.DictReader(handle))
    provenance_path = Path(args.provenance).resolve(strict=True)
    provenance = json.loads(provenance_path.read_text())
    repo = provenance_path.parents[2]
    if provenance.get("schema_version") != 1:
        raise AssertionError("Student-t provenance schema differs")
    records = (
        (provenance["reproduction"]["do_file"],
         provenance["reproduction"]["do_sha256"]),
        (provenance["reproduction"]["log_file"],
         provenance["reproduction"]["log_sha256"]),
        (provenance["fixture_table"]["path"],
         provenance["fixture_table"]["sha256"]),
    )
    for relative, expected_hash in records:
        path = repo / relative
        if sha256(path) != expected_hash:
            raise AssertionError(f"Student-t provenance hash differs: {relative}")
    stata_executable = Path(provenance["stata"]["executable"])
    if stata_executable.is_file() and sha256(stata_executable) != provenance[
        "stata"
    ]["executable_sha256"]:
        raise AssertionError("Student-t Stata executable hash differs")

    high_path = Path(args.high_precision_provenance).resolve(strict=True)
    high = json.loads(high_path.read_text())
    if high.get("schema_version") != 1:
        raise AssertionError("Student-t high-precision provenance schema differs")
    for record_name in (
        "generator", "fixture", "subnormal_generator", "subnormal_fixture",
    ):
        record = high[record_name]
        if sha256(repo / record["path"]) != record["sha256"]:
            raise AssertionError(
                f"Student-t high-precision hash differs: {record_name}"
            )
    host_records = (
        (high["arithmetic"]["boost_version_header"],
         high["arithmetic"]["boost_version_header_sha256"]),
        (high["arithmetic"]["students_t_header"],
         high["arithmetic"]["students_t_header_sha256"]),
        (high["compiler"]["path"], high["compiler"]["sha256"]),
    )
    for raw_path, expected_hash in host_records:
        path = Path(raw_path)
        if path.is_file() and sha256(path) != expected_hash:
            raise AssertionError(
                f"Student-t high-precision host hash differs: {path}"
            )
    completed = subprocess.run(
        [args.executable, "--emit"], text=True, capture_output=True, check=True,
    )
    observed = list(csv.DictReader(io.StringIO(completed.stdout)))
    if [row["name"] for row in observed] != [row["name"] for row in fixtures]:
        raise AssertionError("Student-t fixture inventory/order differs")

    worst_p = 0.0
    worst_critical_scaled = 0.0
    for reference, candidate in zip(fixtures, observed):
        expected_critical = float(reference["critical"])
        for field in ("working_critical", "binary64_critical"):
            value = float(candidate[field])
            error = abs(value - expected_critical) / max(1.0, abs(expected_critical))
            worst_critical_scaled = max(worst_critical_scaled, error)
            if not math.isfinite(value) or error > LIMIT:
                raise AssertionError(
                    f"{reference['name']} {field} differs: {value} vs "
                    f"{expected_critical} ({error})"
                )
        if reference["p"]:
            expected_p = float(reference["p"])
            for field in ("working_p", "binary64_p"):
                value = float(candidate[field])
                error = abs(value - expected_p)
                worst_p = max(worst_p, error)
                if not math.isfinite(value) or error > LIMIT:
                    raise AssertionError(
                        f"{reference['name']} {field} differs: {value} vs "
                        f"{expected_p} ({error})"
                    )
    print(
        "STUDENT_T_PYTHON_PASS "
        f"fixtures={len(fixtures)} p_abs_max={worst_p:.17g} "
        f"critical_scaled_max={worst_critical_scaled:.17g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
