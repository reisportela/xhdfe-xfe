#!/usr/bin/env python3
"""Negative controls for complete covariance and literal binary64 comparisons."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    fixtures = {
        "nan": ([1., 2.], [1., float("nan")]),
        "signed_zero": ([0., 2.], [-0., 2.]),
        "one_ulp": ([1., 2.], [math.nextafter(1., 2.), 2.]),
        "identical": ([1., 2.], [1., 2.]),
        "short_json_matrix": ([1., 2.], [1., 2.]),
        "incomplete_csv": ([1., 2.], [1., 2.]),
        "duplicate_csv": ([1., 2.], [1., 2.]),
    }
    for name, coefficients in fixtures.items():
        for role, values in zip(("baseline", "candidate"), coefficients):
            folder = args.scratch / "raw" / name / "pair_001" / ("order_1_"+role)
            folder.mkdir(parents=True)
            if name.endswith("csv"):
                (folder / "result_full_b.csv").write_text("term,estimate\na,1\nb,2\n")
                entries = "row,column,value\n1,1,1\n1,2,0\n2,1,0\n"
                if name == "duplicate_csv":
                    entries += "1,1,1\n"
                (folder / "result_full_V.csv").write_text(entries)
            else:
                covariance = [[1.]] if name == "short_json_matrix" else [[1., 0.], [0., 1.]]
                (folder / "full_result.json").write_text(json.dumps(dict(
                    b=values, V=covariance, labels=["a", "b"], sample_sha256="same_sample"))+"\n")
    output = args.scratch / "results.json"
    subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("compare_full_covariance.py")),
                    str(args.scratch), "--output", str(output)], check=True)
    rows = {r["dataset"]: r for r in json.loads(output.read_text())}
    for name in ("nan", "short_json_matrix", "incomplete_csv", "duplicate_csv"):
        assert rows[name]["status"] == "REVIEW", name
    assert not rows["nan"]["finite"] and not rows["nan"]["b_bit_identical"]
    for name in ("signed_zero", "one_ulp", "identical"):
        assert rows[name]["status"] == "PASS", name
        assert rows[name]["b_bit_identical"] == (name == "identical"), name
    print("PASS: four malformed/nonfinite cases rejected and three valid controls retained")


if __name__ == "__main__":
    main()
