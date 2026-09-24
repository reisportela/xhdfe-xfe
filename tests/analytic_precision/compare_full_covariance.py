import argparse
import csv
import json
import math
import struct
from pathlib import Path


def identical_binary64(left, right):
    return len(left) == len(right) and all(
        struct.pack("!d", float(a)) == struct.pack("!d", float(b))
        for a, b in zip(left, right))


def read(folder):
    cpp = folder / "full_result.json"
    if cpp.is_file():
        return json.loads(cpp.read_text())
    bp, vp = folder / "result_full_b.csv", folder / "result_full_V.csv"
    if not (bp.is_file() and vp.is_file()):
        return None
    bs = list(csv.DictReader(bp.open()))
    b = [float(row["estimate"]) for row in bs]
    v = [[0.] * len(b) for _ in b]
    entries = list(csv.DictReader(vp.open()))
    if len(entries) != len(b)*len(b):
        raise ValueError("full covariance CSV is incomplete")
    seen = set()
    for row in entries:
        i, j = int(row["row"])-1, int(row["column"])-1
        if not (0 <= i < len(b) and 0 <= j < len(b)) or (i,j) in seen:
            raise ValueError("full covariance CSV has invalid or duplicate entries")
        seen.add((i,j))
        v[i][j] = float(row["value"])
    return dict(b=b, V=v, labels=[row["term"] for row in bs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for pair in sorted((args.root / "raw").glob("*/pair_*")):
        folders = {role: next(iter(pair.glob("*"+role)), None) for role in ("baseline", "candidate")}
        row = dict(dataset=pair.parent.name, pair=pair.name)
        try:
            models = {role: read(folder) if folder else None for role, folder in folders.items()}
        except (ValueError, KeyError, IndexError, TypeError) as error:
            row.update(status="REVIEW", reason=f"malformed output: {error}")
            rows.append(row)
            continue
        if any(m is None for m in models.values()):
            row.update(status="NO_COMPARISON", reason="one or both fits have no estimates")
            rows.append(row)
            continue
        a, b = models["baseline"], models["candidate"]
        if any(len(m["V"]) != len(m["b"]) or
               any(len(values) != len(m["b"]) for values in m["V"])
               for m in (a,b)):
            row.update(status="REVIEW", reason="covariance dimensions do not match coefficients")
            rows.append(row)
            continue
        if a["labels"] != b["labels"] or len(a["b"]) != len(b["b"]):
            row.update(status="REVIEW", reason="coefficient stripe differs")
            rows.append(row)
            continue
        finite_b = all(math.isfinite(v) for v in a["b"] + b["b"])
        be = (max((abs(x-y)/max(1., abs(x)) for x, y in zip(a["b"], b["b"])), default=0.)
              if finite_b else math.inf)
        ve, zeros, finite = 0., True, finite_b
        for i, values in enumerate(a["V"]):
            for j, value in enumerate(values):
                other = b["V"][i][j]
                if not all(math.isfinite(x) for x in (value, other, a["V"][i][i], a["V"][j][j])):
                    finite = False
                    continue
                scale = math.sqrt(max(0., a["V"][i][i])) * math.sqrt(max(0., a["V"][j][j]))
                if scale:
                    ve = max(ve, abs(value-other)/scale)
                else:
                    zeros &= value == other
        same_sample = a.get("sample_sha256") == b.get("sample_sha256") if "sample_sha256" in a else None
        passed = be <= 1e-9 and ve <= 1e-8 and zeros and finite and same_sample is not False
        row.update(b_error=be, V_error=ve, zero_entries_equal=zeros, finite=finite,
                   sample_sha_equal=same_sample,
                   b_bit_identical=finite and identical_binary64(a["b"], b["b"]),
                   V_bit_identical=finite and all(identical_binary64(x, y)
                       for x, y in zip(a["V"], b["V"])),
                   status="PASS" if passed else "REVIEW")
        rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    print({status: sum(r["status"] == status for r in rows)
           for status in ("PASS", "REVIEW", "NO_COMPARISON")})


if __name__ == "__main__":
    main()
