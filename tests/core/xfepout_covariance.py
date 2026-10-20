"""Grade numerical partial-out returns separately from frozen safe refusals."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grade_log(text, backend):
    text = re.sub(r"\n>\s*", "", text)
    lines = text.splitlines()
    assert "DRIVER_RC 0" in lines
    assert "XFEPOUT_THREE_FE_CONTRACT_COMPLETE" in lines
    number = r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"
    successful = re.findall(
        r"^XFEPOUT_THREE_FE (\w+) (\d+) (\w+) b=\s*" + number
        + r" V=\s*" + number + r"$", text, re.M)
    refused = re.findall(
        r"^XFEPOUT_THREE_FE_EXPECTED_REFUSAL (\w+) (\d+) (\w+)$", text, re.M)
    diagnostics = re.findall(
        r"^HDFE partial-out: precision certification failed; iterations=(\d+)/(\d+); "
        r"requested tolerance=" + number + r"; relative residual=" + number, text, re.M)
    rows = []
    for device, threads, variant, b, v in successful:
        b, v = float(b), float(v)
        assert device == backend and math.isfinite(b) and math.isfinite(v)
        assert b <= 1e-9 and v <= 1e-8
        rows.append(dict(backend=device, threads=int(threads), variant=variant,
                         status="PASS", b_scaled=b, V_scaled=v))
    assert len(diagnostics) == len(refused)
    for (device, threads, variant), diagnostic in zip(refused, diagnostics):
        assert device == backend == "cuda" and int(threads) == 16
        assert variant in ("original", "balanced", "reordered")
        iterations, maximum = map(int, diagnostic[:2])
        tolerance, residual = map(float, diagnostic[2:])
        assert 0 < iterations <= maximum == 100000
        assert tolerance == 1e-8 and math.isfinite(residual)
        assert residual > 8 * min(tolerance, 1e-9)
        rows.append(dict(backend=device, threads=int(threads), variant=variant,
                         status="EXPECTED_REFUSAL", iterations=iterations,
                         max_iter=maximum, relative_residual=residual,
                         reason="Necessary absorption check failed after continuation; no estimates."))
    expected = {(t, v) for t in (1, 16)
                for v in ("original", "balanced", "reordered")}
    assert len(rows) == 6 and {(r["threads"], r["variant"]) for r in rows} == expected
    # The do-file catches expected refusals; an uncaught Stata error is a failure.
    assert not re.search(r"^r\(\d+\);", text, re.M)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    fixture_root = args.fixture_root.resolve(strict=True)
    test = args.test.resolve(strict=True)
    data = fixture_root / "three_fe_synthetic.dta"
    oracle = fixture_root / "three_fe_oracle.do"
    provenance = json.loads((fixture_root / "THREE_FE_FIXTURE.json").read_text())
    assert provenance["synthetic"] is True
    assert all(sha(Path(k)) == v for k, v in provenance["files"].items())
    artifacts = [package / n for n in
                 ("xfepout.ado", "xfepout.plugin", "xhdfe.ado", "xhdfe.plugin")]
    frozen = {str(p): sha(p) for p in [*artifacts, test, data, oracle]}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("XHDFE_") and k != "LD_LIBRARY_PATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_CACHE_DISABLE="1",
               OPENBLAS_NUM_THREADS="1")
    groups, rows = [], []
    for backend in ("cpu", "cuda"):
        work = output / backend
        work.mkdir()
        tmp = work / "tmp"
        tmp.mkdir()
        with (work / "driver.do").open("x") as stream:
            stream.write("\n".join([
                f'capture noisily do "{test}" "{package}" "{backend}" "{data}" "{oracle}"',
                "local result = _rc", 'display "DRIVER_RC `result\'"',
                "exit `result'", "",
            ]))
        with (work / "process.log").open("x") as log:
            run = subprocess.run(
                ["/usr/local/stata/stata-mp", "-q", "-b", "do", "driver.do"],
                env=dict(env, STATATMP=str(tmp)), cwd=work,
                stdout=log, stderr=subprocess.STDOUT, timeout=420)
        group = dict(backend=backend, returncode=run.returncode, status="FAIL")
        try:
            assert run.returncode == 0
            outcomes = grade_log((work / "driver.log").read_text(), backend)
            group.update(status="PASS_CONTRACT", outcomes=outcomes)
            rows.extend(outcomes)
        except (AssertionError, ValueError) as error:
            group["reason"] = str(error) or "Log or numerical contract failed"
        groups.append(group)
        print(backend, group["status"], flush=True)
    assert all(sha(Path(k)) == v for k, v in frozen.items())
    passed = len(rows) == 12 and all(g["status"] == "PASS_CONTRACT" for g in groups)
    result = dict(status="PASS_CONTRACT" if passed else "FAIL", groups=groups,
                  numerical_pass_cases=sum(r["status"] == "PASS" for r in rows),
                  expected_refusals=sum(r["status"] == "EXPECTED_REFUSAL" for r in rows),
                  refusal_is_not_numerical_success=True, frozen=frozen)
    with (output / "RESULT.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
