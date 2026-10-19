"""Exercise packaged plugins inside Stata, retaining crashes and actual workers.

Unlike the native SPI release gate, this supplementary host check can run an
Intel Stata slice through Rosetta. Its receipt explicitly records that limit;
translated execution must never substitute for native release certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stata", type=Path, required=True)
    parser.add_argument("--adopath", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=["arm64", "x86_64"])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 0])
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if min(args.threads) < 0 or args.rows < 1000 or args.repeats < 1:
        parser.error("threads must be nonnegative; rows >= 1000; repeats >= 1")
    if len(set(args.threads)) != len(args.threads):
        parser.error("thread requests must be unique")
    if args.arch and platform.system() != "Darwin":
        parser.error("--arch is only supported for macOS Stata")
    package = args.adopath.resolve(strict=True)
    stata = args.stata.resolve(strict=True)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve().parents[1] / "tests/stata/openmp_host_contract.do"
    files = ["xhdfe.plugin", "xfepout.plugin", "xhdfe.ado", "xfepout.ado"]
    if (package / "xhdfe_libomp.dylib").exists():
        files.append("xhdfe_libomp.dylib")
    host = platform.machine()
    hardware_arch = host
    if platform.system() == "Darwin":
        # A translated Python controller can itself report x86_64. Detect the
        # hardware so that its Intel Stata child is still labelled Rosetta.
        arm64 = subprocess.check_output(
            ["/usr/sbin/sysctl", "-in", "hw.optional.arm64"], text=True).strip()
        hardware_arch = "arm64" if arm64 == "1" else "x86_64"
    receipt = dict(status="FAIL", host_arch=host, requested_arch=args.arch or host,
                   hardware_arch=hardware_arch,
                   translated=hardware_arch == "arm64" and (args.arch or host) == "x86_64",
                   stata_path=str(stata), package_path=str(package),
                   runner_sha256=sha(Path(__file__)),
                   script_sha256=sha(script), hashes={n: sha(package/n) for n in files},
                   cases=[], scope="Real Stata host checks; no native release certification claim")
    env = dict(os.environ, XHDFE_STATA_ADOPATH=str(package))
    # Do not inherit an experimental runtime injection or thread override.
    for name in ("DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                 "DYLD_FORCE_FLAT_NAMESPACE", "LD_PRELOAD", "OMP_NUM_THREADS",
                 "KMP_DUPLICATE_LIB_OK"):
        env.pop(name, None)
    for threads in args.threads:
        work = output / f"threads-{threads}"
        work.mkdir()
        rows = max(args.rows, 2000000) if threads == 0 else args.rows
        command = (["/usr/bin/arch", "-"+args.arch] if args.arch else []) + [
            str(stata), "-b", "do", str(script), str(threads), str(rows), str(args.repeats)]
        record = dict(threads=threads, rows=rows, command=command, status="FAIL")
        try:
            result = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                    text=True, timeout=args.timeout, check=False)
            (work / "process.log").write_text(result.stdout+result.stderr)
            log_file = work / "openmp_host_contract.log"
            log = log_file.read_text(errors="replace") if log_file.exists() else ""
            marker = f"OPENMP_HOST_CONTRACT_PASS threads={threads} rows={rows} repeats={args.repeats}"
            cases = [line for line in log.splitlines() if line.startswith("OPENMP_HOST_CASE_PASS ")]
            passed = result.returncode == 0 and marker in log.splitlines() and len(cases) == 5*args.repeats
            record.update(returncode=result.returncode, status="PASS" if passed else "FAIL",
                          completed_cases=len(cases), markers=cases)
        except subprocess.TimeoutExpired:
            record["reason"] = "timeout"
        receipt["cases"].append(record)
        (output / "RESULT.json").write_text(json.dumps(receipt, indent=2)+"\n")
        print(json.dumps({k: record[k] for k in ("threads", "rows", "status", "completed_cases") if k in record}), flush=True)
    receipt["status"] = "PASS" if all(r["status"] == "PASS" for r in receipt["cases"]) else "FAIL"
    receipt["artifacts_unchanged"] = all(sha(package/n) == h for n, h in receipt["hashes"].items())
    if not receipt["artifacts_unchanged"]:
        receipt["status"] = "FAIL"
    (output / "RESULT.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
