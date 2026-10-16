"""Reject an OpenMP runtime that can expose private symbols to a Stata host.

Run on macOS for each final slice. The separate Mach-O dependency/receipt
validator remains portable and continues to require native execution evidence.
"""
import argparse
import fnmatch
from pathlib import Path
import subprocess


def check(runtime: Path, arch: str) -> None:
    patterns = [line.strip() for line in
                Path(__file__).with_name("macos_openmp.exports").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    # -g: external; -U: defined only; -j: just the symbol names.
    output = subprocess.check_output(
        ["/usr/bin/nm", "-arch", arch, "-gUj", str(runtime)], text=True)
    symbols = set(output.splitlines())
    required = {"_omp_get_max_threads", "_omp_set_num_threads", "___kmpc_fork_call"}
    missing = required - symbols
    unexpected = sorted(name for name in symbols
                        if not any(fnmatch.fnmatchcase(name, p) for p in patterns))
    if missing or unexpected:
        raise SystemExit(f"OpenMP export isolation failed ({arch}): "
                         f"missing={sorted(missing)}; private exports={unexpected}")
    print(f"OpenMP export isolation OK: {arch}, {len(symbols)} public ABI symbols")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--arch", choices=["arm64", "x86_64"], required=True)
    args = parser.parse_args()
    check(args.runtime, args.arch)
