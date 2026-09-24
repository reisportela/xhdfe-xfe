#!/usr/bin/env python3
"""Compare FE scaling bits with strict libm across binary64 boundaries and modes."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, required=True)
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).with_suffix(".cpp")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    common = ["-std=c++17", "-I"+str(root/"include"), "-I"+str(root/"r/xhdfe/src/eigen")]
    commands = []
    for name, flags in [
        ("implementation", ["-O3", "-DNDEBUG", "-march=native", "-ffast-math", "-fno-finite-math-only", "-funroll-loops", "-fopenmp", "-DXHDFE_SCALE_IMPLEMENTATION"]),
        ("reference", ["-O2", "-fno-fast-math", "-frounding-math", "-ffp-contract=off", "-fno-builtin-ldexp", "-DXHDFE_SCALE_REFERENCE"]),
    ]:
        commands.append(compiler+common+flags+["-c", str(source), "-o", str(a.scratch/(name+".o"))])
    commands.append(compiler+common+["-O2", "-fno-fast-math", "-frounding-math", "-fopenmp", str(source),
        str(a.scratch/"implementation.o"), str(a.scratch/"reference.o"), "-o", str(a.scratch/"check")])
    (a.scratch/"commands.json").write_text(json.dumps(commands, indent=2)+"\n")
    for i, command in enumerate(commands):
        with (a.scratch/("build_"+str(i)+".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=420)
    result = subprocess.run([str(a.scratch/"check")], capture_output=True, text=True, timeout=420)
    (a.scratch/"check.log").write_text(result.stdout+result.stderr)
    result.check_returncode()
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
