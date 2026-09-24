#!/usr/bin/env python3
"""R Makevars-before-Makeconf control; this is not a native Windows test."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--makevars", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which("g++")
    assert compiler, "GNU C++ is required for this local control"
    records = []
    for label, flags, allow, release, expected in (
        ("late_openmp", "-fopenmp", "", "", True),
        ("late_openmp_override", "-fopenmp", "1", "", True),
        ("missing_openmp", "", "", "", False),
        ("invalid_openmp", "-fno-openmp", "", "", False),
        ("release_serial", "-fopenmp", "1", "1", False),
    ):
        directory = args.scratch / label
        directory.mkdir()
        shutil.copy2(args.makevars, directory / "Makevars.win")
        (directory / "Makeconf").write_text(
            f"CXX17 = {compiler}\nCXX17STD = -std=gnu++17\n"
            f"SHLIB_OPENMP_CXXFLAGS = {flags}\n")
        (directory / "probe.cpp").write_text(
            '#include <omp.h>\n#include <cstdio>\n'
            '#ifndef HDFE_USE_OPENMP\n#error missing xhdfe OpenMP definition\n#endif\n'
            'int main() { int seen=0; omp_set_dynamic(0);\n'
            '#pragma omp parallel num_threads(2) reduction(+:seen)\n'
            '{ seen += 1; } std::printf("%d\\n",seen); return seen==2 ? 0 : 1; }\n')
        (directory / "rules.mk").write_text(
            'probe.o: probe.cpp\n'
            '\t$(CXX17) $(CXX17STD) $(PKG_CPPFLAGS) $(PKG_CXXFLAGS) -c $< -o $@\n'
            'probe: probe.o\n\t$(CXX17) $< $(PKG_LIBS) -o $@\n')
        cmd = ['make', '-f', 'Makevars.win', '-f', 'Makeconf', '-f', 'rules.mk',
               'all', 'SHLIB=probe', f'XHDFE_ALLOW_SERIAL_BUILD={allow}',
               f'XHDFE_RELEASE_BUILD={release}']
        run = subprocess.run(cmd, cwd=directory, capture_output=True, text=True, timeout=60)
        (directory / "build.log").write_text(run.stdout + run.stderr)
        row = dict(case=label, returncode=run.returncode, expected_success=expected)
        if run.returncode == 0:
            probe = subprocess.run([str((directory / "probe").resolve())],
                                   capture_output=True, text=True, timeout=30)
            row.update(probe_returncode=probe.returncode, observed_workers=probe.stdout.strip())
            success = probe.returncode == 0 and probe.stdout.strip() == "2"
        else:
            success = False
        row["status"] = "PASS" if success == expected else "FAIL"
        records.append(row)
    (args.scratch / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    assert all(row["status"] == "PASS" for row in records), records
    print("PASS: late flags, effective OpenMP, missing/broken runtime and release serial rejection")


if __name__ == "__main__":
    main()
