"""Check fit statistics, wide-ID ordering and sample writes on Stata/MP."""

import argparse
import os
from pathlib import Path
import re
import subprocess


def fields(log, label):
    return [
        dict(field.split("=", 1) for field in line.split()[2:])
        for line in log.splitlines()
        if line.startswith(f"plugin_profile label={label} ")
    ]


def check(stata, package, output, limit):
    directory = output / f"team{limit}"
    directory.mkdir()
    script = Path(__file__).with_suffix(".do").resolve()
    env = os.environ | {
        "OMP_NUM_THREADS": "4",
        "OMP_DYNAMIC": "FALSE",
        "OMP_THREAD_LIMIT": str(limit),
        "TMPDIR": str(directory),
        "XHDFE_FUSED_READ": "1",
        "XHDFE_PARALLEL_READ": "1",
        "XHDFE_PROFILE_CPU": "1",
        "XHDFE_WIDE_ID_READ": "1",
    }
    with (directory / "stata.stdout").open("xb") as out, (
        directory / "stata.stderr"
    ).open("xb") as err:
        subprocess.run(
            [stata, "-q"],
            input=(f'do "{script}" "{package}" 4 {limit}\nexit, clear\n').encode(),
            cwd=directory,
            env=env,
            stdout=out,
            stderr=err,
            check=True,
            timeout=420,
        )
    log = (directory / "stata.stdout").read_text(errors="replace")
    if not re.search(r"^POSTFIT_PARALLEL_CONTRACT_PASS\s*$", log, re.MULTILINE):
        raise ValueError(f"Missing success marker in {directory}")
    profile = (directory / "stata.stderr").read_text(errors="replace")
    wide = [row for row in fields(profile, "wide_id_read") if row["what"] == "clusters"]
    if len(wide) != 2:
        raise ValueError(f"Expected two wide-cluster reads in {directory}, got {wide}")
    for row in wide:
        if row["path"] != "team" or int(row["order_team"]) != limit:
            raise ValueError(f"Wrong observed team for wide identifiers: {row}")
        if not 1 <= int(row["order_workers"]) <= limit:
            raise ValueError(f"Wrong number of ordering workers: {row}")
    if int(wide[0]["order_workers"]) != limit or int(wide[0]["sort_team"]) != 0:
        raise ValueError(f"Sorted input did not take the expected path: {wide[0]}")
    writebacks = fields(profile, "writeback_work")
    if writebacks != [{"team": str(limit), "workers": str(limit)}] * 3:
        raise ValueError(f"Expected three observed sample writes with {limit} workers: {writebacks}")
    print(f"POSTFIT_PARALLEL_CONTRACT_PASS requested=4 observed={limit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Directory with Stata ado and plugin")
    parser.add_argument("output", type=Path, help="New directory for the run receipts")
    parser.add_argument("--stata", default="stata-mp", help="Stata MP executable")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    for limit in (4, 2):
        check(args.stata, args.package.resolve(), output, limit)


if __name__ == "__main__":
    main()
