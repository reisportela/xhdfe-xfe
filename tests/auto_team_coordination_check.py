#!/usr/bin/env python3
"""Functional contract for automatic teams shared between one user's fits.

Package A3 (src/auto_team_registry.cpp). Every case prepares a private
registry directory, never /dev/shm, holding records of live helper processes
of this user (``sleep``), runs one fit in a fresh process with
XHDFE_AUTO_TEAM_REGISTRY_DIR pointing at it, and checks the team, the limit
code and reason, and what the fit left in the registry against

    max(legacy, min(requested, capacity // sessions, capacity - leased)).

Synthetic data only. Elapsed time is never an acceptance criterion. Linux
only: the helper records need /proc start times (the C++ contract
tests/auto_team_registry_contract.cpp covers the registry on macOS).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import threads_contract_check as base  # noqa: E402  (design, load_module, densities)

ROWS = 1200000
UNCAPPED_ROWS_PER_THREAD = 500000
PROBLEM_CAP = 16
HEADER = struct.Struct("=8sIIQQ")
RECORD = struct.Struct("=iiQQII")
INTENT, LEASE, EXPLICIT_FLAG = 1, 2, 1
CODE = 6
REASON = "auto_team_coordination"
REGISTRY_ENV = "XHDFE_AUTO_TEAM_REGISTRY_DIR"
SWITCH_ENV = "XHDFE_AUTO_TEAM_COORDINATION"


class ContractFailure(RuntimeError):
    pass


def grant(requested: int, legacy: int, capacity: int, sessions: int, leased: int) -> int:
    requested = max(1, requested)
    legacy = max(1, legacy)
    if legacy >= requested:
        return requested
    share = max(1, capacity) // max(1, sessions)
    free = max(1, capacity) - max(0, leased)
    return max(legacy, min(requested, share, free))


def automatic_requested(capacity: int) -> int:
    absorb = base.absorb_rows_per_thread()
    return min(PROBLEM_CAP, max(1, ROWS // absorb), capacity)


def legacy_team(capacity: int) -> int:
    return min(PROBLEM_CAP, max(1, ROWS // UNCAPPED_ROWS_PER_THREAD), capacity)


def start_time(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    # Field 2 may hold blanks: count from the last ')'; field 22 is index 19.
    return int(text[text.rindex(")") + 1:].split()[19])


def registry_path(directory: Path) -> Path:
    return directory / f"xhdfe-teams-{os.getuid()}"


def registry_bytes(records: list[tuple]) -> bytes:
    payload = HEADER.pack(b"XHDFETR1", 1, RECORD.size, 1000, 0)
    for record in records:
        payload += RECORD.pack(*record)
    return payload


def read_records(directory: Path) -> list[tuple]:
    path = registry_path(directory)
    if not path.exists():
        return []
    data = path.read_bytes()
    records = []
    for offset in range(HEADER.size, len(data) - RECORD.size + 1, RECORD.size):
        record = RECORD.unpack_from(data, offset)
        if record[0] != 0:
            records.append(record)
    return records


def write_registry(directory: Path, records: list[tuple], mode: int = 0o600) -> bytes:
    path = registry_path(directory)
    payload = registry_bytes(records)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)
    return payload


class Helpers:
    """Live processes of this user standing in for other sessions."""

    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []

    def spawn(self, count: int) -> list[tuple[int, int]]:
        spawned = []
        for _ in range(count):
            process = subprocess.Popen(
                ["sleep", "3600"], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.processes.append(process)
            spawned.append((process.pid, start_time(process.pid)))
        return spawned

    def dead(self) -> tuple[int, int]:
        process = subprocess.Popen(["sleep", "3600"])
        identity = (process.pid, start_time(process.pid))
        process.kill()
        process.wait()
        return identity

    def stop(self) -> None:
        for process in self.processes:
            process.kill()
        for process in self.processes:
            process.wait()
        self.processes.clear()


# name: (records description, fit options, environment, expectation kind)
#   records: list of (kind, team, explicit) for live helpers, or "stale"
#   expectation: "auto" formula; "unreduced" team stays requested; "fixed" a
#   given automatic team never reduced; "raise" the fit fails.
CASES = {
    "alone": ([], {}, {}, "auto"),
    "seven_intents": ([("intent", 16, False)] * 7, {}, {}, "auto"),
    "explicit_lease_40": ([("lease", 40, True)], {}, {}, "auto"),
    "full_machine": ([("lease", "capacity", True)], {}, {}, "auto"),
    "many_sessions": ([("intent", 16, False)] * 60, {}, {}, "auto"),
    "stale": ("stale", {}, {}, "auto"),
    "explicit_request": ([("intent", 16, False)] * 7, {"num_threads": 16}, {}, "unreduced"),
    "savefe": ([("intent", 16, False)] * 7, {"retain_fes": True}, {}, "fixed"),
    "fixed_density": ([("intent", 16, False)] * 7,
                      {"target_rows_per_thread": UNCAPPED_ROWS_PER_THREAD}, {}, "fixed"),
    "switch_off": ([("intent", 16, False)] * 7, {}, {SWITCH_ENV: "0"}, "unreduced"),
    "unusable_mode": ([("intent", 16, False)] * 7, {}, {}, "unreduced"),
    "zero_weights": ([("intent", 16, False)] * 7, {"zero_weights": True}, {}, "auto"),
    "exception": ([("intent", 16, False)] * 7, {"nan_outcome": True}, {}, "raise"),
}
STATA_CASES = ("alone", "seven_intents", "explicit_request", "savefe", "switch_off")
R_CASES = ("alone", "seven_intents", "explicit_request")


def prepare(name: str, directory: Path, helpers: Helpers, capacity: int):
    records_spec = CASES[name][0]
    records: list[tuple] = []
    sessions = leased = 0
    if records_spec == "stale":
        dead_pid, dead_start = helpers.dead()
        (live_pid, live_start), = helpers.spawn(1)
        records = [
            (dead_pid, 40, dead_start, 500, LEASE, EXPLICIT_FLAG),
            (live_pid, 40, live_start + 1, 501, LEASE, EXPLICIT_FLAG),
        ]
    else:
        identities = helpers.spawn(len(records_spec))
        for index, ((kind, team, explicit), (pid, start)) in enumerate(
            zip(records_spec, identities)
        ):
            team = capacity if team == "capacity" else team
            records.append((pid, team, start, 100 + index,
                            LEASE if kind == "lease" else INTENT,
                            EXPLICIT_FLAG if explicit else 0))
            sessions += 1
            leased += team if kind == "lease" else 0
    mode = 0o644 if name == "unusable_mode" else 0o600
    before = write_registry(directory, records, mode)
    return before, sessions, leased


def child_fit(case: dict) -> None:
    """Runs in a fresh process: one fit, then the registry as this process sees it."""
    import numpy

    py = base.load_module(case["build"], "cpu")
    y, X, fe1, fe2 = base.design(ROWS)
    options = dict(drop_singletons=False, se_type="cluster", tol=1e-10,
                   tolerance_mode="reghdfe-comparable")
    for key in ("num_threads", "retain_fes", "target_rows_per_thread"):
        if key in case["options"]:
            options[key] = case["options"][key]
    weights = None
    if case["options"].get("zero_weights"):
        weights = numpy.ones(ROWS)
        weights[::97] = 0.0
    if case["options"].get("nan_outcome"):
        y = y.copy()
        y[ROWS // 2] = numpy.nan
    reg = py.HdfeRegressor(**options)
    result = {"pid": os.getpid(), "raised": False}
    try:
        reg.fit(y, X, fes=[fe1, fe2], clusters=fe1, weights=weights)
    except Exception as exc:  # the NaN case must raise
        result["raised"] = True
        result["error"] = str(exc)[:200]
    if not result["raised"]:
        result.update(
            effective=int(reg.threads_effective_), used=int(reg.threads_used_),
            capacity=int(reg.thread_capacity_), code=int(reg.thread_limit_code_),
            reason=str(reg.thread_limit_reason_), converged=bool(reg.converged_),
        )
        if result["effective"] > 1:
            one = py.HdfeRegressor(num_threads=1, **{
                key: value for key, value in options.items() if key != "num_threads"
            })
            one.fit(y, X, fes=[fe1, fe2], clusters=fe1, weights=weights)
            result["b_diff"] = float(numpy.max(numpy.abs(
                numpy.asarray(reg.coef_) - numpy.asarray(one.coef_))))
            result["se_diff"] = float(numpy.max(numpy.abs(
                numpy.asarray(reg.stderr_) - numpy.asarray(one.stderr_))))
    directory = Path(os.environ[REGISTRY_ENV])
    result["own_records_after"] = sum(
        1 for record in read_records(directory) if record[0] == os.getpid()
    )
    print("A3RESULT " + json.dumps(result))


def expected_team(name: str, capacity: int, sessions: int, leased: int) -> tuple[int, int]:
    """(expected effective team, requested team) for the child's capacity."""
    kind = CASES[name][3]
    options = CASES[name][1]
    requested = automatic_requested(capacity)
    if kind == "unreduced":
        requested = min(options.get("num_threads", requested), capacity)
        return requested, requested
    if kind == "fixed":
        team = legacy_team(capacity)
        if options.get("retain_fes"):
            team = min(max(1, ROWS // UNCAPPED_ROWS_PER_THREAD), capacity)
        return team, team
    return grant(requested, legacy_team(capacity), capacity, sessions + 1, leased), requested


def child_environment(directory: Path, extra: dict) -> dict:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", base.AUTO_ROWS_ENV, SWITCH_ENV):
        env.pop(key, None)
    env[REGISTRY_ENV] = str(directory)
    env["XHDFE_GPU_BACKEND"] = "cpu"
    env.update(extra)
    return env


def check_python_case(name: str, args, workdir: Path, capacity: int) -> None:
    directory = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=workdir))
    helpers = Helpers()
    try:
        before, sessions, leased = prepare(name, directory, helpers, capacity)
        helper_pids = {record[0] for record in read_records(directory)}
        case = {"name": name, "build": str(Path(args.build).resolve()),
                "options": CASES[name][1]}
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(case)],
            env=child_environment(directory, CASES[name][2]),
            text=True, capture_output=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.startswith("A3RESULT ")]
        if completed.returncode != 0 or len(lines) != 1:
            raise ContractFailure(f"{name}: child failed (rc={completed.returncode}):\n"
                                  f"{completed.stderr[-3000:]}")
        result = json.loads(lines[0][len("A3RESULT "):])
        after = read_records(directory)
        if result["own_records_after"] != 0 or any(r[0] == result["pid"] for r in after):
            raise ContractFailure(f"{name}: the fit left its record behind: {after}")
        if CASES[name][3] == "raise":
            if not result["raised"]:
                raise ContractFailure(f"{name}: the fit did not fail")
            print(f"PASS Python {name}: the failed fit released its record")
            return
        expected, requested = expected_team(name, result["capacity"], sessions, leased)
        coordinated = expected < requested
        problems = []
        if result["effective"] != expected:
            problems.append(f"effective={result['effective']} expected={expected}")
        if coordinated != (result["code"] == CODE) or coordinated != (result["reason"] == REASON):
            problems.append(f"code={result['code']} reason={result['reason']} "
                            f"coordinated={coordinated}")
        if not result["converged"]:
            problems.append("not converged")
        if not min(result["effective"], 2) <= result["used"] <= max(result["effective"], 1):
            problems.append(f"threads_used={result['used']}")
        for key in ("b_diff", "se_diff"):
            if result.get(key, 0.0) > base.CPU_NUMERIC_ATOL:
                problems.append(f"{key}={result[key]:.3e} against one thread")
        if name in ("switch_off", "unusable_mode"):
            if registry_path(directory).read_bytes() != before:
                problems.append("the registry was written although coordination was off")
        elif name == "stale":
            if after:
                problems.append(f"stale records were kept: {after}")
        elif {r[0] for r in after} != helper_pids:
            problems.append(f"helper records changed: {after}")
        if problems:
            raise ContractFailure(f"{name}: " + "; ".join(problems))
        print(f"PASS Python {name}: effective={result['effective']} "
              f"requested={requested} sessions={sessions + 1} leased={leased} "
              f"code={result['code']} used={result['used']} "
              f"b_diff={result.get('b_diff', 0.0):.1e}")
    finally:
        helpers.stop()
        shutil.rmtree(directory, ignore_errors=True)


STATA_DO = r"""
version 16.0
clear all
set more off
adopath ++ "{root}/stata"
discard
set obs {rows}
generate long row = _n - 1
generate int fe1 = mod(row, 1200)
generate int fe2 = mod(floor(row / 1200), 500)
generate double x1 = sin(row * .001) + mod(row * 17, 101) / 101
generate double x2 = cos(row * .0013) + mod(row * 29, 103) / 103
generate double y = .6*x1 - .25*x2 + .001*fe1 - .0005*fe2 + sin(row * .0021)
quietly {command}
display "A3RESULT|" e(threads_effective) "|" e(thread_limit_code) "|" ///
    e(thread_capacity) "|`e(thread_limit_reason)'|" e(threads_used)
"""


def check_stata_case(name: str, command: str, args, workdir: Path, capacity: int) -> None:
    directory = Path(tempfile.mkdtemp(prefix=f"stata-{name}-", dir=workdir))
    helpers = Helpers()
    try:
        _, sessions, leased = prepare(name, directory, helpers, capacity)
        options = CASES[name][1]
        extra = ""
        if "num_threads" in options:
            extra += f" numthreads({options['num_threads']})"
        # savefe is a suboption of absorb() in xhdfe.
        absorb = "absorb(fe1 fe2, savefe)" if options.get("retain_fes") else "absorb(fe1 fe2)"
        if command == "xhdfe":
            line = (f"xhdfe y x1 x2, {absorb} vce(cluster fe1) keepsingletons "
                    "tolerance(1e-10) tolerancemode(reghdfe-comparable) gpubackend(cpu)")
        else:
            line = ("xfepout y x1 x2, absorb(fe1 fe2) generate(a3r_) keepsingletons "
                    "tolerance(1e-10) gpubackend(cpu)")
        do_file = directory / "a3_case.do"
        do_file.write_text(STATA_DO.format(root=args.repo_root, rows=ROWS,
                                           command=line + extra))
        completed = subprocess.run(
            ["env", "-u", "LD_LIBRARY_PATH", args.stata, "-q", "-b", "do", do_file.name],
            cwd=directory, env=child_environment(directory, CASES[name][2]),
            text=True, capture_output=True,
        )
        log = (directory / "a3_case.log").read_text(errors="replace")
        rows = [line for line in log.splitlines() if line.startswith("A3RESULT|")]
        if completed.returncode != 0 or len(rows) != 1 or "\nr(" in log:
            raise ContractFailure(f"Stata {command} {name}: failed:\n{log[-3000:]}")
        effective, code, cap, reason, used = rows[0].split("|")[1:6]
        expected, requested = expected_team(name, int(cap), sessions, leased)
        coordinated = expected < requested
        if int(effective) != expected or coordinated != (int(code) == CODE) \
                or coordinated != (reason == REASON):
            raise ContractFailure(
                f"Stata {command} {name}: effective={effective} code={code} "
                f"reason={reason}; expected {expected} (coordinated={coordinated})")
        if read_records(directory) and {r[0] for r in read_records(directory)} - {
                p.pid for p in helpers.processes}:
            raise ContractFailure(f"Stata {command} {name}: a record was left behind")
        print(f"PASS Stata {command} {name}: effective={effective} code={code} "
              f"reason={reason} used={used}")
    finally:
        helpers.stop()
        shutil.rmtree(directory, ignore_errors=True)


R_EXPRESSION = r"""
.libPaths(c({lib}, .libPaths()))
library(xhdfe)
n <- {rows}L
row <- 0:(n - 1L)
fe1 <- row %% 1200L
fe2 <- (row %/% 1200L) %% 500L
X <- cbind(sin(row * .001) + ((row * 17L) %% 101L) / 101,
           cos(row * .0013) + ((row * 29L) %% 103L) / 103)
y <- .6 * X[, 1L] - .25 * X[, 2L] + .001 * fe1 - .0005 * fe2 + sin(row * .0021)
fit <- xhdfe_fit(y, X, list(fe1, fe2), cluster = fe1, drop_singletons = FALSE,
                 tol = 1e-10, tolerance_mode = "reghdfe-comparable",
                 backend = "cpu"{extra})
cat("A3RESULT|", fit$threads_effective, "|", fit$thread_limit_code, "|",
    fit$thread_capacity, "|", fit$thread_limit_reason, "|", fit$threads_used,
    "\n", sep = "")
"""


def check_r_case(name: str, args, workdir: Path, capacity: int) -> None:
    directory = Path(tempfile.mkdtemp(prefix=f"r-{name}-", dir=workdir))
    helpers = Helpers()
    try:
        _, sessions, leased = prepare(name, directory, helpers, capacity)
        options = CASES[name][1]
        extra = f", threads = {options['num_threads']}L" if "num_threads" in options else ""
        expression = R_EXPRESSION.format(lib=json.dumps(str(Path(args.r_lib).resolve())),
                                         rows=ROWS, extra=extra)
        completed = subprocess.run(
            ["Rscript", "-e", expression],
            env=child_environment(directory, CASES[name][2]), text=True, capture_output=True,
        )
        rows = [line for line in completed.stdout.splitlines() if line.startswith("A3RESULT|")]
        if completed.returncode != 0 or len(rows) != 1:
            raise ContractFailure(f"R {name}: failed:\n{completed.stderr[-3000:]}")
        effective, code, cap, reason, used = rows[0].split("|")[1:6]
        expected, requested = expected_team(name, int(cap), sessions, leased)
        coordinated = expected < requested
        if int(effective) != expected or coordinated != (int(code) == CODE) \
                or coordinated != (reason == REASON):
            raise ContractFailure(
                f"R {name}: effective={effective} code={code} reason={reason}; "
                f"expected {expected} (coordinated={coordinated})")
        print(f"PASS R {name}: effective={effective} code={code} reason={reason} used={used}")
    finally:
        helpers.stop()
        shutil.rmtree(directory, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=("python", "stata", "r", "all"), default="python")
    parser.add_argument("--build", default=str(ROOT / "build"))
    parser.add_argument("--r-lib", default="")
    parser.add_argument("--stata", default="/usr/local/stata/stata-mp")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--workdir", default="", help="parent of the private registries")
    parser.add_argument("--cases", default="", help="comma-separated subset")
    parser.add_argument("--child", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child_fit(json.loads(args.child))
        return 0
    if not sys.platform.startswith("linux"):
        print("SKIP auto team coordination check: needs /proc start times (Linux)")
        return 0
    capacity = len(os.sched_getaffinity(0))
    if capacity < 2 * PROBLEM_CAP:
        print(f"SKIP auto team coordination check: {capacity} processors, needs {2 * PROBLEM_CAP}")
        return 0
    selected = [name for name in args.cases.split(",") if name] or list(CASES)
    workdir = Path(tempfile.mkdtemp(prefix="xhdfe-a3-", dir=args.workdir or None))
    try:
        if args.surface in ("python", "all"):
            for name in selected:
                check_python_case(name, args, workdir, capacity)
        if args.surface in ("stata", "all"):
            for name in [name for name in selected if name in STATA_CASES]:
                check_stata_case(name, "xhdfe", args, workdir, capacity)
            if "seven_intents" in selected:
                check_stata_case("seven_intents", "xfepout", args, workdir, capacity)
        if args.surface in ("r", "all"):
            if not args.r_lib:
                raise ContractFailure("--r-lib is required for the R surface")
            for name in [name for name in selected if name in R_CASES]:
                check_r_case(name, args, workdir, capacity)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("PASS: automatic team coordination contract")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
