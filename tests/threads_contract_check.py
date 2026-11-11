#!/usr/bin/env python3
"""Functional contract for xhdfe thread requests.

The synthetic balanced design is deliberately large enough to expose useful
parallel work while converging in one deterministic sweep.  This is a
functional gate: elapsed time is printed nowhere and is never an acceptance
criterion.

The explicit checks cover positive thread requests. The automatic checks cover
num_threads=0 at the row-count boundaries of HdfeRegressorV11::resolve_threads
on the CPU backend, one fresh process per case. Builds that predate the
automatic rows-per-thread rule (2.30.1 and earlier) take --checks explicit.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREADS = (1, 2, 8, 16, 24, 48)
CPU_NUMERIC_ATOL = 1e-13
CUDA_NUMERIC_ATOL = 1e-9
# Automatic rule: HdfeRegressorV11::resolve_threads in src/hdfe_regressor_v11.cpp.
CORE_SOURCE = ROOT / "src" / "hdfe_regressor_v11.cpp"
AUTO_ROWS_ENV = "XHDFE_AUTO_ROWS_PER_THREAD"
UNCAPPED_ROWS_PER_THREAD = 500000
MIN_PARALLEL_ROWS = 20000
AUTOMATIC_ROWS = (1000000, 600000, 400000, 250000, 249999, 100000, 99999)


class ContractFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=("python", "r", "all"), default="python")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--build", default=str(ROOT / "build"))
    parser.add_argument("--r-lib", default="")
    parser.add_argument("--n", type=int, default=600000)
    parser.add_argument(
        "--threads",
        default=",".join(str(value) for value in DEFAULT_THREADS),
    )
    parser.add_argument(
        "--checks", choices=("explicit", "automatic", "all"), default="all"
    )
    parser.add_argument("--automatic-case", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.thread_values = tuple(int(value) for value in args.threads.split(","))
    except ValueError as exc:
        parser.error(f"invalid --threads: {exc}")
    if not args.thread_values or any(value < 1 for value in args.thread_values):
        parser.error("--threads must contain positive integers")
    if args.n < 600000:
        parser.error("--n must be >= 600000 to expose 48-way useful work")
    if args.surface in {"r", "all"} and not args.r_lib:
        parser.error("--r-lib is required for the R surface")
    return args


def expected_threads(requested: int) -> int:
    available = os.cpu_count() or 1
    return min(requested, available)


def check_no_process_global_eigen_threads() -> None:
    roots = (
        ROOT / "src",
        ROOT / "include",
        ROOT / "python",
        ROOT / "stata" / "src",
    )
    findings = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".cpp", ".cc", ".cxx", ".hpp", ".h"}:
                continue
            for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if "Eigen::setNbThreads" in line:
                    findings.append(f"{path.relative_to(ROOT)}:{line_number}")
    if findings:
        raise ContractFailure(
            "process-global Eigen::setNbThreads found: " + ", ".join(findings)
        )


def load_module(build_dir: str, backend: str):
    build = Path(build_dir).resolve()
    modules = sorted(build.glob("py_hdfe_v11*.so"))
    if len(modules) != 1:
        raise ContractFailure(
            f"expected one py_hdfe_v11 extension in {build}, found {modules}"
        )
    os.environ["XHDFE_GPU_BACKEND"] = backend
    sys.path.insert(0, str(build))
    py = importlib.import_module("py_hdfe_v11")
    if Path(py.__file__).resolve() != modules[0].resolve():
        raise ContractFailure(
            f"module custody failure: {Path(py.__file__).resolve()} != "
            f"{modules[0].resolve()}"
        )
    return py


def design(n: int):
    import numpy

    row = numpy.arange(n, dtype=numpy.int64)
    fe1 = (row % 1200).astype(numpy.int32)
    fe2 = ((row // 1200) % 500).astype(numpy.int32)
    X = numpy.asfortranarray(
        numpy.column_stack(
            (
                numpy.sin(row * 0.001) + ((row * 17) % 101) / 101,
                numpy.cos(row * 0.0013) + ((row * 29) % 103) / 103,
            )
        )
    )
    y = (
        0.6 * X[:, 0]
        - 0.25 * X[:, 1]
        + 0.001 * fe1
        - 0.0005 * fe2
        + numpy.sin(row * 0.0021)
    )
    return y, X, fe1, fe2


def python_contract(args: argparse.Namespace) -> None:
    import numpy

    py = load_module(args.build, args.backend)
    y, X, fe1, fe2 = design(args.n)
    reference = None
    tolerance = CUDA_NUMERIC_ATOL if args.backend == "cuda" else CPU_NUMERIC_ATOL
    for requested in args.thread_values:
        reg = py.HdfeRegressor(
            num_threads=requested,
            drop_singletons=False,
            se_type="cluster",
            tol=1e-10,
            tolerance_mode="reghdfe-comparable",
        )
        reg.fit(y, X, fes=[fe1, fe2], clusters=fe1)
        expected = expected_threads(requested)
        diagnostics = (
            int(reg.threads_requested_),
            int(reg.threads_effective_),
            int(reg.threads_used_),
        )
        if diagnostics != (requested, expected, expected):
            raise ContractFailure(
                f"Python threads({requested}) diagnostics={diagnostics}, "
                f"expected ({requested}, {expected}, {expected})"
            )
        if not reg.converged_:
            raise ContractFailure(f"Python threads({requested}) did not converge")
        if args.backend == "cuda":
            if not reg.gpu_used_ or int(reg.gpu_status_code_) != 1:
                raise ContractFailure(
                    f"Python threads({requested}) silently failed CUDA use"
                )
        else:
            if reg.gpu_used_ or int(reg.gpu_status_code_) != 0:
                raise ContractFailure(
                    f"Python threads({requested}) unexpectedly used CUDA"
                )
        coefficients = numpy.asarray(reg.coef_, dtype=numpy.float64)
        standard_errors = numpy.asarray(reg.stderr_, dtype=numpy.float64)
        if reference is None:
            reference = (coefficients.copy(), standard_errors.copy())
        b_diff = float(numpy.max(numpy.abs(coefficients - reference[0])))
        se_diff = float(numpy.max(numpy.abs(standard_errors - reference[1])))
        if b_diff > tolerance or se_diff > tolerance:
            raise ContractFailure(
                f"Python threads({requested}) numerical drift: "
                f"b={b_diff:.17g}, se={se_diff:.17g}, atol={tolerance:.1e}"
            )
        print(
            f"PASS Python backend={args.backend} requested={requested} "
            f"effective={expected} used={reg.threads_used_} "
            f"max_b_diff={b_diff:.3e} max_se_diff={se_diff:.3e}"
        )


def r_contract(args: argparse.Namespace) -> None:
    threads = ",".join(f"{value}L" for value in args.thread_values)
    tolerance = CUDA_NUMERIC_ATOL if args.backend == "cuda" else CPU_NUMERIC_ATOL
    gpu_assertions = (
        'fit$gpu_used, fit$gpu_status_code == 1L, fit$gpu_status == "used"'
        if args.backend == "cuda"
        else "!fit$gpu_used, fit$gpu_status_code == 0L"
    )
    r_expression = f'''
.libPaths(c({json_string(str(Path(args.r_lib).resolve()))}, .libPaths()))
library(xhdfe)
n <- {args.n}L
row <- 0:(n - 1L)
fe1 <- row %% 1200L
fe2 <- (row %/% 1200L) %% 500L
X <- cbind(
  sin(row * .001) + ((row * 17L) %% 101L) / 101,
  cos(row * .0013) + ((row * 29L) %% 103L) / 103
)
y <- .6 * X[, 1L] - .25 * X[, 2L] + .001 * fe1 - .0005 * fe2 +
  sin(row * .0021)
reference_b <- reference_se <- NULL
for (requested in c({threads})) {{
  fit <- xhdfe_fit(
    y, X, list(fe1, fe2), cluster = fe1, drop_singletons = FALSE,
    threads = requested, tol = 1e-10,
    tolerance_mode = "reghdfe-comparable", backend = "{args.backend}"
  )
  expected <- min(requested, parallel::detectCores(logical = TRUE))
  stopifnot(
    fit$converged,
    fit$threads_requested == requested,
    fit$threads_effective == expected,
    fit$threads_used == expected,
    {gpu_assertions}
  )
  b <- coef(fit)
  se <- fit$se
  if (is.null(reference_b)) {{
    reference_b <- b
    reference_se <- se
  }}
  b_diff <- max(abs(b - reference_b))
  se_diff <- max(abs(se - reference_se))
  stopifnot(b_diff <= {tolerance}, se_diff <= {tolerance})
  cat(
    "PASS R backend={args.backend} requested=", requested,
    " effective=", expected, " used=", fit$threads_used,
    " max_b_diff=", format(b_diff, scientific = TRUE),
    " max_se_diff=", format(se_diff, scientific = TRUE), "\\n", sep = ""
  )
}}
'''
    completed = subprocess.run(
        ["Rscript", "-e", r_expression], text=True, capture_output=True
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise ContractFailure(
            f"R thread contract failed (rc={completed.returncode}):\n"
            f"{completed.stderr[-4000:]}"
        )


def json_string(value: str) -> str:
    import json

    return json.dumps(value)


def absorb_rows_per_thread() -> int:
    """kAutoRowsPerThreadAbsorb as defined in the canonical core."""
    match = re.search(
        r"constexpr int kAutoRowsPerThreadAbsorb = (\d+);", CORE_SOURCE.read_text()
    )
    if match is None:
        raise ContractFailure(f"kAutoRowsPerThreadAbsorb not found in {CORE_SOURCE}")
    return int(match.group(1))


def automatic_team(rows: int, capped: bool, rows_per_thread: int) -> int:
    """Team resolve_threads selects for num_threads=0, before runtime capacity.

    capped: fixed effects absorbed without being retained.
    """
    if rows < MIN_PARALLEL_ROWS:
        return 1
    team = max(1, rows // rows_per_thread)
    if capped:
        team = min(team, 1 if rows < 100000 else 8 if rows < 250000 else 16)
    return team


def automatic_cases() -> list[dict]:
    absorb = absorb_rows_per_thread()
    cases = [
        dict(rows=rows, retain_fes=False, target=None, env=None,
             team=automatic_team(rows, True, absorb))
        for rows in AUTOMATIC_ROWS
    ]
    # Retained fixed effects keep one thread per 500 000 rows.
    cases.append(dict(rows=1000000, retain_fes=True, target=None, env=None,
                      team=automatic_team(1000000, False, UNCAPPED_ROWS_PER_THREAD)))
    # An explicit density and the environment override both restore the
    # 2.30.1 rule on the capped path.
    for target, env in ((500000, None), (None, "500000")):
        density = target if target is not None else int(env)
        cases.append(dict(rows=600000, retain_fes=False, target=target, env=env,
                          team=automatic_team(600000, True, density)))
    return cases


def case_label(case: dict) -> str:
    return (
        f"rows={case['rows']} retain_fes={case['retain_fes']} "
        f"target_rows_per_thread={case['target']} {AUTO_ROWS_ENV}={case['env']}"
    )


def python_automatic_case(args: argparse.Namespace, case: dict) -> None:
    import numpy

    py = load_module(args.build, "cpu")
    y, X, fe1, fe2 = design(case["rows"])
    options = dict(
        drop_singletons=False,
        se_type="cluster",
        tol=1e-10,
        tolerance_mode="reghdfe-comparable",
        retain_fes=case["retain_fes"],
    )
    if case["target"] is not None:
        options["target_rows_per_thread"] = case["target"]
    reg = py.HdfeRegressor(**options)
    reg.fit(y, X, fes=[fe1, fe2], clusters=fe1)
    label = case_label(case)
    expected = min(case["team"], int(reg.thread_capacity_))
    diagnostics = (int(reg.threads_requested_), int(reg.threads_effective_))
    if diagnostics != (0, expected):
        raise ContractFailure(
            f"Python automatic {label}: (requested, effective)={diagnostics}, "
            f"expected (0, {expected})"
        )
    # The absorber may narrow an automatic team by fixed-effect shape, so
    # threads_used is bounded by the team rather than equal to it.
    used = int(reg.threads_used_)
    if not min(expected, 2) <= used <= expected:
        raise ContractFailure(
            f"Python automatic {label}: threads_used={used} outside "
            f"[{min(expected, 2)}, {expected}]"
        )
    if not reg.converged_:
        raise ContractFailure(f"Python automatic {label} did not converge")
    if reg.gpu_used_ or int(reg.gpu_status_code_) != 0:
        raise ContractFailure(f"Python automatic {label} unexpectedly used CUDA")
    b_diff = se_diff = 0.0
    if expected > 1:
        one = py.HdfeRegressor(num_threads=1, **options)
        one.fit(y, X, fes=[fe1, fe2], clusters=fe1)
        b_diff = float(numpy.max(numpy.abs(
            numpy.asarray(reg.coef_, dtype=numpy.float64)
            - numpy.asarray(one.coef_, dtype=numpy.float64)
        )))
        se_diff = float(numpy.max(numpy.abs(
            numpy.asarray(reg.stderr_, dtype=numpy.float64)
            - numpy.asarray(one.stderr_, dtype=numpy.float64)
        )))
        if b_diff > CPU_NUMERIC_ATOL or se_diff > CPU_NUMERIC_ATOL:
            raise ContractFailure(
                f"Python automatic {label} versus one thread: "
                f"b={b_diff:.17g}, se={se_diff:.17g}, atol={CPU_NUMERIC_ATOL:.1e}"
            )
    print(
        f"PASS Python automatic {label} effective={expected} used={used} "
        f"capacity={reg.thread_capacity_} reason={reg.thread_limit_reason_} "
        f"max_b_diff={b_diff:.3e} max_se_diff={se_diff:.3e}"
    )


def r_automatic_expression(args: argparse.Namespace, case: dict) -> str:
    options = ""
    if case["retain_fes"]:
        options += ", save_fe = TRUE"
    if case["target"] is not None:
        options += f", target_rows_per_thread = {case['target']}"
    return f'''
.libPaths(c({json_string(str(Path(args.r_lib).resolve()))}, .libPaths()))
library(xhdfe)
n <- {case["rows"]}L
row <- 0:(n - 1L)
fe1 <- row %% 1200L
fe2 <- (row %/% 1200L) %% 500L
X <- cbind(
  sin(row * .001) + ((row * 17L) %% 101L) / 101,
  cos(row * .0013) + ((row * 29L) %% 103L) / 103
)
y <- .6 * X[, 1L] - .25 * X[, 2L] + .001 * fe1 - .0005 * fe2 +
  sin(row * .0021)
fit_case <- function(...) xhdfe_fit(
  y, X, list(fe1, fe2), cluster = fe1, drop_singletons = FALSE, tol = 1e-10,
  tolerance_mode = "reghdfe-comparable", backend = "cpu"{options}, ...
)
fit <- fit_case()
expected <- min({case["team"]}L, fit$thread_capacity)
stopifnot(
  fit$converged,
  fit$threads_requested == 0L,
  fit$threads_effective == expected,
  fit$threads_used >= min(expected, 2L),
  fit$threads_used <= expected,
  !fit$gpu_used, fit$gpu_status_code == 0L
)
b_diff <- se_diff <- 0
if (expected > 1L) {{
  one <- fit_case(threads = 1L)
  b_diff <- max(abs(coef(fit) - coef(one)))
  se_diff <- max(abs(fit$se - one$se))
  stopifnot(b_diff <= {CPU_NUMERIC_ATOL}, se_diff <= {CPU_NUMERIC_ATOL})
}}
cat("PASS R automatic {case_label(case)} effective=", expected,
    " used=", fit$threads_used, " capacity=", fit$thread_capacity,
    " reason=", fit$thread_limit_reason,
    " max_b_diff=", format(b_diff, scientific = TRUE),
    " max_se_diff=", format(se_diff, scientific = TRUE), "\\n", sep = "")
'''


def automatic_contract(args: argparse.Namespace) -> None:
    """Check every automatic case in a fresh process on the CPU backend."""
    for case in automatic_cases():
        env = dict(os.environ)
        # The expected team assumes the default OpenMP team size and the
        # built-in density unless the case sets the override.
        env.pop("OMP_NUM_THREADS", None)
        env.pop(AUTO_ROWS_ENV, None)
        if case["env"] is not None:
            env[AUTO_ROWS_ENV] = case["env"]
        commands = []
        if args.surface in {"python", "all"}:
            commands.append(("Python", [
                sys.executable, str(Path(__file__).resolve()),
                "--build", str(Path(args.build).resolve()),
                "--automatic-case", json.dumps(case),
            ]))
        if args.surface in {"r", "all"}:
            commands.append(("R", ["Rscript", "-e", r_automatic_expression(args, case)]))
        for surface, command in commands:
            completed = subprocess.run(
                command, env=env, text=True, capture_output=True
            )
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.returncode != 0:
                raise ContractFailure(
                    f"{surface} automatic {case_label(case)} failed "
                    f"(rc={completed.returncode}):\n{completed.stderr[-4000:]}"
                )


def main() -> int:
    args = parse_args()
    if args.automatic_case:
        python_automatic_case(args, json.loads(args.automatic_case))
        return 0
    check_no_process_global_eigen_threads()
    if args.checks in {"explicit", "all"}:
        if args.surface in {"python", "all"}:
            python_contract(args)
        if args.surface in {"r", "all"}:
            r_contract(args)
        print("PASS: explicit thread contract")
    if args.checks in {"automatic", "all"}:
        automatic_contract(args)
        print("PASS: automatic thread contract")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
