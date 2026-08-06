#!/usr/bin/env python3
"""Functional contract for explicit xhdfe thread requests.

The synthetic balanced design is deliberately large enough to expose useful
parallel work while converging in one deterministic sweep.  This is a
functional gate: elapsed time is printed nowhere and is never an acceptance
criterion.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREADS = (1, 2, 8, 16, 24, 48)
CPU_NUMERIC_ATOL = 1e-13
CUDA_NUMERIC_ATOL = 1e-9


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


def python_contract(args: argparse.Namespace) -> None:
    import numpy

    build = Path(args.build).resolve()
    modules = sorted(build.glob("py_hdfe_v11*.so"))
    if len(modules) != 1:
        raise ContractFailure(
            f"expected one py_hdfe_v11 extension in {build}, found {modules}"
        )
    os.environ["XHDFE_GPU_BACKEND"] = args.backend
    sys.path.insert(0, str(build))
    py = importlib.import_module("py_hdfe_v11")
    if Path(py.__file__).resolve() != modules[0].resolve():
        raise ContractFailure(
            f"module custody failure: {Path(py.__file__).resolve()} != "
            f"{modules[0].resolve()}"
        )

    row = numpy.arange(args.n, dtype=numpy.int64)
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


def main() -> int:
    args = parse_args()
    check_no_process_global_eigen_threads()
    if args.surface in {"python", "all"}:
        python_contract(args)
    if args.surface in {"r", "all"}:
        r_contract(args)
    print("PASS: explicit thread contract")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
