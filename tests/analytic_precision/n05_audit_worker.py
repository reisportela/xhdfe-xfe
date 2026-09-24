"""Run one isolated N05 native fit and emit raw JSON on stdout."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


OLD_AUDIT_FLAGS = (
    "XHDFE_ORDINARY_PRECISION_DIAG",
    "XHDFE_ORDINARY_REFERENCE_TRACE",
    "XHDFE_ORDINARY_SUPPORT_TRACE",
    "XHDFE_ORDINARY_SUPPORT_BATCH",
    "XHDFE_ORDINARY_SUPPORT_BATCH_TRACE",
    "XHDFE_ORDINARY_STRUCTURED_COEFFICIENT",
    "XHDFE_ORDINARY_SANDWICH_PROFILE",
    "XHDFE_ORDINARY_FINAL_INFERENCE",
    "XHDFE_OLS_REFINEMENT_TRACE",
)
JOB_KEYS = frozenset(
    (
        "name",
        "module",
        "module_sha256",
        "backend",
        "fixture",
        "mode",
        "certify",
        "old_flags",
        "cache_mode",
        "cache_path",
        "mutation",
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_arrays(arrays: Dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        descriptor = json.dumps(
            [name, value.dtype.str, list(value.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        raw = memoryview(value).cast("B")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _sign(index: np.ndarray, bit: int) -> np.ndarray:
    return (2 * ((index // (2**bit)) % 2) - 1).astype(float)


def _ordinary_fixture(name: str, mutation: str) -> dict:
    index = np.arange(3072)
    worker = index // 768
    firm = (index // 256) % 3
    a, b, c, d, e, zsign = (_sign(index, bit) for bit in range(6))
    X = np.column_stack(
        (
            a + 0.5 * b + 0.25 * worker,
            c - 0.5 * firm,
            d + 0.125 * worker - 0.25 * firm,
        )
    )
    beta = np.array([0.75, -0.5, 0.25])
    residual = 0.125 * e
    y = X @ beta + 3.0 + 2.0 * worker - firm + residual
    options = dict(
        num_threads=2,
        max_iter=1000,
        tol=1e-8,
        tolerance_mode="reghdfe-comparable",
        drop_singletons=False,
        fit_intercept=True,
        se_type="unadjusted",
    )
    kwargs = {"fes": [worker, firm]}
    arrays = {"y": y, "X": X, "fe0": worker, "fe1": firm}
    expected_beta = beta.copy()
    weights = None
    fweights = False
    expected_omitted = 0

    if name == "ordinary_homo_noconstant":
        options["fit_intercept"] = False
    elif name == "ordinary_robust":
        options["se_type"] = "robust"
    elif name == "ordinary_robust_aweight":
        options["se_type"] = "robust"
        weights = 1.0 + worker + 0.25 * firm
    elif name == "ordinary_oneway":
        options["se_type"] = "cluster"
        clusters = index // 5
        kwargs["clusters"] = clusters
        arrays["cluster0"] = clusters
    elif name == "ordinary_multiway":
        options["se_type"] = "cluster"
        clusters = np.column_stack((index // 5, (index // 7) + 3 * worker))
        kwargs["clusters"] = clusters
        arrays["clusters"] = clusters
    elif name == "ordinary_collinear":
        X = np.column_stack((X, 2.0 * X[:, 0]))
        kwargs["fes"] = [worker, firm]
        arrays["X"] = X
        expected_beta = None
        expected_omitted = 1
    elif name == "ordinary_fweight_robust":
        options["se_type"] = "robust"
        weights = (1 + (worker + firm) % 3).astype(float)
        fweights = True
    elif name == "ordinary_slope":
        slope = 1.0 + 0.25 * zsign
        y = y + np.array([-0.6, -0.2, 0.3, 0.8])[worker] * slope
        kwargs["slopes"] = [(0, slope, True)]
        arrays["y"] = y
        arrays["slope0"] = slope
    elif name == "ordinary_cache":
        weights = 1.0 + worker + 0.25 * firm
        if mutation == "y":
            y = y + 0.125 * X[:, 0]
            expected_beta = beta + np.array([0.125, 0.0, 0.0])
            arrays["y"] = y
        elif mutation == "weight":
            weights = weights + 0.5 * (firm == 1)
        elif mutation != "none":
            raise ValueError(f"unknown cache mutation {mutation}")
    elif name != "ordinary_homo_constant":
        raise ValueError(f"unknown ordinary fixture {name}")

    if weights is not None:
        kwargs["weights"] = weights
        kwargs["fweights"] = fweights
        arrays["weights"] = weights
        effective_weights = weights if fweights else weights * (len(index) / weights.sum())
    else:
        effective_weights = np.ones(len(index))
    expected_rss = float(effective_weights @ (residual * residual))
    return dict(
        y=y,
        X=X,
        options=options,
        kwargs=kwargs,
        arrays=arrays,
        expected_beta=expected_beta,
        expected_residual=residual,
        expected_rss=expected_rss,
        expected_omitted=expected_omitted,
        group_ids=None,
    )


def _iv_fixture(name: str) -> dict:
    index = np.arange(3072)
    worker = index // 768
    firm = (index // 256) % 3
    a, b, c, d = (_sign(index, bit) for bit in range(4))
    X = np.column_stack((a + 0.5 * b + 0.25 * worker, c - 0.5 * firm))
    instruments = (a + 0.25 * firm)[:, None]
    residual = 0.25 * b + 0.125 * d
    beta = np.array([0.75, -0.5])
    y = X @ beta + 3.0 + 2.0 * worker - firm + residual
    score = np.column_stack((a, c))
    assert np.max(np.abs(score.T @ residual)) == 0.0
    assert abs((a + 0.5 * b) @ residual) > 0.1 * len(index)
    se_type = "unadjusted" if name == "iv_homo" else "robust"
    if name not in ("iv_homo", "iv_robust"):
        raise ValueError(f"unknown IV fixture {name}")
    return dict(
        y=y,
        X=X,
        options=dict(
            num_threads=2,
            max_iter=1000,
            tol=1e-8,
            tolerance_mode="reghdfe-comparable",
            drop_singletons=False,
            fit_intercept=True,
            se_type=se_type,
        ),
        kwargs=dict(
            fes=[worker, firm], instruments=instruments, endogenous_idx=[0]
        ),
        arrays=dict(
            y=y,
            X=X,
            fe0=worker,
            fe1=firm,
            instruments=instruments,
        ),
        expected_beta=beta,
        expected_residual=residual,
        expected_rss=float(residual @ residual),
        expected_omitted=0,
        group_ids=None,
    )


def _group_fixture(name: str) -> dict:
    from cases import BETA, generate

    source = "group_uniform_sum" if name == "group_sum" else "group_uniform_mean"
    if name not in ("group_sum", "group_mean"):
        raise ValueError(f"unknown grouped fixture {name}")
    case = generate(source)
    frame = case.frame
    y = frame.y.to_numpy()
    X = frame[["x1", "x2", "x3"]].to_numpy()
    fes = [frame[column].to_numpy(dtype=np.int64) for column in case.fes]
    group = frame.group.to_numpy(dtype=np.int64)
    individual = frame.individual.to_numpy(dtype=np.int64)
    oracle = case.oracle()
    exact = case.group_frame.set_index("group").exact_residual
    arrays = {"y": y, "X": X, "group": group, "individual": individual}
    arrays.update({f"fe{j}": value for j, value in enumerate(fes)})
    return dict(
        y=y,
        X=X,
        options=dict(
            num_threads=2,
            max_iter=100000,
            tol=1e-8,
            tolerance_mode="reghdfe-comparable",
            drop_singletons=False,
            fit_intercept=True,
            se_type="unadjusted",
            dofadjustments="exact",
        ),
        kwargs=dict(
            fes=fes,
            group=group,
            individual=individual,
            aggregation=case.aggregation,
        ),
        arrays=arrays,
        expected_beta=BETA.copy(),
        expected_residual_by_group=exact,
        expected_rss=float(oracle["rss"]),
        expected_omitted=0,
        group_ids=group,
    )


def _fixture(name: str, mutation: str) -> dict:
    if name.startswith("iv_"):
        return _iv_fixture(name)
    if name.startswith("group_"):
        return _group_fixture(name)
    return _ordinary_fixture(name, mutation)


def _configure_environment(job: dict) -> None:
    for name in ("XHDFE_CERTIFY", *OLD_AUDIT_FLAGS):
        os.environ.pop(name, None)
    os.environ["XHDFE_GPU_BACKEND"] = job["backend"]
    os.environ["XHDFE_MOBILITY_MODE"] = "off"
    os.environ["XHDFE_FE_NORMALIZE"] = "component"
    os.environ.pop("XHDFE_ABSORPTION_CACHE", None)
    os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = job["cache_mode"]
    if job["cache_path"]:
        os.environ["XHDFE_ABSORPTION_CACHE"] = job["cache_path"]
    if job["certify"] != "unset":
        os.environ["XHDFE_CERTIFY"] = job["certify"]
    if job["old_flags"]:
        for name in OLD_AUDIT_FLAGS:
            os.environ[name] = "1"


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct native module specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _work_counter(module_path: Path) -> Tuple[object, int]:
    library = ctypes.CDLL(str(module_path))
    counter = library.xhdfe_private_n05_work_count
    counter.argtypes = [ctypes.c_int]
    counter.restype = ctypes.c_ulonglong
    return (counter, int(counter(1)))


def _max_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if not isinstance(job, dict) or frozenset(job) != JOB_KEYS:
        raise ValueError("job keys differ from the pinned N05 worker contract")
    if job["backend"] not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    if job["mode"] not in ("xhdfe-fast", "reghdfe-comparable"):
        raise ValueError("unsupported mode")
    if job["certify"] not in ("unset", "0", "1"):
        raise ValueError("certify must be unset, 0, or 1")
    if job["cache_mode"] not in ("off", "read", "write"):
        raise ValueError("invalid cache mode")
    if not isinstance(job["old_flags"], bool):
        raise ValueError("old_flags must be boolean")

    module_path = Path(job["module"]).resolve()
    if _sha256(module_path) != job["module_sha256"]:
        raise RuntimeError("native module custody mismatch")
    _configure_environment(job)
    cpp = _load_module(module_path)
    counter, counter_before_reset = _work_counter(module_path)
    data = _fixture(job["fixture"], job["mutation"])
    data["options"]["tolerance_mode"] = job["mode"]
    input_sha_before = _digest_arrays(data["arrays"])
    model = cpp.HdfeRegressor(**data["options"])
    result = dict(
        name=job["name"],
        fixture=job["fixture"],
        entrypoint="group_fit" if data["group_ids"] is not None else "fit",
        requested_backend=job["backend"],
        mode=job["mode"],
        certify=job["certify"],
        module=str(module_path),
        module_sha256=job["module_sha256"],
        input_sha_before=input_sha_before,
        counter_before_reset=counter_before_reset,
        fit_intercept=bool(data["options"]["fit_intercept"]),
        structural_columns=int(data["X"].shape[1]),
    )
    try:
        model.fit(data["y"], data["X"], **data["kwargs"])
        coefficients = np.asarray(model.coef_)
        residuals = np.asarray(model.residuals_)
        selected = np.asarray(model.sample_index_, dtype=int)
        if data["group_ids"] is None:
            expected_residual = data["expected_residual"]
        else:
            selected_groups = data["group_ids"][selected]
            expected_residual = data["expected_residual_by_group"].loc[
                selected_groups
            ].to_numpy()
            result["selected_groups"] = selected_groups.tolist()
        beta_error = None
        if data["expected_beta"] is not None:
            p = len(data["expected_beta"])
            beta_error = _max_error(coefficients[:p], data["expected_beta"])
        omitted = np.asarray(model.omitted_, dtype=bool)
        result.update(
            rc=0,
            beta=coefficients.tolist(),
            V=np.asarray(model.covariance_).tolist(),
            residuals=residuals.tolist(),
            rss=float(model.rss_),
            iterations=int(model.num_iterations_),
            method=int(model.absorption_method_used),
            converged=bool(model.converged_),
            precision_certified=bool(model.precision_certified_),
            gpu_used=bool(model.gpu_used_),
            gpu_status=int(model.gpu_status_code_),
            backend_actual="cuda" if model.gpu_used_ else "cpu",
            omitted=omitted.tolist(),
            omitted_count=int(omitted.sum()),
            nobs=int(model.nobs_),
            df_resid=float(model.df_resid_),
            sample_index=selected.tolist(),
            beta_error=beta_error,
            residual_error=_max_error(residuals, expected_residual),
            rss_error=abs(float(model.rss_) - data["expected_rss"]),
            expected_omitted=int(data["expected_omitted"]),
        )
    except Exception as error:
        result.update(rc=1, error=str(error), error_type=type(error).__name__)
    result["input_sha_after"] = _digest_arrays(data["arrays"])
    result["n05_work_count"] = int(counter(0))
    cache_path = Path(job["cache_path"]) if job["cache_path"] else None
    result["cache_exists"] = bool(cache_path is not None and cache_path.is_file())
    result["cache_sha256"] = (
        _sha256(cache_path) if cache_path is not None and cache_path.is_file() else None
    )
    print(json.dumps(result, allow_nan=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
