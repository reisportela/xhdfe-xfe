#!/usr/bin/env python3
"""Analytic FE recovery, unchanged outputs and explicit audit-only receipts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(a):
    import numpy as np
    os.environ["XHDFE_GPU_BACKEND"] = a.backend
    if a.audit:
        os.environ["XHDFE_CERTIFY"] = "1"
    else:
        os.environ.pop("XHDFE_CERTIFY", None)
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", a.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    i = np.arange(64)
    w, f = (i % 4).astype(np.int32), ((i//4) % 4).astype(np.int32)
    x, noise = ((i//16) % 2)*2.-1., (i//32)*2.-1.
    y = .5*x + .25*w - .125*f + .125*noise
    m = core.HdfeRegressor(num_threads=2, retain_fes=True,
                           tolerance_mode="reghdfe-comparable", se_type="unadjusted")
    m.fit(y, x[:, None], [w, f])
    assert m.converged_ and m.fe_recovery_converged_
    assert bool(m.gpu_used_) == (a.backend == "cuda")
    np.testing.assert_allclose(m.coef_, [.5, .1875], rtol=0, atol=1e-12)
    for actual, reference in zip(m.fe_effects_, [.25*(w-1.5), -.125*(f-1.5)]):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.residuals_, .125*noise, rtol=0, atol=1e-12)
    assert abs(m.covariance_[0, 0]-1/3584) < 1e-14
    arrays = dict(b=m.coef_, V=m.covariance_, u=m.residuals_)
    arrays.update({"fe"+str(k):v for k,v in enumerate(m.fe_effects_)})
    result = dict(status="PASS", audit=a.audit, backend=a.backend,
                  hashes={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
                          for k,v in arrays.items()})
    (a.scratch/("audit_"+str(a.audit)+".json")).write_text(json.dumps(result, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--audit", type=int, choices=(0, 1), default=0)
    a = p.parse_args()
    if a.worker:
        worker(a)
        return
    a.scratch.mkdir(parents=True, exist_ok=False)
    for audit in (0, 1):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                   "--module", str(a.module.resolve()), "--scratch", str(a.scratch.resolve()),
                   "--backend", a.backend, "--audit", str(audit)]
        result = subprocess.run(command, cwd=a.scratch, capture_output=True, text=True, timeout=420)
        (a.scratch/("audit_"+str(audit)+".log")).write_text(result.stdout+result.stderr)
        result.check_returncode()
        assert ("XHDFE_N05_AUDIT" in result.stderr) == bool(audit)
    records = [json.loads((a.scratch/("audit_"+str(i)+".json")).read_text()) for i in (0, 1)]
    assert records[0]["hashes"] == records[1]["hashes"]
    print("PASS: analytic b/V/FE/residuals, identical audit on/off outputs, receipts only when requested")


if __name__ == "__main__":
    main()
