#!/usr/bin/env python3
import argparse
import importlib.util
import math
import os
from pathlib import Path
import struct
import sys


os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XHDFE_GPU_BACKEND": "cpu",
})

import numpy as np


def bits(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def load_module(module_dir):
    candidates = sorted(Path(module_dir).glob("py_hdfe_v11*.so"))
    if len(candidates) != 1:
        raise AssertionError(f"expected one py_hdfe_v11 module: {candidates}")
    sys.modules.pop("py_hdfe_v11", None)
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_at_t(module, target_t):
    df = 100000
    n = df + 2
    x = np.zeros(n, dtype=np.float64)
    x[0] = 1.0 / math.sqrt(2.0)
    x[1] = -x[0]
    residual = np.zeros(n, dtype=np.float64)
    residual[-2] = math.sqrt(df / 2.0)
    residual[-1] = -residual[-2]
    y = target_t * x + residual
    reg = module.HdfeRegressor(
        se_type="unadjusted", fit_intercept=True, num_threads=1,
        drop_singletons=False, tol=1e-12, max_iter=1000,
    )
    reg.fit(y, np.asfortranarray(x[:, None]), fes=[])
    if float(reg.df_resid_) != float(df):
        raise AssertionError(f"public df_r differs: {reg.df_resid_}")
    observed_t = float(reg.tvalues_[0])
    if bits(observed_t) != bits(target_t):
        raise AssertionError(
            f"public t calibration differs: {observed_t!r} != {target_t!r}"
        )
    return int(bits(reg.pvalues_[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", required=True)
    args = parser.parse_args()
    module = load_module(args.module_dir)
    fixtures = (
        (37.794586197174261, 0x2874C6C65AC3),
        (38.628450615292988, 0x1),
    )
    observed = []
    for target_t, expected_bits in fixtures:
        actual_bits = fit_at_t(module, target_t)
        if actual_bits != expected_bits:
            raise AssertionError(
                f"public p bits differ at t={target_t!r}: "
                f"0x{actual_bits:x} != 0x{expected_bits:x}"
            )
        observed.append(f"0x{actual_bits:x}")
    print(
        "PUBLIC_INFERENCE_SUBNORMAL_PASS "
        f"module={Path(module.__file__).resolve()} bits={','.join(observed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
