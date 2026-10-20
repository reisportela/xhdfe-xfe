"""CUDA-eligible pandas/Polars formula paths against a finite-sample oracle.

The general frontend suite includes CPU-only no-FE fits and bit-exact CPU
comparisons. This separate contract requires real CUDA execution and applies
numerical limits to complete parameter/covariance/residual outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['XHDFE_GPU_BACKEND'] = 'cuda'
    sys.path.insert(0, str(args.package_root.resolve()))
    import xhdfe
    from xhdfe import py_hdfe_v11 as core
    assert Path(xhdfe.__file__).resolve().is_relative_to(args.package_root.resolve())
    assert Path(core.__file__).resolve().is_relative_to(args.package_root.resolve())
    n = 64*64*32
    row = np.arange(n)
    repeat = row % 32
    firm = row//(32*64)
    year = (row//32) % 64
    x1 = np.where(repeat & 1, 1., -1.)
    x2 = np.where(repeat & 2, 1., -1.)
    noise = np.where(repeat & 4, .125, -.125)
    y = 1.25*x1-.5*x2+3+.25*firm-.125*year+noise
    frame = pd.DataFrame(dict(y=y, x1=x1, x2=x2, firm=firm, year=year))
    frame['firm_text'] = 'f'+frame.firm.astype(str)
    frame['year_text'] = 't'+frame.year.astype(str)
    frame['weight'] = 1 + firm % 3
    # Complete bipartite FE design: rank_D=64+64-1; two independent slopes.
    expected_df = n-127-2
    expected_V = np.eye(3)*.015625/expected_df
    rows = []
    for weighted in (False, True):
        weights = frame.weight.to_numpy() if weighted else np.ones(n)
        constant = float(np.average(3+.25*firm-.125*year, weights=weights))
        expected_b = np.array([1.25, -.5, constant])
        for mode in ('reghdfe-comparable', 'xhdfe-fast'):
            for labels in ('numeric', 'string'):
                formula = ('y ~ x1 + x2 | firm + year' if labels == 'numeric'
                           else 'y ~ x1 + x2 | firm_text + year_text')
                for container in ('pandas', 'polars'):
                    data = frame if container == 'pandas' else pl.from_pandas(frame)
                    fit = xhdfe.feols(formula, data, weights='weight' if weighted else None,
                                      num_threads=2, tolerance_mode=mode,
                                      se_type='unadjusted', drop_singletons=False)
                    be = float(np.max(abs(np.asarray(fit.coef_)-expected_b)))
                    ve = float(np.max(abs(np.asarray(fit.covariance_)-expected_V))/expected_V[0, 0])
                    re = float(np.max(abs(np.asarray(fit.residuals_)-noise)))
                    passed = (fit.converged_ and fit.precision_certified_ and fit.gpu_used_
                              and int(fit.gpu_status_code_) == 1 and fit.nobs_ == n
                              and fit.df_resid_ == expected_df and be < 1e-9
                              and ve < 1e-8 and re < 1e-7)
                    rows.append(dict(weighted=weighted, mode=mode, labels=labels,
                                     container=container, b_error=be, V_error=ve,
                                     residual_error=re, gpu_used=bool(fit.gpu_used_),
                                     threads_used=int(fit.threads_used_),
                                     status='PASS' if passed else 'FAIL'))
    result = dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
                  package=xhdfe.__file__, module=core.__file__,
                  module_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                  rows=rows)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'status': result['status'], 'cases': len(rows),
                      'failed': [r for r in rows if r['status'] != 'PASS']}))
    return result['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
