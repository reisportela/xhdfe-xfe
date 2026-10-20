"""CG cold restart must share the original iteration budget on CPU and CUDA."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

import numpy as np

from cg_projection_full_covariance_contract import fixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module', type=Path, required=True)
    p.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    os.environ.update(XHDFE_GPU_BACKEND=args.backend,
                      XHDFE_ABSORPTION_CACHE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, X, fes, _, _ = fixture('aggregate', False)
    rows = []
    for maximum in (1, 2, 4, 1000):
        reg = core.HdfeRegressor(num_threads=1, max_iter=maximum,
                                fit_intercept=False, drop_singletons=False,
                                convergence='reghdfe')
        row = {'max_iter': maximum}
        try:
            reg.fit(y, X, fes=fes)
            row.update(returned=True, iterations=int(reg.num_iterations_),
                       converged=bool(reg.converged_),
                       certified=bool(reg.precision_certified_),
                       gpu_used=bool(reg.gpu_used_))
            row['status'] = ('PASS' if maximum == 1000 and reg.converged_
                             and reg.precision_certified_
                             and 0 <= reg.num_iterations_ <= maximum
                             and bool(reg.gpu_used_) == (args.backend == 'cuda')
                             else 'FAIL')
        except RuntimeError as error:
            message = str(error)
            match = re.search(r'iterations=(\d+)/(\d+)', message)
            stale = any(np.asarray(getattr(reg, name)).size
                        for name in ('coef_', 'covariance_', 'residuals_'))
            row.update(returned=False, error=message, stale_estimates=stale)
            row['status'] = ('PASS' if maximum < 1000 and match is not None
                             and int(match[2]) == maximum
                             and 0 <= int(match[1]) <= maximum
                             and not stale and not reg.converged_
                             and not reg.precision_certified_ else 'FAIL')
        rows.append(row)
    result = dict(module=str(args.module.resolve()),
                  module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  backend=args.backend, rows=rows,
                  status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL')
    if args.output is not None:
        with args.output.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
    print(json.dumps(result))
    return result['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
