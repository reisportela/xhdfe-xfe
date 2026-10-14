# Weak-instrument 2SLS under Driscoll–Kraay with first-stage diagnostics

Offline supplemental contract (13oct2026). It imports the 11oct2026 oracle
(`../dkraay_20261011/independent_oracle.py`, unchanged) for the structural
2SLS/DK reference and adds an explicit first-stage oracle: the DK Wald
statistic of the excluded instrument `b_z^2 / V_zz`, its residual degrees of
freedom, its p-value (regularised incomplete beta) and the partial R², graded
against `e(first)` from `ffirst`. Loadings 0.05, 0.15 and 0.9 on the excluded
instrument give first-stage F of about 3.6–5.3, 12–16 and 220–270; the
structural allowance is predeclared from the 2SLS score Gram condition number
(`|Δb| ≤ 1e-9 + 64·u·κ₂·‖b‖`, scaled V ≤ 1e-8 + 64·u·κ₂²).

```sh
OPENBLAS_NUM_THREADS=1 python3 tests/analytic_precision/supplemental/weak_iv_dkraay_20261013/weak_iv_dk_oracle.py <new output dir>
OPENBLAS_NUM_THREADS=1 python3 tests/analytic_precision/supplemental/weak_iv_dkraay_20261013/weak_iv_dk_oracle.py <new output dir> --backend cuda
```

Require `complete: true` and every case PASS in `RESULTS.json` (exit 0). On
CUDA the three no-FE cases are expected refusals (`gpubackend(cuda)` requires
an absorbed dimension and fails closed); the FE cases must report
`gpu_used = 1` for the structural fit and the first stage. The runner uses a
fresh Stata process, private ado directories and synthetic data only.
