# AKM/KSS and the leave-out connected sample

Package documentation version: 2.35.0.20261125.

`xhdfe.akm.leave_out_set()` corresponds to Stata's `xhdfeconnected` and R's
`xhdfe_akm_leave_out_set()`. `xhdfe.akm.akm_kss()` corresponds to
`xhdfeakm` and `xhdfe_akm_kss()`. For the worker-projection assumptions and
diagnostics, open `python -m xhdfe akm-inference`.

## Connected sample

`leave_out_set(worker, firm, fweights=None, *, num_threads=0, gpu=False,
verbose=0)` returns a `keep` mask over the input rows. It finds the largest
leave-one-worker-out connected set. The result includes the number of
workers, firms, matches, movers and stayers, plus thread and backend
diagnostics. With frequency weights, `n_obs_input`, `n_obs_connected` and
`n_obs` count weighted person-years. The mask still indexes the supplied
rows.

`gpu=True` requests CUDA sorting of match keys for eligible large inputs.
Graph pruning runs on the CPU. Check `gpu_used` and `gpu_status_code`.

## Decomposition and inference

`akm_kss(y, worker, firm, X=None, **options)` fits the two-way model.
Supply one outcome and two identifiers per row. `X` contains numeric
controls partialled out before the two-way fit. Their FWL step uses the
certified CPU reference absorber. The two-way solver has separate CUDA
diagnostics.

| Options | Meaning and defaults |
| --- | --- |
| `leave_out_level`, `prune` | `"match"` (default) or `"obs"`. The connected-set pruning is on unless `prune=False`. |
| `leverages`, `exact_max_rows`, `jla_draws`, `seed` | `"auto"` uses exact leverages at up to 10,000 input person-years before pruning, then JLA. With frequency weights, the threshold uses their sum. `"exact"` and `"jla"` select the method directly. JLA uses 200 draws and seed 20260705 by default. |
| `fweights` | Positive integer replication counts. Supported for match-level point decomposition. |
| `compute_se`, `eigen_diagnostics` | Both default to `False`. Under the default centring, the former computes SEs and the latter adds weak-identification diagnostics and Andrews–Mikusheva intervals. `eigen_diagnostics=True` implies computing SEs. At match level, worker-variance inference is unavailable. |
| `sigma_centering` | `"leaveouttwoway"` (default) or `"person_year"`. The latter makes point components invariant to a common outcome shift. Its optional worker-projection inference covers firm variance and worker–firm covariance with 2–128 firms and unit observation weights. |
| `se_nsim`, `eig_trace_nsim`, `se_sigma_lowess`, `Z` | Legacy inference controls. Defaults are 1,000 quadratic simulations, 100 trace draws and the binned variance fit. `se_sigma_lowess=True` selects the lowess fit. `Z` is an optional design for legacy linear combinations. |
| `direct_max_firms`, `direct_max_nnz`, `cg_tol`, `cg_max_iter` | Two-way solver controls. Defaults are 50,000 firms, 40,000,000 nonzeros, tolerance 1e-10 and automatic CG iteration limit. |
| `fwl_tol`, `fwl_max_iter` | Controls-step tolerance 1e-10 and maximum 100,000 iterations. |
| `num_threads`, `gpu`, `verbose` | Automatic thread selection at 0. A positive request is bounded by available capacity. `gpu=True` requests CUDA for the two-way solve. `verbose=1` reports phase progress. |

The result contains `plugin`, `agsu` and `kss` variance components, a
connected-set mask and solver diagnostics. Check `converged` and `notes`.
Legacy SEs accompany the corresponding `theta_*` estimates in
`component_se`. The alternative inference is in `projection_inference`.

## Public example

Open the AKM/KSS block from `python -m xhdfe examples`.
