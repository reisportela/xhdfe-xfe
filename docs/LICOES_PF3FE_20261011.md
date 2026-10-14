# PF3FE — 11 October 2026

Status: corrected and validated locally in the private 2.29.0.20261006
candidate. Global performance certification and release remain pending.

## Demonstrated estimator error

The primary CUDA Comparable transform on the ten-million-row PF fixture
fails the necessary original-design normal equation. Extending the old
continuation alone returns an inaccurate covariance. The independently
qualified offline Schur oracle uses two validated gauges; the CPU fit agrees
with it. The original inputs and oracle thresholds remain unchanged.

An observation-space CG step test and the exact nested-cluster intercept
identity pass the original clustering but fail a non-nested cluster control.
Stable evaluation of I-G and a small coarse space based on the third FE also
fail. These experiments remain recorded and are not accepted solutions.

The slow directions belong mainly to the large two-FE incidence graph. The
correction reuses the production graph preconditioner in CGLS on the FE
coefficients. CUDA performs D, D', residual updates and row reductions; the
host builds/applies the FE-space preconditioner using its production packed
parallel implementation. This is a mixed execution
algorithm with actual CUDA row work, with no hidden CPU fit. Only the single
N1 correction for unweighted three-FE GS/SGS Comparable fits without retained
FEs or heterogeneous slopes enters it. The primary route and Fast contract
are preserved. The correction reconstructs from the original inputs because
orthogonal drift in a failed primary transform cannot be removed by applying
another projection to that transform.

The public tolerance and remaining iteration budget are unchanged. The
stopping quantity uses the actual projected step norm, and the existing
necessary FE/normal checks still apply. On this short correction, a failed
necessary check is retried at the next eligible step, without an arbitrary
64-step delay. No independent oracle, covariance bound or tolerance ladder
is used inside an estimate.

For a single cluster partition containing complete FE levels, cluster
residual sums are zero. The constant covariance follows the exact identity
`c=-B*xbar`, `a=xbar'B*xbar`, preserving B. Nesting is proved from the actual
partitions, including when cluster DoF adjustment is disabled. Non-nested
controls explicitly reject this formula.

Reproducer and evidence:
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/pf3fe_20261011/`.
Permanent test: `tests/analytic_precision/pf3fe_covariance_contract.py`.
The definitive artifacts are `parallel_cpu/`, `parallel_cuda/` and
`parallel_stata/` in the evidence directory. Earlier diagnostic and `final_*`
builds are preserved as historical evidence.

## Thread-phase coverage and harness lesson

The first graph wrapper parallelised only over the eleven RHS. Its aggregate
thread counter could report a team of 48 while at most eleven workers applied
the preconditioner. This was missing coverage. Reusing `prepare_lanes`,
`prepare_workspace` and `apply_lanes` exposes 48/96 independent work units in
the level passes. The triangular solve reports its eleven tasks separately.
The permanent test's `--phase-proof` now checks actual teams, available work
units and useful workers. Dynamic scheduling need not assign work to every
thread on every pass: requiring exactly 48 active workers was a harness error,
not evidence of a software cap. Actual counts remain in the receipts.

## Final evidence

- PF3FE native: 17 independent numerical passes and one expected clean refusal.
- Stata: ten independent numerical passes, plus clean failed-refit/recovery.
- Explicit 1/2/8/16/24/48 requests pass in native and Stata CUDA, with phase evidence.
- The permanent 984-row campaign retains 619 xhdfe passes and 223 expected
  refusals; all verdicts match the previous campaign. The 29 inherited
  reference-comparator failures remain visible.
- Protected IV controls pass; the required large CPU plugin smoke takes
  76.461 seconds, inside the preceding 74.792–76.556 range, with identical
  coefficients, covariance, residual/sample signatures and iterations.

The large smoke is a local build check, not a core24 × 8 performance verdict.
Windows/macOS are not run; R and resource/release gates remain separate.
No new private packet was sent externally, and no installation or publication
was performed. The correction manifest and inverse patch preserve the earlier
dirty candidate and permit a selective rollback.
