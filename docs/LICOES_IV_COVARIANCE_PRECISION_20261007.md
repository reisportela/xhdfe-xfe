# IV covariance precision — 7 October 2026

Classification: **demonstrated estimator error**, present before the current
proportional-instrument repair. Candidate 2.29.3.20261007 repairs it.

## Reproducer and mechanism

`iv_intercept_covariance_contract.py` supplies known Walsh IV scores, a
recovered absorbed constant, delta=2^-27, and equivalent two-way cluster
partitions. The unweighted and frequency-weighted zero-mean cases have a
positive covariance, but slope variances around 1e13–1e14 coexist with an
intercept variance around 0.002. Rounding the wide sandwich to binary64 before
standardization and PSD correction creates a spurious negative eigenvalue.
Converting the rounded matrix back to wide arithmetic cannot restore its low
words. The old error affected both CPU and CUDA, in both tolerance modes.

Genuine-indefinite controls revealed an earlier cancellation too: evaluating
bread × meat × bread in a nearly collinear basis can lose the small covariance
eigenvalues even before that final conversion. Passing an independently
constructed covariance directly through the new PSD helper gives errors below
1.5e-16, isolating this second loss to sandwich construction. The initial
precision-retention patch passed the original 46 cases but still failed three
of the 44 new controls. These results and source/binary snapshots are retained.

## Repair and permanent guard

On the existing wide multiway IV path with a recovered intercept, solve for
each cluster's influence using the existing factorization, then sum signed
outer products under the same inclusion/exclusion and SSC rules. Keep the
augmented covariance wide through final scaling and both PSD corrections;
convert to binary64 for the public output. No arbitrary-precision oracle enters
the estimator. The existing OlsWide type, rank decisions and tolerances remain.

The original MP90/MP120 contract is unchanged. The new
`iv_covariance_psd_contract.py` adds 44 cases per backend/mode, including
positive controls and fixed, genuinely indefinite partitions. Its Stata
companion checks the public PSD flag and every covariance entry. Final results:
184/184 original cases, 176/176 new native cases, and 176/176 Stata covariance
fits pass, with real CUDA use checked. First-stage/reference, proportionality,
repeated-control, conditioning, saved-FE and cache guards also pass.

## Harness lessons

- Requiring every shifted negative case to detect a disabled projection was a
  harness error: a real correction can be below the fixed scaled-error budget
  when raw variances are huge. The final battery retains every case and its
  original numerical threshold, requires at least 12 detectable negative
  controls overall, and separately checks every Stata PSD flag. No estimator
  failure or tolerance was removed.
- An early Stata launch loaded the initial plugin three seconds before the
  final link completed. Its recorded hash differs from the final artifact;
  it is invalid final-candidate evidence and was rerun after build completion.
- Stata batch mode chose the last path argument as its log name. A wrapper
  do-file makes log collection deterministic. Two repeated-control invocations
  also used the wrong test directory and were rerun at the correct path.
- Preserve failed runs and distinguish harness failures from numerical ones.

Performance depends on the number of cluster intersections. The local CPU
cost is about 1–2% in the equivalent-cluster reproducer but 8–15% on affected
weak-IV stress cases with distinct partitions. Testing only equivalent labels
would have understated the cost. Keep the correction for review and optimize
that narrow path before a PR. See the
[report](certification/iv-covariance-precision-20261007.md) for measurements and
remaining CUDA/full-release performance gates. Do not present focused tests
as global performance certification or overwrite the separate Athena campaign.

## Authorized concurrent GPU follow-up

A subsequent explicit user request authorized timing while Athena's separate
campaign remained active. All 28 prescribed cells and 9,658 timed fits used
CUDA, with before/after binary custody and per-pair load snapshots. The
distinct-cluster point estimates were approximately 8–13% slower, consistent
with the CPU finding. Wide intervals in some cells and noisy ordinary controls
prevent a precise clean-host conclusion. Keep concurrent measurements labeled
as such, preserve every observation, and retain the optimization-before-PR
recommendation and the remaining full-release gates.
