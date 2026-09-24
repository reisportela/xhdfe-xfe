# Validation scope for 2.28.0.20260924

This record distinguishes estimator validation from build, loader and
distribution validation. The precision reference is published 2.26.2;
Fast and Comparable retain their distinct stopping and approximation contracts.

## Estimator and frontend validation

The local Linux/H100 campaign completed 24 specifications across eight cells:
C++/Python and Stata, CPU and CUDA, Fast and Comparable. It recorded 266 paired
runs (532 legs), with frozen reference/candidate binaries and source hashes.
Of 192 candidate cells, 188 converged. Every CPU cell and every Fast cell
converged. Four CUDA Comparable cells still refuse estimation: the directors
and difficult 10-million-row three-FE specifications, through both frontends.
Use CPU for those specifications.

Full covariance and coefficient comparisons yielded 256 direct passes, three
cases requiring independent adjudication and seven pairs without comparable
estimates. The three adjudications were:

- Two saturated Stata CPU designs had zero residual degrees of freedom.
  Independent exact oracles validated identified coefficients; their variance
  was unidentified. These were covariance-grading errors, not a waiver for
  small positive variances.
- One workers CUDA Comparable reference covariance failed the independent
  oracle (scaled error 1.44e-8); the candidate passed (4.06e-10).

The seven pairs without comparable estimates comprise the four refusals above
and three old Fast AKM failures. Four candidate Fast AKM cells converged and
were additionally checked against valid CPU references, without redefining
Fast's contract. Failed reference fits do not support performance claims.

Stata estimation masks were checked row by row: 130 matched pairs, four pairs
without returned estimates, eight workers masks matching independent singleton
pruning, and four Fast AKM masks matching valid references. Separate savefe
validation completed 19 pairs, including full FE comparisons and reconstruction
checks. CPU C++ FE totals were bit-identical; reconstruction error was at most
1.78e-15. This is separate recovery coverage, not a claim that every core cell
was tested with every post-estimation option.

Focused batteries also cover IV conditioning and instrument spans, covariance,
zero weights and singletons, exact cache identity, FE units and extraction,
Python formula/refit behaviour, R formula/offset/prediction metadata, and Stata
sample and generated-variable custody. Permanent reproducers are under
`tests/analytic_precision/` and `tests/stata/`.

## Performance trade-offs

No tolerance was relaxed or tightened to improve benchmark results. The
accepted local costs are reported alongside gains; subsecond cases and host
noise require particular care.

| Comparison | Observed candidate change |
| --- | ---: |
| Large ready C++ CPU Comparable, six interleaved pairs | +0.77% median |
| AKM1 C++ CUDA Fast, six interleaved pairs | +9.63%, about 0.99 s |
| AKM2 C++ CPU Comparable, valid balanced comparison | -40.77% |
| Workers CUDA Comparable against prior valid reference | -17.8% to -22.3% C++; -28.6% to -32.5% Stata |
| Savefe aggregate medians | +0.5% to +2.2% |
| Sparse 32k-row CUDA control with an absorbed regressor | +3 to +5 ms, +27% to +37% |

Other measured local costs include Marta CPU +8.7% to +9.1% (about 0.32–0.45 s)
and difficult-panel CUDA Comparable +11.6% to +13.9% (about 0.23–0.27 s).
The final large Stata CPU smoke used 16 actual workers: 76.373 s versus
81.351 s for its reference, identical sample, coefficient error 3.47e-18 and
scaled covariance error 9.40e-15. That single smoke is not a general speedup
estimate.

## Remaining limits

Besides the four matrix refusals, separate tests retain three extreme-origin
ordinary CUDA refusals and 18 inherited group/individual CUDA refusals. CPU is
the available route for those cases. A successful GPU-use diagnostic is not a
universal certificate for inference, residuals, recovered effects or
decompositions. There is no claim of exhaustive correctness on all possible
inputs or equal speed on every specification.

## Release artifact gates

Publication requires the exact version-tag assets, not just the local binaries:
Linux CPU/OpenMP, Windows installed-layout loading and two real estimator
workers, both native macOS slices, Python/R package checks, and exact CUDA
plugin execution on the H100. The Windows gate includes a dynamic-DLL negative
control reproducing Stata's letter-directory loader problem (public issue #11).
Static plugin provenance authenticates the GNU link inputs and final hashes.

The workflow stages a draft first. Its assets and net-install snapshot are
published only after these gates pass; see the
[release procedure](../release-workflow.md). The
[preflight native jobs](https://github.com/reisportela/xhdfe-xfe/actions/runs/36022535095)
passed on all platforms; its initial assembly failed on packaging metadata. The
[separate assembly retry](https://github.com/reisportela/xhdfe-xfe/actions/runs/36029203100)
validates the corrected packaging against authenticated native inputs, without
claiming that the original run passed as a whole. Native receipts and runtime
provenance are included in the offline release bundle; the
[versioned release](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.28.0.20260924)
identifies the final assets and checksums.
