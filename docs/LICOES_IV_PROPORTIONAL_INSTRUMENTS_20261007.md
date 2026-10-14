# IV proportional instruments: lessons from 7 October 2026

The reproduced CUDA refusal was caused by checking transformed inputs as though
they preserved exact raw identities. CUDA absorption perturbed `zdup=2*z1` by
8.88e-16; the wide rank check interpreted that perturbation as identifying
information, and a later FWL check rejected the result. Classification:
**undue verifier refusal**. Tightening absorption or loosening all rank
thresholds would affect genuinely distinct small directions.

Candidate 2.29.2 instead removes pairwise exactly proportional raw instruments
before absorption, on the retained sample. A bounded binary64 cross-product
comparison uses portable integer arithmetic and performs no in-fit
arbitrary-precision verification. All other dependencies retain their existing
handling. The permanent rational oracle checks extreme scales and one-bit
changes; native and Stata CPU/CUDA tests check coefficients, full covariance,
F statistics, saved FEs, true underidentification and actual GPU use.

A new fixture initially combined multiple endogenous Walsh directions with
weights constant over an incomplete cycle, invalidating its covariance oracle.
Classification: **evaluation-harness error**. The corrected fixture explicitly
asserts its score/FE/residual orthogonality assumptions. The failed evidence is
preserved; neither the estimator nor acceptance tolerances changed.

Broader tests also reproduce two existing multiway-covariance edge-case errors
on CPU and CUDA. Classification: **demonstrated estimator error** relative to
the independent oracle. Across all 184 covariance cases, the frozen old module
and candidate have identical coefficients and covariance matrices. These
failures remain open and cannot be counted as passes for release certification.

See [the scoped validation report](certification/iv-proportional-instruments-20261007.md)
for results, build provenance, permanent test names and pending release gates.
