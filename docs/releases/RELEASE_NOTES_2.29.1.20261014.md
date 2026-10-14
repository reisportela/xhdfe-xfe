# xhdfe 2.29.1.20261014

This corrective release addresses the 14 October adversarial audit.

- IV Driscoll–Kraay covariance retains time-cluster influences from the existing
  wide factorisation, including the recovered constant. This avoids losing the
  well-identified direction when projected regressors are nearly collinear and
  have common nonzero means. Absorption, point estimates and tolerance modes
  retain their existing contracts.
- First-stage regressions inherit absorbed terms without FE output aliases.
  `first`, `ffirst` and `estat firststage` can therefore accompany named saved
  effects without recreating or overwriting the structural equation's variables.
- The analytic validation covers the identified constant and
  complete covariance, with negative controls and explicit CUDA/Fast coverage.
- Native platform gates exercise the DK IV counterexample in both tolerance
  modes with one and two threads. Its covariance reference is independently
  calculated at 90 and 120 decimal digits; the previous plugin fails the gate.
- MinGW wide arithmetic computes the product roundoff independently of the
  Windows runtime's inaccurate software FMA. Clang wide arithmetic preserves
  compensation under the release compiler flags. These platform corrections
  restore the intended coefficient and covariance accuracy without changing
  solver targets or tolerance-mode semantics.

See [validation and release coverage](VALIDATION_2.29.1.20261014.md).
Earlier
performance and platform receipts belong to their recorded source/binary hashes;
they are not a new performance campaign for this correction.
