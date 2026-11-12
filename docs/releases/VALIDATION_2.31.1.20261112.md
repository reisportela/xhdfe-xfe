# Validation — 2.31.1.20261112

Scope: the explicit AKM/KSS person-year-centred point estimator and optional
full-model worker projection inference, with the legacy default preserved.
The release was published on 12 November 2026 at 06:15:41 UTC. Exact artifact,
public-installation and canonical-runtime gates passed as recorded below.
The numerical and performance results identify their scope; this is not
whole-package certification.

The base is private commit `dc42782af30a7d2725b89e27c8568e9fc01572c7`, the
2.31.0 source line. Numerical comparisons use the authenticated frozen
`cand_2310_lean_r2` CPU/CUDA modules. No solver or tolerance changes were made.
Private evidence is in `Verifications/issue18_codex_20261111/` in the canonical
development repository; `BASE_STATE.json` records the original identities.

## Published artifacts and canonical closure

[Preflight 43153369775](https://github.com/reisportela/xhdfe-xfe/actions/runs/43153369775)
and [version-tag 43158637224](https://github.com/reisportela/xhdfe-xfe/actions/runs/43158637224)
passed. All thirteen version-tag assets match their GitHub digests and
SHA256SUMS. Ten final plugin/platform receipts, two installed-wheel receipts
and four macOS serial negative controls are bound to the exact attached bytes.
The new AKM fixture passed on Windows and both native macOS universal slices;
Windows also passed the Student-t x87 host-regime guard and negative control.

The exact downloaded Linux CPU/CUDA plugins passed licensed Stata tests on
the H100: ordinary Fast/Comparable and partial-out analytic outputs, new
centering/projection inference, response scales 2^+-500, the 1,000-control
reproducer and the eight-million-row CPU smoke. Actual CUDA use was asserted.
The installed Linux wheel passed the full point and projection contracts.
CI additionally passed all 97 formula/frontend checks and rebuilt the source
distribution. R package tests and the actual-worker probe passed, with two
warnings and four notes retained: unusual source-file extensions and the GCC
Eigen P=1 assignment warning in the unchanged OLS path. The warning was not
silently declared a proved false positive.

[Publication 43166835168](https://github.com/reisportela/xhdfe-xfe/actions/runs/43166835168)
published those same assets and the certified net-install snapshot. A fresh
installation from the public URL matched all 49 site files and 37 installed
or ancillary files, then passed ordinary, centering and projection tests.
Both canonical main checkouts were reconciled without discarding existing
local work. All four CPU/CUDA builds passed 21/21 CTests, full new projection
contracts and useful-worker requests 1, 2, 8, 16, 24 and 48. Fresh processes
confirmed the active Python/Stata paths. All 816 common files in the recorded
source/interface/build scope are byte-identical. Both rebuilt CUDA modules
retain twenty legacy cases, including 8,059 floating-point values, bit-for-bit
against the frozen Claude base; their distinct binary hashes are recorded.

Native SPI checks are separate from a licensed Stata host. Final native Mac
ARM/Intel Stata/MP remains NOT_RUN under the documented publication limit;
its packaged OpenMP runtime is byte-identical to 2.30.1. Windows CI likewise
tests native loading and the host precision regime, not licensed Windows
Stata. Linux CPU/H100 licensed Stata execution is verified. No new full core24
performance certificate is claimed.

Final receipts include `FINAL_ASSETS_AUTHENTICATED_2311.json`,
`FINAL_NATIVE_PLATFORM_RECEIPTS_2311.json`,
`EXACT_VERSION_ASSET_ACCEPTANCE_2311.json`, `FINAL_CANONICAL_BUILDS_2311.json`
and `PUBLIC_INSTALL_ACCEPTANCE_2311.json`. The generated local active-checkout
seal follows the final documentation commits; it is not a source-code change.

## Statistical contract

The user's explicit choice is to retain the default and offer the invariant
alternative with its limits. An independent rational calculation shows why
this is a statistical choice: for some supported four-match and six-match
graphs, the linear constraints for a symmetric quadratic estimator to preserve
the covariance target, have zero bias under arbitrary diagonal match-error
covariance, and be location-invariant are inconsistent. The coefficient and
augmented ranks are 10/11 and 21/22 respectively. Combined variance has the
same incompatibility. This statement is restricted to the specified model
and quadratic class, not every possible estimator.

The previous 16-person-year bias counterexample was also reproduced exactly:
the person-year-centred firm-variance bias is
`-1008382635137/56862549744000`. This is not a floating-point error.
These offline calculations are not transplanted into the fit.

The alternative promises point-formula and location correctness, with its
finite-sample bias documented. The new inference uses a full-model contrast
region and worker clusters, with simultaneous projection intervals at all
signal strengths under the explicit asymptotic conditions in
[`akm-projection-inference.md`](../akm-projection-inference.md). Delta SEs
require first-order regularity and a negligible point correction. No universal
finite-sample unbiasedness or finite-sample coverage claim is made. Lowess,
lincom and worker-variance inference remain unavailable in this mode.
Existing default inference retains its documented limitations, including
unpropagated FWL-control uncertainty.

## Final local inference-integration evidence

The final CPU and H100 builds each passed 116 fits and 48 independent
full-design oracle cases, with five expected refusals recorded separately.
Coverage includes pruned/unsorted samples, unequal match lengths and firm
shares, a retained control after an omitted column, exact/JLA, obs/match,
response scales 2^+-500 and a low-noise case. Maximum SE/CI differences were
about 1.29e-16 / 6.63e-14. CUDA point use was asserted; inference CPU is explicit.

The closed-form C++ test passed 48 cases plus one expected fweight/eigen-only
refusal. The complete CPU CTest suite passed 21/21. Final Stata CPU/CUDA and
R bridge contracts passed, including Stata ci-only and two useful inference
workers. The exact local plugin passed the native SPI gate, including twelve
new closed-form AKM cases. These tests are not a substitute for final remote
platform-asset validation.

Final legacy parity was checked against the frozen 2.31.0 CPU/CUDA modules:
all 20 cases per backend are bit-identical, including 8,059 floating-point
values per backend and the associated sample/diagnostic values. The final
eight-million-row Stata CPU smoke retains identical coefficients, full
covariance, sample and ten-iteration convergence with sixteen workers.
Its 1.559 s versus 1.582 s baseline is a build diagnostic, not an independent
performance claim (`final_legacy_parity.json`).

The issue-sized Stata fixture passed with 149,058 observations and 1,000
control columns. Against tight-tolerance reghdfe on the same retained sample,
the continuous-coefficient gap was 6.3283e-15 and the maximum centred-firm-effect
gap was 1.9041e-14. A simultaneous factor-reference change and outcome shift
preserved the new points, SEs and intervals within the declared limits.

Native point timings used three rotated pairs per CPU cell and six per CUDA
cell against the frozen Claude r2 build. Person-year/default-baseline median
ratios were 0.9926/1.0018 (CPU, 200k/2m rows) and 1.0003/1.0181 (CUDA).
The corresponding legacy-route medians were 0.8572/0.7108 and 1.0073/0.9661.
Large timing variation is retained rather than presented as a speedup claim.
Phase measurements located it in unchanged solves; extra corrected-component
work was only a few milliseconds at two million rows. The separate
1,000-control comparison on the same sixteen physical cores gave median
ratios 1.0021 (legacy) and 0.9960 (person-year). These measurements support
scoped non-regression within host resolution, not universal runtime equality
or a new core24 performance certificate.

## Earlier integration evidence

- Actual compiled CPU module: 70 fits, 30 dense full-design oracle cases,
  four expected refusals recorded separately. Maximum SE difference
  1.232e-16; maximum interval-endpoint difference 6.623e-14. Cases cover
  controls, duplicated/omitted columns, stayers, local controls, exact/JLA,
  observation/match and shifts. Point results are identical with and without
  inference. Inference is independent of the JLA point seed.
- Current CPU point contract: 90 fits, 89 checks and three expected refusals.
- The new C++ translation unit also passed 76 independent standalone checks,
  including response scales 2^+-200, shifts of 1e6, permutations, reference
  recoding and the design-only bread fallback. The standalone recipe used
  different Eigen/compiler flags from production and is additional evidence,
  not a substitute for the compiled-package contract.
- The operator prototype passed 48 checks with 1, 2, 8, 16, 24 and 48 observed
  workers. Prototype and implementation-core receipts identify exact binaries.
- Independent synthetic research holdouts: six match-cluster and six
  worker-cluster designs, each with 20,000 replications, passed the
  predeclared projected-coverage checks. Worker projected coverage ranged
  from 0.95205 to 1.000; interval widths are retained. One underlying joint
  region had finite-sample coverage 0.94085 with 160 workers. This limitation
  is retained and does not establish a finite-sample guarantee. The final
  fixed-dimension joint region is at least as large in structurally redundant
  cases as the reduced-dimension regions used in that research pilot.
- Issue-sized synthetic controlled fit: 149,058 rows, 9,255 workers, 20 firms
  and 1,000 control columns; the point and new inference completed with
  certified FWL convergence in 34 iterations and 16 observed workers. Its
  single elapsed time of 34.485 s is diagnostic, not performance acceptance.
- Four actual Claude Code Fable 5.2/max reviews, algebraic counterexamples
  and an independent check of the final covariance/coverage argument are
  retained privately. The implementation-specific proof is public in the
  guide; it is not attributed to a published theorem for another estimator.

Evidence filenames include `candidate_projection_cpu_contract.json`,
`candidate_projection_point_cpu_contract.json`, `projection_core_check.json`,
`projection_worker_holdout_20000.json` and
`issue_sized_projection_cpu_diagnostic.json`. These are local scope claims.

Final artifact and publication results are recorded above. Earlier
point-only results below
are historical evidence for their recorded binaries, not fresh passes for
the integrated inference build.

## Earlier point-only candidate evidence

- Native CPU: 90 synthetic fits, 89 point/identity comparisons and five
  expected input/inference rejections. The rejections are not numerical passes.
- Native H100 CUDA: the same scoped contract passed, with actual GPU use.
  The recorded GPU backend/status fields are `cuda`/`used`.
- Legacy parity: all 11,005 numerical values in each of 20 CPU and 20 CUDA
  cases are bit-identical to their corresponding frozen r2 module. These
  include exact/JLA, observation/match, stayers, factor controls, fweights,
  and the existing SE/AM output paths.
- Independent dense point oracles cover the alternative and the legacy
  convention. Further checks cover outcome shifts, factor reference changes,
  frequency replication, and plugin/leverage preservation.
- Stata CPU and CUDA contracts passed in new processes. Both use the rebuilt
  OpenMP/native/sm_90 plugin. CUDA execution is asserted, not inferred from
  successful completion. The selected convention is checked after mixed calls.
- The R wrapper and a freshly built Rcpp bridge passed the scoped contract
  without installing a package. Existing Rcpp dependencies were read from the
  repository library. This is a bridge test, not an R distribution test.
- The existing AKM validation suite passed 180 checks; the frontend test passed;
  the full CPU CTest run passed 20/20, including the Student-t host-regime tests.
- After the plugin rebuild, an eight-million-row synthetic CPU fit used 16
  workers and converged in ten iterations. Its coefficients and full covariance
  were bit-identical to the frozen plugin. One fit each took 1.582 s (base) and
  1.517 s (candidate); this is a large CPU smoke, not a speedup claim.

Source tests are `tests/analytic_precision/akm_centering_contract.py`,
`tests/analytic_precision/akm_centering_r.R` and
`tests/stata/akm_centering_contract.do`.

## Performance and unexecuted scope

Interleaved native AKM timings, raw records and host conditions are kept in
the private evidence directory. They are scoped synthetic measurements;
they do not replace a core24 campaign or establish package-wide performance.
An initially occupied GPU and utilisation-sampling lag are recorded as timing
coverage/harness issues, not estimator failures. No other user's process was
modified or terminated.

Native Windows/macOS execution of the new alternative, R package checks,
final CI assets, public source synchronisation, net-install readback and
canonical reconciliation passed within the scopes stated above. Licensed
Mac/Windows Stata host coverage is distinct and remains unexecuted here.
Legacy inference location sensitivity is intentionally retained by default;
the invariant alternative requires explicit selection and its stated conditions.

## Harness findings

- The standalone R bridge recipe initially lacked the existing backend-default
  macro; adding the build definition fixed that harness error. Product source
  was not changed in response.
- Stata names batch logs from trailing arguments on this host; the first
  reader looked for the wrong basename. The completed CPU test was recovered
  and accepted without rerunning it.
- Stata `%g` output can omit a leading zero and is not JSON. The large CPU
  baseline was recovered losslessly; the output formatter uses scientific
  notation for coefficient and covariance arrays. The accepted baseline was
  not rerun for formatting.

Old outputs and failed harness attempts remain preserved. Release conclusions
use the final artifact and public-installation receipts, with the statistical
and platform coverage limits above.
