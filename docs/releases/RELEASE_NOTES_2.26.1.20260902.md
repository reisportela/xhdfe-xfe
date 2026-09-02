# xhdfe 2.26.1 / xfepout 1.13.0 - 02sep2026

Precision and cache-contract hardening for ordinary and grouped absorption,
heterogeneous slopes, inference, and the `xfepout` partial-out frontend. The estimator
definition, public tolerance, supported coefficient/inference outputs, and
default formatting remain unchanged. Failed precision or backend checks remain
fail-closed; this release does not introduce a silent CPU fallback for an
explicit CUDA request.

## Windows Student-t inverse portability fix

The safeguarded normal and Student-t inverse solvers now terminate cleanly
when a valid bracket reaches the effective arithmetic resolution of the
platform. Final bracket and relative-tail postconditions remain authoritative;
no inference tolerance or estimator definition changed. This fixes the
CPython 3.12 wheel on Strawberry MinGW GCC 13.2, including a perfect-fit smoke
with 28 residual degrees of freedom. The tagged `2.26.0.20260902` workflow
stopped at this Windows gate and produced no public release; 2.26.1 supersedes
that unissued candidate.

## Stata Viewer SMCL safety

The active Stata help files are rewrapped without changing paragraph text so
every physical source line is at most 160 bytes and every line balances its
SMCL braces. This prevents the Stata 19.5 GUI Viewer's 245-character source-line
truncation from cutting the `xfepout` tolerance paragraph or the `xhdfe`
net-install URL. The tagged release workflow now enforces both conditions via
`tests/release_lint.sh`; translator output is not used as a substitute for the
Viewer check.

## Ordinary Krylov parity floor

Standard high-level ordinary-FE LSMR/MLSMR calls in reghdfe-comparable or
strict-residual mode now use an internal tolerance no looser than `1e-11` when
the caller requests a weaker public tolerance. The public requested tolerance
continues to govern the reported contract; the tighter internal solve prevents
the AKM1-style plateau that could otherwise move residuals while satisfying a
weak global stopping statistic.

The floor is deliberately narrow. It applies only to ordinary no-slope
LSMR/MLSMR dispatched through the standard fit/partial-out APIs. MAP/GS/SGS,
Jacobi, PCG, heterogeneous slopes, group/individual, CUDA, internal FE recovery,
direct AKM/KSS and Gelbach consumers retain their existing contracts. Additive
diagnostics report the internal Krylov tolerance, maximum final backward error,
maximum running condition estimate, and their product. Non-Krylov paths report
zero for these fields.

## Heterogeneous-slope certificate and continuation

Ordinary heterogeneous slopes retain the scale-invariant per-block and
per-moment certificate introduced during this release cycle. If an Auto GS/SGS
primary candidate fails that honest certificate, the same method may continue
at tighter internal tolerances, always from the original outcome/design and
always judged against the original public tolerance. Iterations are cumulative;
exhaustion is fail-closed. Explicit CUDA never falls back to CPU and Krylov is
not used for slopes.

## Bounded ordinary Auto retry

For unweighted ordinary CPU fits at the default comparable `1e-8` contract,
an Auto-selected GS/SGS result whose independent all-RHS absorption certificate
exceeds `1e-11` now receives one cold explicit MLSMR solve from the original
outcome, regressors, and fixed effects. The retry is returned only when it
converges and passes the unchanged public certificate. A failed retry retains
the already valid primary result and reports the attempt explicitly.

The policy is deliberately bounded to the Auto selector's supported promotion
band. It excludes weights, heterogeneous slopes, `group()`/`individual()`, FE
recovery, `xfepout`, IV, custom tolerances, explicit methods, cache hits,
nested OpenMP regions, and all CUDA requests. Additive diagnostics expose the
primary and retry methods, iterations, certificate values, status, and retry
time. `XHDFE_AUTO_ROUTING_RETRY=0` is an audit/A-B kill switch.

## Group/individual LSMR precision

CPU and CUDA joint LSMR for `group()`/`individual()` now honor the existing
grouped comparable internal floor of `1e-12`; the public requested tolerance
and estimator definition are unchanged. This closes the literal individual-
incidence normal equations on difficult sum-aggregation panels. Any cold
certificate retry uses the grouped floor rather than the looser ordinary
tolerance.

## Student-t extreme tails

Large-degrees-of-freedom two-sided Student-t probabilities now fall back from
an uncertified Cornish-Fisher inverse to the existing exact incomplete-beta
path. The transition is explicit rather than exception-wide: invalid inputs
and invalid probability results remain fail-closed. Representable subnormal
probabilities keep their prior bits; tails below binary64 range return positive
zero instead of aborting otherwise valid large-sample fits.

## Absorption cache generation 4

Absorption caches now use `xhdfe_absorption_cache_v4`. Their signature includes
the solve-contract generation, the ordinary-Krylov parity-floor state, exact
outcome/regressor/FE/weight inputs, intercept and singleton policy, effective
threads, backend identity, public tolerance, method and solver-selection knobs.
The signature also binds the bounded Auto-retry policy and kill-switch state;
the payload includes both Krylov and retry diagnostics plus a domain-separated
two-word digest.

Readers validate magic, signature, dimensions, method/iteration ranges,
finite diagnostics and transformed values, vector lengths, digest, truncation,
and exact EOF. Generation-2 files, signature mismatches, corrupt/truncated
payloads and trailing bytes are safe misses. `auto` repairs a miss only after a
new result converges and passes the independent precision certificate;
`read` never rewrites. This explicit generation input invalidates generation-3-to-4
absorption-cache reuse while preserving new-to-new hits. FE-structure and
grouped-mobility signatures are unchanged because their solve contracts did
not change.

## xfepout 1.13.0 cache/profile completion

`xfepout` now applies the same generation-4 absorption-cache and mobility-profile
semantics in `HdfeRegressorV11::partial_out()` that the standard estimator uses.
Read occurs before absorption; write occurs only after convergence and precision
certification. FE-structure and absorption caches compose through a canonical
FE signature, independent of raw-label versus cached-group encoding.

Documented aliases now parse consistently: `abscachefile()`,
`absorptioncachemode()`, `mobilityfile()`, `mobilityprofilefile()`, and the
FE-cache aliases. Duplicate canonical/alias spellings are rejected. Pathless
`abscachemode(read|write|auto)` fails with `r(198)` unless a mobility-profile
path supplies the companion-cache authority; pathless `off` remains valid.
CPU-only Krylov/sparse profile hints are ignored under explicit CUDA, and CPU
cache material never suppresses a real CUDA attempt.

## Validation and honest limits

The exact final Linux CPU comparable module
`ca95b512fa79519ecd3e3a5fc77f433cf4ce125ff9e7fcc057b7bf98adcc4363`
passed all 24 core24-quick surfaces: 21 `PASS_STRICT`, PF3 and
`synthetic-assortative` under their pre-reviewed target/numerical-equivalence
contracts, and saturated `synthetic-zigzag` under its structural contract. The
control-to-candidate class delta was exactly the nine pre-registered fixes;
the other fifteen classifications were invariant.

The nine changed CPU targets are `credit2`, `directors`, `enron`, `github`,
`patents`, `schools`, `synthetic-assortative`,
`synthetic-uniform-harder`, and `marta_group_individual_3m`. The first eight
changed because the bounded Auto retry replaced an otherwise accepted but
insufficient GS/SGS result with a certified MLSMR result; Marta changed because
the grouped comparable LSMR floor now reaches its intended `1e-12` value.

The exact final Linux H100 `sm_90` CUDA comparable module
`d9aaa908190fcaf5130e6163b1ac5b0d6e9a39a4c118f3a72b07318d2d96917d`
accepted 13/24 surfaces (12 strict plus the saturated structural case), with
three real-H100 attempts on each accepted surface. In particular, the full
Marta `group()`/`individual(), aggregation(sum)` surface passed strictly.
Eleven difficult ordinary FE graphs did **not** meet the complete numerical
contract and remain unambiguously un-certified:

- `akm_v02_firstreg`, `directors`, `enron`, `github`,
  `main_95_21_ready`, `patents`;
- `pf_difficult_10m_3fe`, `schools`, `synthetic-assortative`,
  `synthetic-uniform-harder`, and `workers`.

For AKM1, the public CUDA result had maximum residual difference about
`1.51e-2`; the existing internal-`1e-10` diagnostic reduced it to about
`1.51e-3` but still failed the unchanged authority. A CPU MLSMR continuation
passed that authority but cost about 111 seconds including the GPU stage,
versus about 86.5 seconds for direct CPU MLSMR, so it was rejected as a
default GPU-to-CPU hybrid. No generic residual waiver, post-hoc tolerance, or
silent CPU fallback was introduced. Use CPU for any fully certified result on
these hard ordinary graphs: some failures include coefficient or standard-error
gates as well as residual/fixed-effect gates. CUDA binaries ship as optional test/performance assets,
not as a claim of universal numerical certification. A/B against the pre-retry
CUDA bytes passed for AKM1 and `directors`; no broader non-regression claim is
made for the other nine failed cells.

Separate final-byte gates passed for Stata CPU and real-H100 CUDA, Python wheel
and source distribution, R CPU/CUDA tests, generation-4 cache/profile behavior,
OpenMP linkage, source-archive closure, and unified version/date metadata. The
R source check completed with the one CUDA-source-filename warning and one
stdout/stderr note documented by the package; unavailable suggested packages
were not forced. Windows and macOS results are compile/link/package evidence
from the tagged CI workflow, not local runtime execution.

## Version scope

- Shared C++/Python/R package and release tag: `2.26.1.20260902`.
- Stata `xhdfe`, `xhdfe_p`, `xhdfe_estat`, and `xhdfegpu`: `2.26.1`.
- Stata `xfepout`: `1.13.0`.
- Companion feature versions remain unchanged.
- All production Stata text files carry `02sep2026`.

## Attribution

This release preserves the explicit credit to Alexander Fischer and Kristof
Schröder for the graph-preconditioned MLSMR/additive-Schwarz architecture and
to Marta Silva for the sustained functional, numerical-precision, and
`reghdfe`-comparison testing that led to the precision and certification work.

The previous public `v2.25.1.20260827` release and its assets remain immutable rollback
material. The tagged workflow builds the public multiplatform artifacts from
the exact release commit; publication follows read-back and the explicit H100
hardware gate for those CI-produced CUDA bytes.
