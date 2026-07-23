{smcl}
{* *! version 1.4.0  23jul2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xhdfeakm" "help xhdfeakm"}{...}
{title:Title}

{p2colset 5 22 24 2}{...}
{p2col :{cmd:xhdfegelbach} {hline 2}}Gelbach (2016) conditional decomposition of coefficient movements{p_end}
{p2colreset}{...}

{pstd}
{bf:Version 1.4.0 (23jul2026), distributed with shared package
2.20.0.20260723.} This release adds near-FE and few-cluster diagnostics,
truthful GPU metadata, full-model block coefficients, and joint-covariance
inference for base-coefficient shares.{p_end}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfegelbach} {depvar} {ifin} {weight}{cmd:,}
{opt x1(varlist)}
[{it:options}]

{synoptset 26 tabbed}{...}
{synopthdr}
{synoptline}
{synopt :{opt x1(varlist)}}base covariates whose coefficient movement is decomposed; {bf:required}{p_end}
{synopt :{opt x2groups(spec)}}named covariate groups added in the full model{p_end}
{synopt :{opth fes(varlist)}}fixed-effect dimensions added in the full model{p_end}
{synopt :{opt absorbedtargets(varlist)}}opt-in X1 targets absorbed by {opt fes()}; full coefficients imposed at zero{p_end}
{synopt :{opt focal(varlist)}}subset of {opt x1()} displayed as focal; reporting only{p_end}
{synopt :{opt shares(type)}}signed shares of {cmd:movement}, {cmd:base}, or {cmd:base_fixed}{p_end}
{synopt :{opt sharetol(#)}}absolute denominator threshold for shares; default {cmd:1e-12}{p_end}
{synopt :{opt level(#)}}confidence level used for share intervals; default {cmd:95}{p_end}
{synopt :{opt vce(vcetype)}}{cmd:unadjusted} (default), {cmd:robust}, or {cmd:cluster}{p_end}
{synopt :{opth cluster(varname)}}cluster identifier (with {cmd:vce(cluster)}){p_end}
{synopt :{opt gamma0}}reproduce b1x2's {cmd:gamma0} variant{p_end}
{synopt :{opt cov0}}reproduce b1x2's {cmd:cov0} variant{p_end}
{synopt :{opt tol(#)}}FE absorption tolerance; default {cmd:1e-8}{p_end}
{synopt :{opt threads(#)}}OpenMP threads (0 = library default){p_end}
{synopt :{opt gpu}}request CUDA for the full HDFE absorption{p_end}
{synopt :{opt verbose}}stream deterministic phase progress and elapsed time live{p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}{cmd:aweight}s and {cmd:fweight}s are allowed (b1x2 conventions).{p_end}


{title:Description}

{pstd}
{cmd:xhdfegelbach} implements the Gelbach (2016) decomposition: it accounts for the
movement of the {opt x1()} coefficients between the {it:base} regression
({it:y} on {it:x1}) and the {it:full} regression ({it:y} on {it:x1} plus the
covariate groups in {opt x2groups()} and the fixed effects in {opt fes()}) as a
sum of per-group contributions

{p 8 8 2}{it:delta_g = (X1'X1)^-1 X1'X2g beta_g}{p_end}

{pstd}
with a standard error for each contribution. The contributions sum exactly to
the total coefficient movement ({it:base} minus {it:full}); the residual of
that identity is returned in {cmd:r(identity_gap)}. Numerical semantics follow
Gelbach's {cmd:b1x2} (the reference implementation this command is validated
against), including its {opt gamma0}/{opt cov0} variants and the aweight/
fweight conventions. Fixed-effect groups are absorbed with the xhdfe backend,
so high-dimensional FE groups are practical. Shares the compiled backend with
the Python {cmd:xhdfe.gelbach.decompose} and R {cmd:xhdfe_gelbach} front-ends.

{pstd}
The default is the standard Gelbach estimand and continues to reject every
rank-deficient {opt x1()} or X2 column. {opt absorbedtargets()} activates a
different, constrained estimand for named {opt x1()} columns that belong to
the span of the added FE (for example, a worker-invariant group indicator with
worker FE). Every declared target must be classified by the backend
specifically as collinear with {opt fes()}; its full-model coefficient is
imposed at zero, not estimated. Undeclared omissions and other rank failures
remain errors. The result is labelled {cmd:absorbed_target_allocation}.{p_end}

{pstd}{bf:Interpretation warning.} This is a decomposition of coefficient
movement for two declared specifications, not causal mediation. Causal
interpretation requires a separately justified research design. Adding
post-treatment variables, colliders, or fixed effects that alter the identifying
comparisons can change the estimand or introduce bias. Block names and causal
roles are supplied by the researcher and are not validated by the command.{p_end}


{title:Specification and sample contract}

{pstd}
The decomposition always uses one jointly declared full model. Every
{opt x2groups()} block and every {opt fes()} dimension is added simultaneously;
their order does not create a sequential or path-dependent allocation. Group
names must be valid, unique Stata names. Observed variables may appear in only
one block and may not overlap {opt x1()}. Empty groups, duplicate variables,
cross-block rank dependencies and undeclared omissions fail closed.{p_end}

{pstd}
Stata constructs one common complete-case sample using the outcome, all X1/X2
columns, FE identifiers, cluster identifier and weight. The backend can then
drop recursive FE singletons. Base, full and auxiliary projections use exactly
that retained sample and the same weight inner product. Inspect
{cmd:r(n_obs_input)}, {cmd:r(n_obs)}, {cmd:r(n_obs_effective)} and
{cmd:r(n_singletons_dropped)}. With {cmd:fweight}, the displayed and effective
N is the sum of retained weights; {cmd:r(n_obs)} remains the retained row
count.{p_end}

{pstd}
The implicit intercept belongs to both models. Do not include a manually
generated constant in {opt x1()}. The matrices {cmd:r(delta)}, {cmd:r(se)} and
{cmd:r(total)} nevertheless contain an explicit final {cmd:_cons} row so the
normalization shift is auditable.{p_end}


{title:Displayed output}

{pstd}
The default display is organized as one panel for each coefficient in
{opt x1()}, followed by the intercept. Each {opt x1()} panel first reports the
base-model coefficient, full-model coefficient, total movement
({it:base minus full}), and its standard error. It then reports each declared
covariate and absorbed-FE block with its contribution and standard error. When
{opt fes()} is used, an FE subtotal is shown without adding it again to the
overall total.{p_end}

{pstd}
In absorbed-target mode, the display identifies the constrained targets next
to the estimand. The full-coefficient cell itself reads {cmd:0 (imposed)} so
the restriction remains visible when a row is transcribed without its header.
It must not be described as an estimated within-FE coefficient.{p_end}

{pstd}
With {opt focal()}, only the selected coefficient panels are printed; all
variables in {opt x1()} remain in both the base and full models and all
full-precision rows remain in {cmd:r()}. This separates the empirical focal
coefficient from low-dimensional controls common to both specifications
without changing the estimand. With {opt shares()}, the table adds signed
percentages. Negative shares and totals above 100 percent are deliberately
preserved and never renormalized.{p_end}

{pstd}
The final status panel reports convergence and the maximum residual from the
summation identity. This residual is a consistency check, not an accuracy or
causal-identification certificate. Displayed values use compact significant-
digit formatting; all matrices in {cmd:r()} retain their full double-precision
values. Solver and inferential warnings are printed in full immediately and
are also retained in {cmd:r(notes)}.{p_end}


{title:Options}

{phang}{opt x1(varlist)} lists the base covariates whose coefficient movement
is decomposed. {bf:Required}.

{phang}{cmd:aweight} and {cmd:fweight} apply the same strictly positive weight
to the base, full and auxiliary projections. Frequency weights must be positive
integers; their sum defines {cmd:r(n_obs_effective)}. Probability, importance
and survey weights are not implemented by this command.{p_end}

{phang}{opt x2groups(spec)} names the covariate groups added in the full model
with the b1x2-style syntax {cmd:x2groups("A = a1 a2 : B = b1 b2 b3")} (groups
separated by colons; each group is {it:name} {cmd:=} {it:varlist}). Each named
group becomes one column of the contribution matrix.

{phang}{opth fes(varlist)} gives fixed-effect dimensions (numeric categorical
identifiers) added in the full model, each absorbed with the xhdfe backend and
treated as its own group (always gamma0-style). Raw codes outside the signed
32-bit range, and non-integer numeric labels, are compacted internally without
changing category membership or results.

{phang}{opt absorbedtargets(varlist)} must be a subset of {opt x1()} and
requires {opt fes()}. It is an explicit request for absorbed-target allocation,
not an instruction to ignore arbitrary collinearity. Each named target must be
omitted because it is in the FE span; if it remains identified, is omitted for
another rank dependency, or any undeclared X1/X2 column is omitted, the command
fails. {cmd:r(b_full_status)} labels each X1 entry as {cmd:estimated} or
{cmd:imposed_zero}.{p_end}

{phang}{opt focal(varlist)} selects a nonempty subset of {opt x1()} for the
human-facing panels and share table. It is reporting metadata only. In
particular, a common control must remain in {opt x1()} even when it is omitted
from {opt focal()}; moving it into {opt x2groups()} changes the base model and
therefore changes the decomposition.{p_end}

{phang}{opt shares(type)} adds signed share estimates and stores their
full-precision matrices. {cmd:shares(movement)} divides each contribution by
the total movement and computes ratio SEs by the delta method from the joint
component covariance. {cmd:shares(base)} divides by the base coefficient, the
normalization used in several applications, and computes full ratio SEs and
normal-approximation confidence intervals from {cmd:r(cov)},
{cmd:r(base_cov)}, and {cmd:r(cov_delta_bbase)}. Its variance includes
uncertainty in both numerator and denominator and their covariance:
{p_end}
{p 8 8 2}
{it:Var(delta_g/b) = Var(delta_g)/b^2
 + delta_g^2 Var(b)/b^4 - 2 delta_g Cov(delta_g,b)/b^3}.
{p_end}
{pstd}
This full ratio convention is labelled
{cmd:joint_base_covariance_delta_method}.
{cmd:shares(base_fixed)} is a separate descriptive convention that scales
component SEs while holding the reported base coefficient fixed; it is labelled
{cmd:fixed_base_denominator_scaling} and must not be presented as full ratio
inference. Contributions in levels remain the primary result.{p_end}

{phang}{opt sharetol(#)} sets the absolute threshold below which a share
denominator is treated as undefined. It must be finite and nonnegative; the
default is {cmd:1e-12}. Undefined shares are missing and generate a warning.
The warning is printed and appended to {cmd:r(notes)}. This protects against
unstable percentages when the gap or movement is nearly zero.{p_end}

{phang}{opt level(#)} sets the normal-approximation confidence level for share
intervals. It has no effect unless {opt shares()} is specified and never
changes the underlying Gelbach fit.{p_end}

{pstd}{it:Common fixed effects and generated terms.} Low-dimensional effects
common to base and full can be entered as explicit indicator columns in
{opt x1()} and hidden from the display with {opt focal()}. High-dimensional
effects in {opt fes()} are, at present, added components of the full model;
there is no claim that the command absorbs a separate HDFE set common to both
models. Polynomial, spline, factor, and interaction blocks are supported after
the researcher generates the corresponding numeric columns and groups those
columns explicitly. This keeps the exact design matrix auditable.{p_end}

{pstd}For absorbed FE blocks, the reported standard errors are conditional/
{cmd:gamma0}: uncertainty from estimating the absorbed effects is not fully
included. {cmd:r(fe_total)} reports the aggregate of all absorbed-FE dimensions;
this is the preferred FE object when the FE graph has several mobility
components.{p_end}

{phang}{opt vce(vcetype)} selects the variance estimator: {cmd:unadjusted}
(default), {cmd:robust}, or {cmd:cluster} (supply {opth cluster(varname)}).
The cluster identifier is compacted under the same exact categorical rule.

{pstd}{it:Few clusters.} One-way clustered estimates remain available when
there are at least two retained clusters, but fewer than 30 triggers a loud
finite-sample warning in the Results window and {cmd:r(notes)}. The count and
warning threshold are returned in {cmd:r(n_clusters)} and
{cmd:r(few_cluster_warning_threshold)}. The command does not silently replace
the requested covariance with a bootstrap or another small-G procedure.{p_end}

{pstd}{bf:Absorbed-target inference.} For a declared target,
{it:total_j = b_base_j - 0} is the base-coefficient estimator itself. Its
target-target entry in {cmd:r(total_cov)} is therefore exactly the requested
base-model VCE. Inference for a target invariant within an absorbing FE must
be clustered at that FE dimension. {cmd:unadjusted}, {cmd:robust}, or
clustering on a crossed dimension can be anti-conservative; these choices are
retained for descriptive accounting but print a loud warning and set
{cmd:r(absorbed_target_inference_valid)} to 0. Inspect
{cmd:r(inference_status)} and {cmd:r(absorbing_fe_index)}.{p_end}

{pstd}The Gelbach covariance follows {cmd:b1x2}'s random-design stacked-
moment variance, including uncertainty in the auxiliary projections. It is
not the smaller variance conditional on the realised covariate design.
Absorbed-FE component covariances remain conditional/{cmd:gamma0}; hence an
absorbed-target total is labelled
{cmd:target_exact_base_vce_mixed_components}.{p_end}

{phang}{opt gamma0} retains the auxiliary-regression part of the component
variance and omits the sampling variance of the full-model added coefficients,
matching Gelbach's {cmd:b1x2}. Absorbed-FE components always carry this
conditional/{cmd:gamma0} treatment because their full sampling covariance is
not available.{p_end}

{phang}{opt cov0} removes the robust stacked-system cross terms, matching
{cmd:b1x2}. It affects robust or clustered covariance and is a no-op for
{cmd:vce(unadjusted)}. {cmd:r(observed_se_type)} and
{cmd:r(total_se_type)} disclose the effective inference contract.{p_end}

{phang}{opt tol(#)} sets the fixed-effect absorption tolerance. It must be
finite and strictly positive; the default {cmd:1e-8} preserves the historical
effective tolerance. It does not control the separate absorption-
classification boundary. The backend classifies an X1 column as FE-collinear
when {it:||M_D x||^2 / ||x||^2 <= 1e-9} (relative norm about 3.16e-5) and
returns that exact squared-norm threshold in
{cmd:r(fe_collinear_ss_ratio_tol)}. For every X1 column,
{cmd:r(x1_fe_collinear_ratio)} returns the measured ratio and
{cmd:r(x1_near_collinear_mask)} marks ratios in
{it:(1e-9, 1e-4]}. Such a column remains in the standard estimand, but the
command prints a loud warning because its component split and SEs can be
sensitive near the classification boundary. The diagnostic does not change
any estimate. {cmd:XHDFE_GELBACH_NEAR_COLLINEAR_WARN=0} suppresses only that
warning for controlled A/B runs; the matrices remain available.{p_end}

{phang}{opt threads(#)} requests the OpenMP team used by the CPU absorption,
FE-recovery and covariance phases; 0 delegates selection to the library.
Some small phases remain serial and CUDA kernel scheduling is separate.
{cmd:r(threads_used)} reports the effective maximum CPU team, which can be
smaller than the request when the retained problem is too small to benefit.

{phang}{opt gpu} requests the CUDA backend for the full HDFE specification.
The exact MLSMR fixed-effect recovery and the Gelbach covariance algebra stay
on CPU; this hybrid split preserves the certified FE normalization while
accelerating the absorption phase that benefits from the GPU. If CUDA is not
compiled, unavailable, fails, or does not converge, the existing xhdfe
fallback rules preserve the CPU result. Inspect {cmd:r(gpu_used)} and
{cmd:r(gpu_status)} to distinguish real CUDA use from fallback. CUDA applies
only to the full-model HDFE absorption; it does not move the base regression,
auxiliary covariance algebra, share delta method, or FE recovery to the GPU.

{phang}{opt verbose} streams phase-level progress live for the full fit,
certified FE recovery, base fit, covariance construction, and final
convergence check. Each line is flushed before the next phase. It changes
output only: quiet and verbose runs with the same inputs return identical
matrices.

{pstd}At least one of {opt x2groups()} or {opt fes()} must be supplied.

{pstd}{it:Performance environment switches} (A/B reproduction only — NOT a
safety fallback): {cmd:XHDFE_GELBACH_FAST_FIT=0} restores the single retained
full-model fit; {cmd:XHDFE_GELBACH_WARM_RECOVERY=0} disables the shared MLSMR
absorption that warm-starts the fixed-effect recovery and provides the within
transform (a convergence gate falls back to the retained fit automatically
when the recovery does not certify); {cmd:XHDFE_GELBACH_WITHIN_BATCH=0}
restores per-column within fits.

{pstd}{it:Warning (adversarial audit, 09jul2026).} The default fast path is
the MORE-converged one: it resolves the per-dimension FE split with an exact
normal-equations (MLSMR) solve behind a fail-closed convergence gate,
validated against direct sparse oracles up to 1M rows on adversarial FE
graphs. The legacy retained path recovers the split with Gauss-Seidel map
sweeps whose stopping rule is blind to the slow modes of ill-conditioned FE
graphs: on such graphs it can return a materially wrong per-dimension split
(sign flips observed) while reporting convergence. Do not set
{cmd:XHDFE_GELBACH_FAST_FIT=0} "to be safe" — it exists only to reproduce
pre-2.14.1 output bitwise for A/B comparison. Since 2.14.2 the legacy and
fallback paths cross-check their FE split against the exact normal equations
and set {cmd:r(converged)}=0 with an explanatory note when the check fails.
On well-conditioned designs the two paths agree to ~1e-11.

{pstd}{it:Near-collinear observed blocks.} A full model can converge and the
Gelbach summation identity can be exact while columns inside one
{opt x2groups()} block are so nearly collinear that the block's standard
errors depend materially on solver/ISA rounding. The backend runs a
bounded-cost normalized-Gram diagnostic and writes a warning to
the Results window and {cmd:r(notes)} for severe cases; point estimates are not
altered. In the
11jul2026 adversarial fixture (correlation 0.9999999999995), the default
tolerance agreed with a dense oracle within 0.024%, but alternative tight
tolerances moved the ill-conditioned block SE by as much as 7%, while a
well-conditioned block stayed within 2e-8. Such a warning means the split SE
is numerically indeterminate; tightening {opt tol()} does not select a unique
correct answer.{p_end}

{pstd}{it:Near-FE-collinear X1 columns.} This is distinct from near-collinearity
inside an observed X2 block. A standard-estimand X1 column with
{it:||M_D x||^2 / ||x||^2} just above the FE-omission boundary is still
estimated, but its location in the warning band is disclosed per column in
{cmd:r(x1_fe_collinear_ratio)} and {cmd:r(x1_near_collinear_mask)}. Review the
declared specification; use {opt absorbedtargets()} only when the variable is
conceptually invariant within an added FE and the constrained estimand is
actually intended.{p_end}

{pstd}{it:Floating-point reproducibility.} For {cmd:vce(cluster)}, the streamed
cluster-meat kernel may contract multiply-add operations (FMA). Relative to
the former materialized-score path this changes converged-cell SEs by at most
one last-place unit in the audited matrix (1.388e-17 absolute; coefficients
and deltas bit-identical). This deterministic rounding is accepted to retain
the large memory/runtime improvement. On deliberately ill-conditioned,
already non-converged cells it can be amplified; rely on {cmd:r(converged)}
and the conditioning note, not cross-build bit identity. Well-conditioned
Python/Stata/R specifications retain machine-precision parity. A separate
Stata-vs-Python plateau of about 7e-8 (or 2.6e-7 with fweights) was observed
only in an ill-conditioned FE/block split; its block-SE differences were much
larger and are covered by the warning above, so no universal 1e-7 accuracy
claim is made.{p_end}

{pstd}{it:Interpretation of per-FE-dimension contributions.} When the FE
graph has two or more mobility components, the split of the combined FE
contribution into per-dimension deltas depends on a normalization convention
(the component mean-shift used by {cmd:xhdfe}); the total across FE
dimensions and b_base - b_full are convention-invariant. Within a single
connected mobility component the x1-row split is identified. A small
{cmd:r(identity_gap)} certifies the decomposition identity only — it is NOT
evidence that the per-dimension split is accurate; check {cmd:r(converged)}.


{title:Programmatic matrix layout}

{pstd}
Rows of {cmd:r(delta)} and {cmd:r(se)} are {opt x1()} in declared order followed
by {cmd:_cons}. Columns are all observed groups in {opt x2groups()} order,
followed by the FE dimensions in {opt fes()} order. {cmd:r(total)} has the same
rows and columns {cmd:delta} and {cmd:se}. {cmd:r(b_base)}, {cmd:r(b_full)} and
{cmd:r(absorbed_mask)} contain only the X1 columns, not the intercept.{p_end}

{pstd}
{cmd:r(cov)} is group-major. If {it:k1 = number of X1 columns + 1}, its ordering
is
{cmd:[group1:X1, group1:_cons, group2:X1, group2:_cons, ...]}.
This is the authoritative object for sums and linear contrasts because it
retains every cross-component covariance. Do not add component variances or
SEs as if blocks were independent. Python's {cmd:gelbach.contrast()} and R's
{cmd:xhdfe_gelbach_contrast()} provide convenience wrappers around the same
algebra; Stata users can construct the corresponding weight vector against
{cmd:r(cov)}.{p_end}

{pstd}
{cmd:r(base_cov)} is the requested-VCE covariance of the base coefficients in
{cmd:[X1, _cons]} order. {cmd:r(cov_delta_bbase)} has group-major contribution
rows and {cmd:[X1, _cons]} base-coefficient columns.
{cmd:r(cov_total_bbase)} is their sum over groups. These matrices support
joint inference for base-normalized shares and other smooth functions; they
must be used rather than assuming an independent denominator.{p_end}

{pstd}
{cmd:r(gamma)} contains the full-model coefficients of observed
{opt x2groups()} blocks. Columns follow observed-block declaration order;
rows are coefficient positions within a block, padded with missing values to
the widest observed block. FE blocks have no columns in {cmd:r(gamma)} because
their effects are absorbed rather than represented by one finite coefficient
vector.{p_end}

{pstd}
When {opt shares()} is requested, {cmd:r(share)}, {cmd:r(share_se)} and the two
CI matrices have the same shape and names as {cmd:r(delta)}. The one-column
{cmd:r(share_defined)} marks usable denominators; an unusable denominator
makes its share, SE and interval missing and adds a warning to {cmd:r(notes)}.
{cmd:r(residual_share)} is defined only for X1 rows under a base-coefficient
denominator; it is missing for movement shares and for undefined denominators.
Stored shares are fractions even though the display multiplies them by 100.{p_end}


{title:Deliberate limits}

{pstd}
The current command supports linear OLS coefficient-movement accounting with
unadjusted, robust or one-way clustered inference. It does not implement
multiway clustering, wild-cluster bootstrap, IV/2SLS or LATE allocation,
split-panel/dynamic corrections, nonconditional recovered-FE covariance,
kernel/MM-quantile/KHB/GLM/distributional decompositions, Oaxaca wrappers, or
causal-mediation estimands. These require separate estimators and must not be
approximated by relabelling the OLS output.{p_end}

{pstd}
Low-dimensional FE common to base and full may be represented by explicit
indicators in {opt x1()} and hidden with {opt focal()}. Every HDFE dimension in
{opt fes()} is currently an added full-model component; a separate HDFE set
absorbed in both specifications is not implemented. Stata factor-variable
notation is not expanded inside {opt x1()} or {opt x2groups()}; generate the
numeric indicators, powers, bins, splines and interactions explicitly. Because
the intercept is implicit, common categorical indicators must use a full-rank
reference coding rather than all category dummies.{p_end}

{pstd}
For a binary outcome, the command can decompose coefficients from a declared
linear probability model. Contributions are then in probability units (or
percentage points after multiplying by 100), and robust or substantively
appropriate clustered inference is normally preferable to unadjusted OLS
inference. This remains a linear coefficient-movement accounting exercise;
it is not a decomposition of nonlinear response probabilities and does not
repair misspecification or justify a causal interpretation.{p_end}


{title:Stored results}

{pstd}{cmd:xhdfegelbach} stores the following in {cmd:r()}:
{cmd:r(fe_total)} exists only with {opt fes()}; share matrices and share macros
exist only with {opt shares()}; {cmd:r(gamma)} exists only with
{opt x2groups()}; {cmd:r(notes)} exists only when the backend or the reporting
layer emits a note or warning.{p_end}

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Matrices}{p_end}
{synopt:{cmd:r(delta)}}contributions, rows for {opt x1()} plus {cmd:_cons}, one column per group{p_end}
{synopt:{cmd:r(se)}}standard errors matching {cmd:r(delta)}{p_end}
{synopt:{cmd:r(total)}}total movement for {opt x1()} plus {cmd:_cons}; columns {cmd:delta} and {cmd:se}{p_end}
{synopt:{cmd:r(b_base)}}base-model coefficients on {opt x1()}{p_end}
{synopt:{cmd:r(b_full)}}full-model coefficients on {opt x1()}; absorbed targets are imposed zero{p_end}
{synopt:{cmd:r(absorbed_mask)}}backend classification mask in X1 order (1 = imposed absorbed target){p_end}
{synopt:{cmd:r(x1_fe_collinear_ratio)}}per-X1 squared residual-norm ratio after absorbing {opt fes()}{p_end}
{synopt:{cmd:r(x1_near_collinear_mask)}}per-X1 warning-band indicator{p_end}
{synopt:{cmd:r(gamma)}}padded full-model coefficients for observed X2 blocks{p_end}
{synopt:{cmd:r(cov)}}joint covariance of all group contributions{p_end}
{synopt:{cmd:r(total_cov)}}covariance of the total movement{p_end}
{synopt:{cmd:r(base_cov)}}requested-VCE covariance of base coefficients plus intercept{p_end}
{synopt:{cmd:r(cov_delta_bbase)}}cross-covariance of group-major contributions with base coefficients{p_end}
{synopt:{cmd:r(cov_total_bbase)}}cross-covariance of total movement with base coefficients{p_end}
{synopt:{cmd:r(fe_total)}}aggregate absorbed-FE contribution and conditional SE for {opt x1()} plus {cmd:_cons}{p_end}
{synopt:{cmd:r(share)}}signed component shares; returned only with {opt shares()}{p_end}
{synopt:{cmd:r(share_se)}}delta-method share SEs; fixed-denominator scaling only under {cmd:base_fixed}{p_end}
{synopt:{cmd:r(share_ci_low)}, {cmd:r(share_ci_high)}}share confidence limits{p_end}
{synopt:{cmd:r(share_defined)}}row indicator that the denominator exceeds {opt sharetol()}{p_end}
{synopt:{cmd:r(residual_share)}}full-model residual divided by the base coefficient, when defined{p_end}

{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(identity_gap)}}residual of the summation identity (should be ~0){p_end}
{synopt:{cmd:r(n_obs_input)}}observations entering the backend after Stata markout{p_end}
{synopt:{cmd:r(n_obs)}}retained row count (historical field){p_end}
{synopt:{cmd:r(n_obs_effective)}}reported N: retained rows normally, sum of retained weights under {cmd:fweight}{p_end}
{synopt:{cmd:r(n_singletons_dropped)}}observations removed by recursive FE singleton dropping{p_end}
{synopt:{cmd:r(df_full)}}residual degrees of freedom of the full model{p_end}
{synopt:{cmd:r(df_base)}}residual degrees of freedom of the base model{p_end}
{synopt:{cmd:r(n_clusters)}}retained independent clusters for {cmd:vce(cluster)}; 0 otherwise{p_end}
{synopt:{cmd:r(converged)}}1 if the computation converged{p_end}
{synopt:{cmd:r(tol)}}fixed-effect absorption tolerance used{p_end}
{synopt:{cmd:r(focal_selection_explicit)}}1 when {opt focal()} was specified{p_end}
{synopt:{cmd:r(conf_level)}}requested {opt level()} as a fraction; returned even without shares{p_end}
{synopt:{cmd:r(share_tol)}}requested denominator threshold; returned even without shares{p_end}
{synopt:{cmd:r(fe_collinear_ss_ratio_tol)}}squared-norm FE-classification threshold ({cmd:1e-9}){p_end}
{synopt:{cmd:r(near_fe_warn_upper)}}upper warning-band edge ({cmd:1e-4}); the shortened name respects Stata's identifier limit{p_end}
{synopt:{cmd:r(few_cluster_warning_threshold)}}cluster-count warning threshold ({cmd:30}){p_end}
{synopt:{cmd:r(absorbed_target_inference_valid)}}1 only when absorbed-target inference is clustered at a matching absorbing FE{p_end}
{synopt:{cmd:r(absorbing_fe_index)}}zero-based matching FE dimension, or -1{p_end}
{synopt:{cmd:r(threads_used)}}effective thread count reported by the backend{p_end}
{synopt:{cmd:r(gpu_requested)}}1 when CUDA was requested by {opt gpu} or the active backend selector{p_end}
{synopt:{cmd:r(gpu_used)}}1 only if CUDA was actually used{p_end}
{synopt:{cmd:r(gpu_status_code)}}0 not requested; 1 used; 2 unavailable; 3 not converged; 4 failed; 5 CPU cache; 6 not applicable{p_end}
{synopt:{cmd:r(gpu_attempted)}}1 if GPU absorption was attempted{p_end}
{synopt:{cmd:r(gpu_absorption_converged)}}1 if attempted GPU absorption converged{p_end}
{synopt:{cmd:r(gpu_absorption_iterations)}}iterations of attempted GPU absorption{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(vce)}}the variance estimator used{p_end}
{synopt:{cmd:r(groups)}}the group names{p_end}
{synopt:{cmd:r(x1_names)}}all {opt x1()} variable names in design order{p_end}
{synopt:{cmd:r(focal_indices)}}zero-based reporting indices; all X1 indices when {opt focal()} is omitted{p_end}
{synopt:{cmd:r(focal_names)}}reporting names; all X1 names when {opt focal()} is omitted{p_end}
{synopt:{cmd:r(share_denominator)}}{cmd:movement}, {cmd:base}, or {cmd:base_fixed}{p_end}
{synopt:{cmd:r(share_se_type)}}movement/base joint delta method or fixed-denominator scaling{p_end}
{synopt:{cmd:r(share_units)}}{cmd:fraction}; the display multiplies by 100{p_end}
{synopt:{cmd:r(notes)}}any solver notes{p_end}
{synopt:{cmd:r(estimand)}}{cmd:coefficient_movement} or {cmd:absorbed_target_allocation}{p_end}
{synopt:{cmd:r(identity_status)}}{cmd:exact_ols} or {cmd:exact_ols_constrained}{p_end}
{synopt:{cmd:r(absorbed_targets)}}zero-based backend-classified absorbed X1 indices, consistent across frontends{p_end}
{synopt:{cmd:r(absorbed_target_names)}}backend-classified absorbed X1 variable names{p_end}
{synopt:{cmd:r(b_full_status)}}per-X1 {cmd:estimated}/{cmd:imposed_zero} labels, in X1 order{p_end}
{synopt:{cmd:r(focal_status)}}per-X1 identification labels, {cmd:identified} or {cmd:absorbed}; unrelated to the reporting selector{p_end}
{synopt:{cmd:r(observed_se_type)}}{cmd:full}, {cmd:gamma0}, or {cmd:cov0}{p_end}
{synopt:{cmd:r(total_se_type)}}whether total inference is full, conditional, or mixed{p_end}
{synopt:{cmd:r(inference_status)}}absorbed-target clustering status or {cmd:not_applicable}{p_end}
{synopt:{cmd:r(causal_interpretation)}}{cmd:no}{p_end}
{synopt:{cmd:r(fe_se_type)}}{cmd:conditional_gamma0}{p_end}
{synopt:{cmd:r(gpu_backend)}}effective backend, {cmd:cuda} or {cmd:cpu}{p_end}
{synopt:{cmd:r(gpu_status)}}{cmd:not_requested}, {cmd:used}, {cmd:unavailable}, {cmd:not_converged}, {cmd:failed}, {cmd:cpu_cache}, or {cmd:not_applicable}{p_end}
{p2colreset}{...}


{title:Examples}

{pstd}Account for the movement of the education coefficient using an ability
block, a job-covariate block and a firm fixed-effect block:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ) x2groups("ability = ability : job_covariates = tenure exper") fes(firm_id)}{p_end}
{phang2}{cmd:. matrix list r(delta)}{p_end}
{phang2}{cmd:. matrix list r(total)}{p_end}

{pstd}Cluster-robust inference by firm:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ) x2groups("job = tenure exper") vce(cluster) cluster(firm_id)}{p_end}

{pstd}Keep age and year indicators common to both models, report only education,
and obtain signed shares of the coefficient movement:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ age y2005 y2006) focal(educ) x2groups("ability = ability : job = tenure exper") fes(firm_id) shares(movement)}{p_end}
{phang2}{cmd:. matrix list r(share)}{p_end}
{phang2}{cmd:. matrix list r(share_se)}{p_end}
{phang2}{cmd:. matrix list r(share_ci_low)}{p_end}

{pstd}Report shares of the base coefficient with full ratio inference using
the numerator-denominator cross-covariance:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ age) focal(educ) x2groups("job = tenure exper") fes(firm_id) shares(base)}{p_end}
{phang2}{cmd:. matrix list r(share)}{p_end}
{phang2}{cmd:. matrix list r(share_se)}{p_end}

{pstd}Allocate a worker-invariant group coefficient after worker FE absorb it.
The zero in {cmd:r(b_full)} is imposed and is labelled accordingly:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(female age) x2groups("job = tenure exper") fes(worker_id firm_id) absorbedtargets(female) vce(cluster) cluster(worker_id)}{p_end}
{phang2}{cmd:. di "`r(estimand)'  `r(b_full_status)'"}{p_end}

{pstd}Request real CUDA absorption and show phase progress:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ) x2groups("job = tenure exper") fes(firm_id year) vce(cluster) cluster(worker_id) gpu verbose}{p_end}
{phang2}{cmd:. return list}{p_end}

{pstd}Two executable empirical designs ship in all three frontends: the
standard decomposition and the absorbed-target allocation. The Stata files
are {bf:examples/gelbach_example.do} and
{bf:examples/gelbach_absorbed_target.do}; matching Python and R files are in
the same directory. Thus the example suite is two designs by three
frontends, not six different estimands.{p_end}


{title:References}

{pstd}
Gelbach, J. B. 2016. When do covariates matter? And which ones, and how much?
{it:Journal of Labor Economics} 34(2): 509-543. Validated against his
{cmd:b1x2} Stata implementation.{p_end}


{title:Also see}

{psee}
{helpb xhdfe}, {helpb xhdfeakm}, {helpb xhdfeconnected}
{p_end}
