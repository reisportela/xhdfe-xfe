{smcl}
{* *! version 2.18.0  11jul2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xhdfeakm" "help xhdfeakm"}{...}
{title:Title}

{p2colset 5 22 24 2}{...}
{p2col :{cmd:xhdfegelbach} {hline 2}}Gelbach (2016) conditional decomposition of coefficient movements{p_end}
{p2colreset}{...}


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
{synopt :{opt vce(vcetype)}}{cmd:unadjusted} (default), {cmd:robust}, or {cmd:cluster}{p_end}
{synopt :{opth cluster(varname)}}cluster identifier (with {cmd:vce(cluster)}){p_end}
{synopt :{opt gamma0}}reproduce b1x2's {cmd:gamma0} variant{p_end}
{synopt :{opt cov0}}reproduce b1x2's {cmd:cov0} variant{p_end}
{synopt :{opt tol(#)}}FE absorption tolerance; default {cmd:1e-8}{p_end}
{synopt :{opt threads(#)}}OpenMP threads (0 = library default){p_end}
{synopt :{opt gpu}}request CUDA for the full HDFE absorption{p_end}
{synopt :{opt verbose}}print deterministic phase progress and elapsed time{p_end}
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

{pstd}{bf:Interpretation warning.} This is a decomposition of coefficient
movement for two declared specifications, not causal mediation. Causal
interpretation requires a separately justified research design. Adding
post-treatment variables, colliders, or fixed effects that alter the identifying
comparisons can change the estimand or introduce bias. Block names and causal
roles are supplied by the researcher and are not validated by the command.{p_end}


{title:Options}

{phang}{opt x1(varlist)} lists the base covariates whose coefficient movement
is decomposed. {bf:Required}.

{phang}{opt x2groups(spec)} names the covariate groups added in the full model
with the b1x2-style syntax {cmd:x2groups("A = a1 a2 : B = b1 b2 b3")} (groups
separated by colons; each group is {it:name} {cmd:=} {it:varlist}). Each named
group becomes one column of the contribution matrix.

{phang}{opth fes(varlist)} gives fixed-effect dimensions (numeric categorical
identifiers) added in the full model, each absorbed with the xhdfe backend and
treated as its own group (always gamma0-style). Raw codes outside the signed
32-bit range, and non-integer numeric labels, are compacted internally without
changing category membership or results.

{pstd}For absorbed FE blocks, the reported standard errors are conditional/
{cmd:gamma0}: uncertainty from estimating the absorbed effects is not fully
included. {cmd:r(fe_total)} reports the aggregate of all absorbed-FE dimensions;
this is the preferred FE object when the FE graph has several mobility
components.{p_end}

{phang}{opt vce(vcetype)} selects the variance estimator: {cmd:unadjusted}
(default), {cmd:robust}, or {cmd:cluster} (supply {opth cluster(varname)}).
The cluster identifier is compacted under the same exact categorical rule.

{phang}{opt gamma0} and {opt cov0} reproduce the corresponding options of
Gelbach's {cmd:b1x2}.

{phang}{opt tol(#)} sets the fixed-effect absorption tolerance. It must be
finite and strictly positive; the default {cmd:1e-8} preserves the historical
effective tolerance.

{phang}{opt threads(#)} sets OpenMP threads (0 = library default).

{phang}{opt gpu} requests the CUDA backend for the full HDFE specification.
The exact MLSMR fixed-effect recovery and the Gelbach covariance algebra stay
on CPU; this hybrid split preserves the certified FE normalization while
accelerating the absorption phase that benefits from the GPU. If CUDA is not
compiled, unavailable, fails, or does not converge, the existing xhdfe
fallback rules preserve the CPU result. Inspect {cmd:r(gpu_used)} and
{cmd:r(gpu_status)} to distinguish real CUDA use from fallback.

{phang}{opt verbose} prints phase-level progress for the full fit, certified
FE recovery, base fit, covariance construction, and final convergence check.
It changes output only: quiet and verbose runs with the same inputs return
identical matrices.

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
{cmd:r(notes)} for severe cases; point estimates are not altered. In the
11jul2026 adversarial fixture (correlation 0.9999999999995), the default
tolerance agreed with a dense oracle within 0.024%, but alternative tight
tolerances moved the ill-conditioned block SE by as much as 7%, while a
well-conditioned block stayed within 2e-8. Such a warning means the split SE
is numerically indeterminate; tightening {opt tol()} does not select a unique
correct answer.{p_end}

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


{title:Stored results}

{pstd}{cmd:xhdfegelbach} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Matrices}{p_end}
{synopt:{cmd:r(delta)}}contributions, one row per {opt x1()} variable, one column per group{p_end}
{synopt:{cmd:r(se)}}standard errors matching {cmd:r(delta)}{p_end}
{synopt:{cmd:r(total)}}total movement per {opt x1()} variable (base minus full) with its SE{p_end}
{synopt:{cmd:r(b_base)}}base-model coefficients on {opt x1()}{p_end}
{synopt:{cmd:r(b_full)}}full-model coefficients on {opt x1()}{p_end}
{synopt:{cmd:r(cov)}}joint covariance of all group contributions{p_end}
{synopt:{cmd:r(total_cov)}}covariance of the total movement{p_end}
{synopt:{cmd:r(fe_total)}}aggregate absorbed-FE contribution and conditional SE (when {opt fes()} is used){p_end}

{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(identity_gap)}}residual of the summation identity (should be ~0){p_end}
{synopt:{cmd:r(n_obs)}}number of observations{p_end}
{synopt:{cmd:r(df_full)}}residual degrees of freedom of the full model{p_end}
{synopt:{cmd:r(converged)}}1 if the computation converged{p_end}
{synopt:{cmd:r(tol)}}fixed-effect absorption tolerance used{p_end}
{synopt:{cmd:r(threads_used)}}effective thread count reported by the backend{p_end}
{synopt:{cmd:r(gpu_requested)}}1 when {opt gpu} was specified{p_end}
{synopt:{cmd:r(gpu_used)}}1 only if CUDA was actually used{p_end}
{synopt:{cmd:r(gpu_status_code)}}0 not requested; 1 used; 2 unavailable; 3 not converged; 4 failed; 5 CPU cache; 6 not applicable{p_end}
{synopt:{cmd:r(gpu_attempted)}}1 if GPU absorption was attempted{p_end}
{synopt:{cmd:r(gpu_absorption_converged)}}1 if attempted GPU absorption converged{p_end}
{synopt:{cmd:r(gpu_absorption_iterations)}}iterations of attempted GPU absorption{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(vce)}}the variance estimator used{p_end}
{synopt:{cmd:r(groups)}}the group names{p_end}
{synopt:{cmd:r(notes)}}any solver notes{p_end}
{synopt:{cmd:r(estimand)}}{cmd:coefficient_movement}{p_end}
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

{pstd}Request real CUDA absorption and show phase progress:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ) x2groups("job = tenure exper") fes(firm_id year) vce(cluster) cluster(worker_id) gpu verbose}{p_end}
{phang2}{cmd:. return list}{p_end}

{pstd}A fuller walkthrough (Stata/Python/R) ships as
{bf:examples/gelbach_example.do}.{p_end}


{title:References}

{pstd}
Gelbach, J. B. 2016. When do covariates matter? And which ones, and how much?
{it:Journal of Labor Economics} 34(2): 509-543. Validated against his
{cmd:b1x2} Stata implementation.{p_end}


{title:Also see}

{psee}
{helpb xhdfe}, {helpb xhdfeakm}, {helpb xhdfeconnected}
{p_end}
