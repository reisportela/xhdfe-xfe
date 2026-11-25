{smcl}
{* *! version 1.14.7 25nov2026}{...}
{vieweralsosee "hdfe" "help hdfe"}{...}
{vieweralsosee "reghdfe" "help reghdfe"}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{viewerjumpto "Development process" "xfepout##development"}{...}
{viewerjumpto "Syntax" "xfepout##syntax"}{...}
{viewerjumpto "Absvar Syntax" "xfepout##absvar"}{...}
{viewerjumpto "Options" "xfepout##options"}{...}
{viewerjumpto "Examples" "xfepout##examples"}{...}
{viewerjumpto "Stored results" "xfepout##results"}{...}
{title:Title}

{p2colset 5 18 20 2}{...}
{p2col :{cmd:xfepout} {hline 2}}Partial-out variables with respect to multiple levels of fixed effects via the xhdfe C++ absorption core{p_end}
{p2colreset}{...}

{pstd}{cmd:xfepout} is a separate package from {cmd:xhdfe} and ships its own plugin binary
({cmd:xfepout.plugin}). It is a partialling-out command (no coefficient table).{p_end}

{marker development}{...}
{title:Development process}

{pstd}{cmd:xfepout} shares the {help xhdfe##development:xhdfe development and validation process}.{p_end}

{marker syntax}{...}
{title:Syntax}

{pstd}Replace current dataset:

{p 8 15 2}
{cmd:xfepout}
{it:varlist}
{ifin}
{it:{weight}}
{cmd:,}
{opth abs:orb(xfepout##absvar:absvars)}
{opt clear}
[{opth keepv:ars(varlist)} {opt keepid:s}]
[{opth clusterv:ars(varlist)} {help xfepout##options:options}]
{p_end}

{pstd}Keep current dataset and add transformed variables:

{p 8 15 2}
{cmd:xfepout}
{it:varlist}
{ifin}
{it:{weight}}
{cmd:,}
{opth abs:orb(xfepout##absvar:absvars)}
{opt gen:erate(stubname)}
[{opt sample(newvarname)}]
[{opth clusterv:ars(varlist)} {help xfepout##options:options}]
{p_end}

{pstd}Show package version:

{p 8 15 2}
{cmd:xfepout, version}
{p_end}
{p 4 6 2}The version query clears the current {cmd:e()} results.{p_end}

{p 4 6 2}{bf:Important:} {it:varlist} must contain existing numeric variables (no factor-variable or time-series operators in the varlist).{p_end}
{p 4 6 2}{opt absorb(absvars)} is required.{p_end}
{p 4 6 2}Specify exactly one of {cmd:clear} or {cmd:generate()}. {cmd:sample()} is only allowed with
{cmd:generate()}, while {cmd:keepvars()} and {cmd:keepids} are only allowed with {cmd:clear}.{p_end}
{p 4 6 2}{cmd:fweight}s, {cmd:aweight}s, {cmd:pweight}s, and {cmd:iweight}s are allowed; see {help weight}.{p_end}
{p 4 6 2}Negative importance weights are rejected; zero-weight rows are excluded.{p_end}

{marker absvar}{...}
{title:Absvar Syntax}

{pstd}{cmd:absorb()} and {cmd:clustervars()} accept categorical identifiers and categorical interactions
with two or more components, written with {cmd:#}/{cmd:##} (for example {cmd:a#b#c}). For reghdfe/xhdfe
compatibility, categorical-only {cmd:##} terms in {cmd:absorb()} are treated as one joint absorbed FE, not
expanded into main effects plus the interaction. In {cmd:clustervars()}, two-component {cmd:##} expands to
the two main cluster dimensions plus their interaction; {cmd:##} with more than two components is rejected
(list the cluster dimensions explicitly). {cmd:reghdfe} treats {cmd:a##b} in {cmd:vce(cluster)} as one joint
factor, like {cmd:a#b}; use {cmd:a#b} for that meaning in both commands. The grammar is intentionally narrower than {cmd:xhdfe}: these
options are for fixed-effect or cluster identifiers, not for heterogeneous slopes.{p_end}

{p 4 6 2}- Interactions are materialized with {cmd:egen group()} over all their components after the
estimation sample is defined. Plain numeric identifiers are passed directly to the plugin;
string identifiers are coded with {cmd:egen group()} before the plugin call ({cmd:xfepout} does not use the
frame-based coding of string identifiers in {helpb xhdfe}).{p_end}
{p 4 6 2}- Factor-style prefixes such as {cmd:i.}, {cmd:ib#.}, {cmd:bn.}, and {cmd:ibn.} are stripped
for compatibility before the underlying variable names are resolved.{p_end}
{p 4 6 2}- Prefix {cmd:c.} and heterogeneous-slope syntax such as {cmd:var#c.x} or {cmd:var##c.x}
are rejected. They are not interpreted as categorical identifiers in {cmd:xfepout}.{p_end}
{p 4 6 2}- {cmd:absorb()} suboptions such as {cmd:savefe} are not supported in {cmd:xfepout}.{p_end}

{marker options}{...}
{title:Options}

{dlgtab:HDFE-specific}
{phang}{opt clear} overwrite the current dataset with the kept estimation sample, replacing {it:varlist}
by its residualized version. The resulting dataset keeps the transformed variables plus absorbed/clustering
variables, weights, optional {cmd:groupvar()}, anything requested in {cmd:keepvars()}, and any
materialized FE/cluster ids requested through {cmd:keepids}.{p_end}
{phang}{opt generate(stub)} keep the current dataset and create new double-precision transformed variables
named {it:stub}{it:varname}. Each new variable is labelled {cmd:Residuals: <source label>}.{p_end}
{phang}{opt sample(newvar)} save a sample marker (1=kept, 0=dropped/outside sample). Only with
{opt generate()}; the generated variable is labelled {cmd:[XFEPOUT Sample]}.{p_end}
{phang}{opt keepvars(varlist)} in {opt clear} mode, keep additional variables in the rewritten dataset.{p_end}
{phang}{opt keepids} in {opt clear} mode, keep the materialized integer FE/cluster ids created for the
plugin (for example {cmd:__xfe_ID*__} and {cmd:__xfe_CL*__}).{p_end}
{phang}{opt clustervars(varlist)} supplies cluster identifiers for absorbed-DoF
bookkeeping in {cmd:e(df_a)}. Missing cluster identifiers exclude rows from
the sample used for all transformed variables. Aliases are {opt cl()} and
{opt cluster()}.{p_end}

{dlgtab:Optimization}
{marker opt_tolerance}{...}
{phang}{opt tolerance(#)} absorber tolerance (default 1e-8; valid range (0,1)).
The selected method uses reghdfe-comparable semantics, matching the default mode
of {cmd:xhdfe}, together with the required projection-precision checks.
Ordinary three-effect CUDA projections can use a completion step at the same
tolerance.{p_end}
{phang}{opt maxiter(#)} maximum total absorption iterations (default 100000),
including any continuation within the remaining budget.{p_end}
{pmore}A failed necessary precision check after continuation returns no estimates.
Some ill-conditioned three-effect CUDA partial-out cases still refuse with multiple
workers; use {cmd:gpubackend(cpu)} for these designs.
{p_end}
{phang}{opt transform(str)} accept hdfe-style transform names: {it:kac}, {it:cim}, {it:sym}. {it:sym}
turns on symmetric sweeps; {it:kac} and {it:cim} only steer method choice when
{cmd:absorptionmethod(auto)} is in effect.{p_end}
{phang}{opt absorptionmethod(str)} backend method. Supported values are:{p_end}
{pmore}{it:auto}; {it:gauss-seidel} (aliases {it:gauss_seidel}, {it:gs});
{it:symmetric-gauss-seidel} (aliases {it:symmetric_gauss_seidel}, {it:sym},
{it:symmetric}, {it:symgs}); and {it:jacobi}.{p_end}
{pmore}{it:lsmr} (aliases {it:plain-lsmr}, {it:plain_lsmr});
{it:mlsmr} (aliases {it:modified-lsmr}, {it:modified_lsmr}, {it:within},
{it:within-additive}, {it:within_additive}); and {it:auto-mlsmr}
(aliases {it:auto_mlsmr}, {it:mlsmr-auto}, {it:mlsmr_auto}).{p_end}
{pmore}The default is {it:symmetric-gauss-seidel}. Explicit {it:lsmr} and {it:mlsmr}
requests are CPU-only.{p_end}
{pmore}For partial-out with analytic weights and two or three fixed effects,
the implicit method or {cmd:absorptionmethod(auto)} can retry if the initial
sweep projection fails its precision check. This automatic retry requires that
{cmd:transform()} and {cmd:symmetricsweep} be omitted. It uses MLSMR on CPU or
device LSMR on CUDA (up to 2,000,000 estimation rows) and shares the original
{cmd:maxiter()} budget. {cmd:e(method_used)} reports the accepted method.{p_end}
{pmore}{cmd:xfepout} never enters the reduced-system tier of {help xhdfe##reduced_pcg:xhdfe}: the absorption
runs the route selected here, whatever {cmd:XHDFE_REDUCED_PCG} is set to.{p_end}
{phang}{opt symmetricsweep} force forward/backward sweeps per iteration.{p_end}
{phang}{opt jacobirelaxation(#)} Jacobi relaxation parameter (default 0).{p_end}
{phang}{opt keepsingletons} do not drop singleton observations.{p_end}

{dlgtab:Threading}
{phang}{opt numthreads(#)} direct OpenMP thread request (0=auto). A positive
request bypasses auto-only workload and {opt maxthreads()} heuristics and is
limited only by the logical processors/OpenMP limit visible to the process.{p_end}
{phang}{opt defaultthreads(#)} default auto-thread count when {opt numthreads(0)}.{p_end}
{phang}{opt maxthreads(#)} cap for auto-thread selection (0=no cap).{p_end}
{phang}{opt minparallelrows(#)} minimum rows before automatic parallelisation is considered.{p_end}
{phang}{opt targetrowsperthread(#)} target row budget per thread in auto-threading.
The default, 0, is automatic: one thread per 25,000 rows within the problem-size
cap (one thread below 100,000 rows, at most 8 below 250,000 and at most 16
otherwise). A positive value applies that fixed budget; 500000 restores the
automatic rule of package version 2.30.1. Setting {cmd:XHDFE_AUTO_ROWS_PER_THREAD}
to a positive integer before Stata starts replaces the 25,000.{p_end}
{phang}On Linux and macOS the automatic teams of one user's simultaneous fits
share the machine as described under {help xhdfe##opt_parallel:parallel execution}
in {help xhdfe}: an automatic team never exceeds the rule above nor falls below
that of {cmd:targetrowsperthread(500000)}, a positive {opt numthreads()} is never
reduced, and {cmd:e(thread_limit_code)} 6 reports a team lowered by the
coordination. {cmd:XHDFE_AUTO_TEAM_COORDINATION=0} before Stata starts turns it
off.{p_end}

{dlgtab:GPU}
{phang}{opt gpubackend(cpu|cuda|metal)} selects the backend. CUDA needs a
CUDA-enabled plugin and an available device. Metal is reserved. An explicit
non-CPU request returns error 498 if the selected backend cannot provide a
converged result or if the result would come from a CPU transform.
Check {cmd:e(gpu_used)}, {cmd:e(gpu_backend)}, {cmd:e(gpu_status)} and
{cmd:e(cuda_tail)}. See {helpb xhdfegpu} for Linux CUDA builds. After replacing
a plugin or changing its {cmd:adopath} entry, run {cmd:discard} with no arguments
or restart Stata.{p_end}

{dlgtab:Cache and Profile}
{phang}{opt mobilityprofile} write/update a mobility profile at the path supplied by
{cmd:mobfile()} or, if no path is supplied, at {cmd:xfepout_mobility_profile.txt}.{p_end}
{phang}{opt mobfile(path)} mobility profile path; enables profile auto mode at that path.
{cmd:mobilityfile()} and {cmd:mobilityprofilefile()} are accepted as aliases.{p_end}
{pmore}Version-2 profiles have explicit {cmd:standard} scope and FE-structure signatures. They are
method-selection hints, not transformed-data caches: the applicable signature and scope must match before
reuse. Legacy version-1 files and mismatches are safe misses. A suggested method is retained only
after convergence and independent precision certification; otherwise automatic selection restarts
from the original data. CPU-only method hints are not forced on CUDA runs, and profiles are written
only after convergence and certification.{p_end}
{phang}{opt absorptioncache(path)} specifies a cache of transformed variables.
{opt abscachefile()} is an alias.{p_end}
{phang}{opt abscachemode(mode)} selects {it:off}, {it:auto}, {it:read}, or {it:write}.
{opt absorptioncachemode()} is an alias. Active caching needs an explicit
cache path or a profile path from which the companion cache path is derived.
Reuse requires matching input, FE structure, weights and solver contract.
Version-6 entries written by a fit with saved effects have a distinct
signature. A version-5 entry is eligible only where its solver signature
or recorded CPU refinement identifies a writer without captured effects.
{it:read} leaves the file untouched, {it:write} writes a fresh certified
result, and {it:auto} reads valid entries and can replace misses.
CUDA absorption skips reads from this cache.{p_end}
{phang}{opt fescache(path)} path for FE-structure cache payload.{p_end}
{phang}{opt festructurecache} use default FE-structure cache path relative to {cmd:xfepout.ado}.
{cmd:festructurecache()} and {cmd:festructurecachefile()} are accepted as aliases for
{cmd:fescache()}.{p_end}
{phang}{opt fecachemode(mode)} FE-structure cache mode: {it:off}, {it:auto}, {it:read}, {it:write}.
{cmd:fescachemode()} and {cmd:festructurecachemode()} are accepted as aliases.{p_end}

{dlgtab:DoF}
{phang}{opt dofadjustments(list)} reghdfe-style DoF adjustments (tokens include {it:all},
{it:none}, {it:firstpair}/{it:first}, {it:pairwise}/{it:pair}, {it:clusters}/{it:cluster},
{it:continuous}/{it:cont}). {cmd:dof()} is accepted as an alias.{p_end}
{phang}{opt groupvar(newvar)} save the first mobility-group id (connected component;
1-based) as a new variable when the backend computes the relevant mobility structure.{p_end}

{dlgtab:Compatibility}
{phang}{opt maxiterations(#)} alias for {opt maxiter(#)}.{p_end}
{phang}{opt iterate(#)}, {opt iterations(#)}, and {opt iter(#)} are aliases for {opt maxiter(#)}.{p_end}
{phang}{opt a(...)}/{opt abs(...)} are aliases for {opt absorb(...)}; {opt groupv()} is an alias
for {opt groupvar()}.{p_end}
{phang}{opt nowarn} suppress singleton warning messages; transformation is unchanged.{p_end}
{phang}{opt poolsize(#)}, {opt acceleration(str)}, {opt prune} and
{opt verbose(#)} are accepted for hdfe compatibility and leave the
transformation unchanged.{p_end}
{phang}{opt timeit} print plugin wall-clock time from the ado layer.{p_end}

{marker examples}{...}
{title:Examples}

{pstd}Partial out firm and year effects, then calculate the within slopes.
Use {helpb xhdfe} when you need absorbed DoF and regression inference.{p_end}
{phang2}{cmd:. use "https://www.stata-press.com/data/r19/grunfeld.dta", clear}{p_end}
{phang2}{cmd:. xfepout invest mvalue kstock, absorb(company year) generate(within_) sample(in_fit)}{p_end}
{phang2}{cmd:. regress within_invest within_mvalue within_kstock if in_fit, noconstant}{p_end}

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:xfepout} stores the following in {cmd:e()}:

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Scalars}{p_end}
{synopt:{cmd:e(df_a)}}absorbed DoF after configured adjustments{p_end}
{synopt:{cmd:e(df_a_levels)}}total FE levels before redundancy adjustments{p_end}
{synopt:{cmd:e(df_a_exact)}}exact absorbed DoF count used in reporting{p_end}
{synopt:{cmd:e(df_a_nested)}}DoF nested in clustering dimensions{p_end}
{synopt:{cmd:e(N_hdfe)}}number of absorbed FE dimensions{p_end}
{synopt:{cmd:e(N)}}kept observations after singleton handling{p_end}
{synopt:{cmd:e(N_full)}}candidate observations before singleton handling{p_end}
{synopt:{cmd:e(num_singletons)}}number of dropped singleton observations{p_end}
{synopt:{cmd:e(iterations)}}absorption iterations used by the backend{p_end}
{synopt:{cmd:e(converged)}}1 if converged, 0 otherwise{p_end}
{synopt:{cmd:e(threads_requested)}}raw request; 0 denotes automatic selection{p_end}
{synopt:{cmd:e(threads_effective)}}request after runtime/automatic limits{p_end}
{synopt:{cmd:e(threads_used)}}largest OpenMP team observed executing real work{p_end}
{synopt:{cmd:e(parallel_workers_active)}}largest number of workers doing useful work in one observed region{p_end}
{synopt:{cmd:e(thread_capacity)}}runtime-visible logical processor capacity{p_end}
{synopt:{cmd:e(openmp_enabled)}}1 when the plugin has OpenMP support{p_end}
{synopt:{cmd:e(thread_limit_code)}}0 none, 1 runtime capacity, 2 auto policy, 3 auto maxthreads, 4 OpenMP unavailable,
5 automatic request made inside an active OpenMP region, 6 automatic team shared with the same user's
simultaneous fits; its label is the macro {cmd:e(thread_limit_reason)}{p_end}
{synopt:{cmd:e(gpu_used)}}1 if GPU backend was effectively used, 0 if CPU path was used{p_end}
{synopt:{cmd:e(gpu_status_code)}}GPU status code: 0 not requested, 1 used, 2 unavailable, 3 not converged, 4 failed, 5 CPU cache/profile result{p_end}
{synopt:{cmd:e(gpu_attempted)}}1 if GPU absorption was attempted, 0 otherwise{p_end}
{synopt:{cmd:e(gpu_absorption_converged)}}1 if attempted GPU absorption converged, 0 if not, missing if not attempted{p_end}
{synopt:{cmd:e(gpu_absorption_iterations)}}GPU absorption iterations, missing if GPU absorption was not attempted{p_end}
{synopt:{cmd:e(cuda_tail_code)}}CUDA completion: 0 none reported, 1 device LSMR, 2 host MLSMR (see {help xhdfe##opt_gpu:backend selection}){p_end}
{synopt:{cmd:e(method_used)}}method code: 0 auto, 1 Gauss-Seidel, 2 symmetric Gauss-Seidel,
3 Jacobi, 4 Schwarz/CG, 5 LSMR, 6 MLSMR, 7 Auto-MLSMR, 9 Krylov-PCG
(a profile-selected route). Internal direct methods can report the chosen sweep code.{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Macros}{p_end}
{synopt:{cmd:e(absvars)}}canonical FE specification used by {cmd:absorb()}{p_end}
{synopt:{cmd:e(extended_absvars)}}alias of {cmd:e(absvars)}{p_end}
{synopt:{cmd:e(absorption_method)}}reported absorption method label; see {cmd:e(method_used)}{p_end}
{synopt:{cmd:e(gpu_backend_requested)}}requested backend from {opt gpubackend()}, when provided{p_end}
{synopt:{cmd:e(gpu_backend)}}effective backend when {opt gpubackend()} was specified{p_end}
{synopt:{cmd:e(gpu_status)}}GPU status label: {cmd:not_requested}, {cmd:used},
{cmd:backend_unavailable}, {cmd:gpu_absorption_not_converged},
{cmd:gpu_backend_failed}, or {cmd:cpu_cache_or_profile_result}{p_end}
{synopt:{cmd:e(cuda_tail)}}label of {cmd:e(cuda_tail_code)}: {cmd:none}, {cmd:device-lsmr}, or {cmd:host-mlsmr}{p_end}
{synopt:{cmd:e(thread_limit_reason)}}label of {cmd:e(thread_limit_code)}{p_end}
{synopt:{cmd:e(version)}}xfepout version string{p_end}
{synopt:{cmd:e(cmd)}}{cmd:xfepout}{p_end}
{synopt:{cmd:e(cmdline)}}command line used to invoke xfepout{p_end}

{p2colreset}{...}
{p 4 6 2}{cmd:xfepout} does not post {cmd:e(b)}, {cmd:e(V)}, or {cmd:e(sample)}. It is a partialling-out
command, not a regression estimator.{p_end}
