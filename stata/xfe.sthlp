{smcl}
{* *! version 1.10.0 09jul2026}{...}
{vieweralsosee "hdfe" "help hdfe"}{...}
{vieweralsosee "reghdfe" "help reghdfe"}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{viewerjumpto "Development process" "xfe##development"}{...}
{viewerjumpto "Syntax" "xfe##syntax"}{...}
{viewerjumpto "Absvar Syntax" "xfe##absvar"}{...}
{viewerjumpto "Options" "xfe##options"}{...}
{viewerjumpto "Stored results" "xfe##results"}{...}
{title:Title}

{p2colset 5 18 20 2}{...}
{p2col :{cmd:xfe} {hline 2}}Partial-out variables with respect to multiple levels of fixed effects via the xhdfe C++ absorption core{p_end}
{p2colreset}{...}

{pstd}{cmd:xfe} is a separate package from {cmd:xhdfe} and ships its own plugin binary ({cmd:xfe.plugin}). It is a partialling-out command (no coefficient table).{p_end}

{marker development}{...}
{title:Development process}

{pstd}
{cmd:xfe} shares the AI-assisted development workflow used for {cmd:xhdfe}.
The project used AI coding tools to study existing HDFE packages, review relevant
literature, inspect the Stata/C++ codebase, draft and revise patches, instrument
slow paths, and improve absorption performance. No AI tool or system is named or
credited as a contributor.{p_end}

{pstd}
The listed human authors did not write the package line by line. Their role was
to define the econometric target, set numerical and performance guardrails,
choose the benchmark and validation criteria, direct the optimization process,
decide which AI-assisted changes to keep, and retain responsibility for the
package. No software tool or AI system is credited as an author or co-author.
{p_end}

{marker syntax}{...}
{title:Syntax}

{pstd}Replace current dataset:

{p 8 15 2}
{cmd:xfe}
{it:varlist}
{ifin}
{it:{weight}}
{cmd:,}
{opth a:bsorb(xfe##absvar:absvars)}
{opt clear}
[{opth keepv:ars(varlist)} {opt keepid:s}]
[{opth clusterv:ars(varlist)} {help xfe##options:options}]
{p_end}

{pstd}Keep current dataset and add transformed variables:

{p 8 15 2}
{cmd:xfe}
{it:varlist}
{ifin}
{it:{weight}}
{cmd:,}
{opth a:bsorb(xfe##absvar:absvars)}
{opt g:enerate(stubname)}
[{opt sample(newvarname)}]
[{opth clusterv:ars(varlist)} {help xfe##options:options}]
{p_end}

{pstd}Show package version:

{p 8 15 2}
{cmd:xfe, version}
{p_end}

{p 4 6 2}{bf:Important:} {it:varlist} must contain existing numeric variables (no factor-variable or time-series operators in the varlist).{p_end}
{p 4 6 2}{opt absorb(absvars)} is required.{p_end}
{p 4 6 2}Specify exactly one of {cmd:clear} or {cmd:generate()}. {cmd:sample()} is only allowed with
{cmd:generate()}, while {cmd:keepvars()} and {cmd:keepids} are only allowed with {cmd:clear}.{p_end}
{p 4 6 2}{cmd:fweight}s, {cmd:aweight}s, {cmd:pweight}s, and {cmd:iweight}s are allowed; see {help weight}.{p_end}

{marker absvar}{...}
{title:Absvar Syntax}

{pstd}{cmd:absorb()} and {cmd:clustervars()} accept categorical identifiers and two-way interactions
with {cmd:#}/{cmd:##}. For reghdfe/xhdfe compatibility, categorical-only {cmd:##} terms in
{cmd:absorb()} are treated as one joint absorbed FE, not expanded into main effects plus the interaction.
In {cmd:clustervars()}, {cmd:##} expands to the two main cluster dimensions plus their interaction.
The grammar is intentionally narrower than {cmd:xhdfe}: these options are for fixed-effect or cluster
identifiers, not for heterogeneous slopes.{p_end}

{p 4 6 2}- Interactions must have exactly two components and are materialized with {cmd:egen group()}
after the estimation sample is defined. Plain numeric identifiers are passed directly to the plugin;
string identifiers are encoded by Stata before the plugin call.{p_end}
{p 4 6 2}- Factor-style prefixes such as {cmd:i.}, {cmd:ib#.}, {cmd:bn.}, and {cmd:ibn.} are stripped
for compatibility before the underlying variable names are resolved.{p_end}
{p 4 6 2}- Prefix {cmd:c.} and heterogeneous-slope syntax such as {cmd:var#c.x} or {cmd:var##c.x}
are rejected. They are not interpreted as categorical identifiers in {cmd:xfe}.{p_end}
{p 4 6 2}- {cmd:absorb()} suboptions such as {cmd:savefe} are not supported in {cmd:xfe}.{p_end}

{marker options}{...}
{title:Options}

{dlgtab:HDFE-specific}
{phang}{opt clear} overwrite the current dataset with the kept estimation sample, replacing {it:varlist}
by its residualized version. The resulting dataset keeps the transformed variables plus absorbed/clustering
variables, weights, optional {cmd:groupvar()}, anything requested in {cmd:keepvars()}, and any
materialized FE/cluster ids requested through {cmd:keepids}.{p_end}
{phang}{opt generate(stub)} keep the current dataset and create new double-precision transformed variables
named {it:stub}{it:varname}. Each new variable is labeled {cmd:Residuals: <source label>}.{p_end}
{phang}{opt sample(newvar)} save a sample marker (1=kept, 0=dropped/outside sample). Only with
{opt generate()}; the generated variable is labeled {cmd:[XFE Sample]}.{p_end}
{phang}{opt keepvars(varlist)} in {opt clear} mode, keep additional variables in the rewritten dataset.{p_end}
{phang}{opt keepids} in {opt clear} mode, keep the materialized integer FE/cluster ids created for the
plugin (for example {cmd:__xfe_ID*__} and {cmd:__xfe_CL*__}).{p_end}
{phang}{opt clustervars(varlist)} cluster identifiers used only for DoF bookkeeping and mobility
adjustments in {cmd:e(df_a)}. {cmd:xfe} does not estimate clustered standard errors.
{cmd:cl()} and {cmd:cluster()} are accepted as aliases.{p_end}

{dlgtab:Optimization}
{phang}{opt tolerance(#)} absorber tolerance (default 1e-8; valid range (0,1)). Since version 1.9.0 the absorber stops when one full sweep moves the working data by less than this value in relative norm (reghdfe-comparable semantics, matching the default tolerancemode used by {cmd:xhdfe}).{p_end}
{phang}{opt maxiter(#)} maximum absorption iterations (default 100000).{p_end}
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
{pmore}Default is {it:symmetric-gauss-seidel}. The LSMR/MLSMR family is CPU-only and
applies to standard categorical fixed effects; {cmd:xfe} rejects heterogeneous-slope syntax before the plugin call.{p_end}
{phang}{opt symmetricsweep} force forward/backward sweeps per iteration.{p_end}
{phang}{opt jacobirelaxation(#)} Jacobi relaxation parameter (default 0).{p_end}
{phang}{opt keepsingletons} do not drop singleton observations.{p_end}

{dlgtab:Threading}
{phang}{opt numthreads(#)} direct absorber thread count (0=auto).{p_end}
{phang}{opt defaultthreads(#)} default auto-thread count when {opt numthreads(0)}.{p_end}
{phang}{opt maxthreads(#)} cap for auto-thread selection (0=no cap).{p_end}
{phang}{opt minparallelrows(#)} minimum rows before parallelization is considered.{p_end}
{phang}{opt targetrowsperthread(#)} target row budget per thread in auto-threading.{p_end}

{dlgtab:GPU}
{phang}{opt gpubackend(str)} backend selector: {it:cpu}, {it:cuda}, or {it:metal}.
CUDA requires a CUDA-enabled plugin build and a CUDA device; {it:metal} is reserved for
Metal-enabled builds. If a requested non-CPU backend is unavailable, {cmd:xfe} exits
with an error instead of silently falling back to CPU. For CUDA builds, use
{cmd:XHDFE_CUDA_ARCH} (single SM target) or {cmd:XHDFE_CUDA_ARCHS} (multi-target list)
when running {cmd:bash stata/tools/build-xfe-plugin.sh}. If the plugin was rebuilt
or switched in the current session, run {cmd:discard} (with no arguments) before retrying.{p_end}

{phang}On this workstation (H100), rebuild with
{cmd:XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash stata/tools/build-xfe-plugin.sh --linux --openmp}.{p_end}

{phang}After running {cmd:xfe, ... gpubackend(cuda)}, verify {cmd:e(gpu_used)},
{cmd:e(gpu_backend)}, and {cmd:e(gpu_status)}. If CUDA is unavailable, fails before a
converged absorption result, or returns a CPU cache/profile result, {cmd:xfe} exits
with error 498 instead of silently returning CPU output. If the plugin was rebuilt
or switched during the current Stata session, run {cmd:discard} (with no arguments)
before rerunning the command.{p_end}

{phang}Practical rule: after rebuilding {cmd:xfe.plugin}, after switching between CPU and
CUDA plugin binaries, or after changing the active {cmd:adopath} entry that provides
{cmd:xfe.ado}/{cmd:xfe.plugin}, run {cmd:discard} (not {cmd:discard xfe}) before
rerunning {cmd:xfe}.{p_end}

{phang}Ordinary repeated calls in the same Stata session do {bf:not} require {cmd:discard}
when the plugin binary itself has not changed; for example, rerunning {cmd:xfe} first
on CPU and then with {cmd:gpubackend(cuda)} now reuses the same loaded plugin correctly.{p_end}

{dlgtab:Cache and Profile}
{phang}{opt mobilityprofile} write/update mobility profile at default path {cmd:xfe_mobility_profile.txt}.{p_end}
{phang}{opt mobfile(path)} mobility profile path; enables profile auto mode at that path.{p_end}
{phang}{opt absorptioncache(path)} path for absorption cache payload.{p_end}
{phang}{opt abscachemode(mode)} absorption cache mode: {it:off}, {it:auto}, {it:read}, {it:write}.
{cmd:absorptioncachemode()} is accepted as an alias.{p_end}
{phang}{opt fescache(path)} path for FE-structure cache payload.{p_end}
{phang}{opt festructurecache} use default FE-structure cache path relative to {cmd:xfe.ado}.
{cmd:festructurecache()} and {cmd:festructurecachefile()} are accepted as aliases for
{cmd:fescache()}.{p_end}
{phang}{opt fecachemode(mode)} FE-structure cache mode: {it:off}, {it:auto}, {it:read}, {it:write}.
{cmd:fescachemode()} and {cmd:festructurecachemode()} are accepted as aliases.{p_end}

{dlgtab:DoF}
{phang}{opt dofadjustments(list)} reghdfe-style DoF adjustments (tokens include {it:all},
{it:none}, {it:firstpair}/{it:first}, {it:pairwise}/{it:pair}, {it:clusters}/{it:cluster},
{it:continuous}/{it:cont}). {cmd:dof()} is accepted as an alias.{p_end}
{phang}{opt groupvar(newvar)} save the first mobility-group id (connected component; 1-based) as a new variable when the backend computes the relevant mobility structure.{p_end}

{dlgtab:Compatibility}
{phang}{opt maxiterations(#)} alias for {opt maxiter(#)}.{p_end}
{phang}{opt iterate(#)}, {opt iterations(#)}, and {opt iter(#)} are aliases for {opt maxiter(#)}.{p_end}
{phang}{opt a(...)}/{opt abs(...)} are aliases for {opt absorb(...)}; {opt groupv()} is an alias
for {opt groupvar()}.{p_end}
{phang}{opt nowarn} suppress singleton warning messages; transformation is unchanged.{p_end}
{phang}{opt poolsize(#)} accepted for hdfe compatibility; currently ignored by {cmd:xfe}.{p_end}
{phang}{opt acceleration(str)} accepted for hdfe compatibility; currently ignored by {cmd:xfe}.{p_end}
{phang}{opt prune} accepted for hdfe compatibility; currently ignored by {cmd:xfe}.{p_end}
{phang}{opt verbose(#)} accepted for hdfe compatibility; currently ignored by {cmd:xfe}.{p_end}
{phang}{opt timeit} print plugin wall-clock time from the ado layer.{p_end}

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:xfe} stores the following in {cmd:e()}:

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
{synopt:{cmd:e(threads_used)}}threads used by the backend{p_end}
{synopt:{cmd:e(gpu_used)}}1 if GPU backend was effectively used, 0 if CPU path was used{p_end}
{synopt:{cmd:e(gpu_status_code)}}GPU status code: 0 not requested, 1 used, 2 unavailable, 3 not converged, 4 failed, 5 CPU cache/profile result{p_end}
{synopt:{cmd:e(gpu_attempted)}}1 if GPU absorption was attempted, 0 otherwise{p_end}
{synopt:{cmd:e(gpu_absorption_converged)}}1 if attempted GPU absorption converged, 0 if not, missing if not attempted{p_end}
{synopt:{cmd:e(gpu_absorption_iterations)}}GPU absorption iterations, missing if GPU absorption was not attempted{p_end}
{synopt:{cmd:e(method_used)}}numeric method code: 0 auto, 1 gauss-seidel, 2 symmetric-gauss-seidel, 3 jacobi, 4 Schwarz/CG, 5 LSMR, 6 MLSMR, 7 auto-MLSMR{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Macros}{p_end}
{synopt:{cmd:e(absvars)}}canonical FE specification used by {cmd:absorb()}{p_end}
{synopt:{cmd:e(extended_absvars)}}alias of {cmd:e(absvars)}{p_end}
{synopt:{cmd:e(absorption_method)}}resolved sweep-method label; inspect {cmd:e(method_used)} for the full method enum{p_end}
{synopt:{cmd:e(gpu_backend_requested)}}requested backend from {opt gpubackend()}, when provided{p_end}
{synopt:{cmd:e(gpu_backend)}}effective backend ({cmd:cpu}, {cmd:cuda}, or {cmd:metal}) when the command completed successfully{p_end}
{synopt:{cmd:e(gpu_status)}}GPU status label: {cmd:not_requested}, {cmd:used}, {cmd:backend_unavailable}, {cmd:gpu_absorption_not_converged}, {cmd:gpu_backend_failed}, or {cmd:cpu_cache_or_profile_result}{p_end}
{synopt:{cmd:e(version)}}xfe version string{p_end}
{synopt:{cmd:e(cmd)}}{cmd:xfe}{p_end}
{synopt:{cmd:e(cmdline)}}command line used to invoke xfe{p_end}

{p2colreset}{...}
{p 4 6 2}{cmd:xfe} does not post {cmd:e(b)}, {cmd:e(V)}, or {cmd:e(sample)}. It is a partialling-out
command, not a regression estimator.{p_end}
