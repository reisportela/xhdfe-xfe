{smcl}
{* *! version 1.0.0  28jul2026}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{vieweralsosee "xhdfegelbachetable" "help xhdfegelbachetable"}{...}
{vieweralsosee "xhdfegelbachcoefplot" "help xhdfegelbachcoefplot"}{...}
{title:Title}

{p2colset 5 30 32 2}{...}
{p2col :{cmd:xhdfegelbachbootstrap} {hline 2}}Full-refit pairs bootstrap for Gelbach coefficient-movement accounting{p_end}
{p2colreset}{...}

{pstd}{bf:Version 1.0.0 (28jul2026), distributed with xhdfegelbach 1.5.0.}{p_end}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfegelbachbootstrap} {depvar} {ifin} {weight}{cmd:,}
{opt x1(varlist)}
{opt method(pairs|cluster_pairs)}
[{it:options}]

{synoptset 30 tabbed}{...}
{synopthdr}
{synoptline}
{synopt :{opt x2groups(spec)}}observed blocks added in the full model{p_end}
{synopt :{opth commonfes(varlist)}}FEs absorbed in base and full{p_end}
{synopt :{opth fes(varlist)}}FEs added only in the full model{p_end}
{synopt :{opth bootcluster(varname)}}explicit resampling unit for {cmd:cluster_pairs}{p_end}
{synopt :{opt reps(#)}}attempted replications; default 999, minimum 2{p_end}
{synopt :{opt seed(#)}}nonnegative reproducibility seed; default 0{p_end}
{synopt :{opt minvalid(#)}}minimum successful refits; default ceil(0.9*reps){p_end}
{synopt :{opt bootci(type)}}{cmd:percentile} (default) or {cmd:basic}{p_end}
{synopt :{opt level(#)}}bootstrap confidence level; default {cmd:c(level)}{p_end}
{synopt :{opt requiregpu}}fail if requested CUDA is not really used{p_end}
{synopt :{opt sharetol(#)}}share-denominator threshold; default 1e-12{p_end}
{synoptline}
{p2colreset}{...}

{pstd}
The following {help xhdfegelbach} point-estimate options are accepted:
{opt vce()}, {opt cluster()}, {opt gamma0}, {opt cov0},
{opt absorbedtargets()}, {opt focal()}, {opt tol()}, {opt threads()},
{opt gpu}, {opt verbose}, {opt connected()} and {opt connectivityfes()}.
The reporting-only {opt shares()}, {opt sharetmin()}, {opt sampleaudit} and
{opt generate()} options are not bootstrap arguments; construct shares and
tables from the returned draw/interval objects.
The point VCE and the bootstrap unit are separate choices.{p_end}


{title:Description}

{pstd}
{cmd:method(pairs)} samples observations with replacement.
{cmd:method(cluster_pairs)} samples the blocks declared in
{opt bootcluster()}; the unit is never inferred from {opt vce(cluster)} or
{opt cluster()}.{p_end}

{pstd}
Every accepted replication resamples the outcome, X1, every observed block,
common/added FE identifiers and analytic weights together. It then calls the
same public {cmd:xhdfegelbach} estimator from scratch. Base/full fitting,
singleton removal, auxiliary projections, FE recovery, connectivity
diagnostics and the accounting identity are all recomputed. The point result
keeps the requested VCE. Replications use unadjusted VCE because only their
re-estimated point functionals enter the empirical bootstrap distribution.{p_end}

{pstd}
The caller's Stata RNG state is restored on success and on a fail-closed
completion. Repeating a call with the same seed and estimator build reproduces
the draw matrices exactly. Python uses PCG64 child streams and R uses
L'Ecuyer-CMRG streams, so the same integer seed is reproducible within each
frontend but does not imply identical resamples across frontends. Every
attempted draw is retained in
{cmd:r(bootstrap_ledger)} with validity, return code, drawn/retained rows,
singletons, identity gap and GPU use. Fewer than {opt minvalid()} successful
refits is an error.{p_end}

{pstd}
Frequency-weight bootstrap is refused because resampling a compressed row is
not the same empirical distribution as resampling its expanded copies.
Expand those observations first. Analytic weights remain attached to sampled
observations.{p_end}

{pstd}{bf:Inference warning.} Percentile/basic bootstrap intervals do not by
themselves cure product nonregularity, few-cluster inference, weak
identification or a normalization-dependent FE split. Existing
{cmd:xhdfegelbach} diagnostics remain authoritative. They are nevertheless
the calibrated alternative exposed by the package when
{cmd:r(share_interval_status)} reports
{cmd:weak_denominator_delta_method_unreliable} or
{cmd:r(fe_variance_status)} reports
{cmd:conditional_only_between_fe_dominant}. The companion
{helpb xhdfegelbachetable} consumes component intervals from
{cmd:r(bootstrap_delta_ci)} as well as endpoint and share intervals, and
labels their source explicitly. This is specification accounting, not causal
mediation.{p_end}


{title:Stored results}

{pstd}
The command returns the ordinary point matrices {cmd:r(delta)},
{cmd:r(se)}, {cmd:r(total)}, {cmd:r(b_base)} and {cmd:r(b_full)}; the joint
point covariance contract {cmd:r(cov)}, {cmd:r(total_cov)},
{cmd:r(base_cov)}, {cmd:r(cov_delta_bbase)} and
{cmd:r(cov_total_bbase)}; plus:{p_end}

{synoptset 38 tabbed}{...}
{synopt:{cmd:r(bootstrap_delta_draws)}}group-major contribution draws{p_end}
{synopt:{cmd:r(bootstrap_total_draws)}}total-movement draws{p_end}
{synopt:{cmd:r(bootstrap_base_draws)}}base-coefficient draws{p_end}
{synopt:{cmd:r(bootstrap_full_draws)}}full-coefficient draws{p_end}
{synopt:{cmd:r(bootstrap_share_base_draws)}}component/base draws{p_end}
{synopt:{cmd:r(bootstrap_share_movement_draws)}}component/movement draws{p_end}
{synopt:{cmd:r(bootstrap_full_share_base_draws)}}full/base draws{p_end}
{synopt:{cmd:r(bootstrap_total_share_base_draws)}}movement/base draws{p_end}
{synopt:{cmd:r(bootstrap_*_ci)}}four rows: low, high, bootstrap SE, valid count{p_end}
{synopt:{cmd:r(bootstrap_ledger)}}one row per attempt: valid, rc, n_rows, n_obs, n_singletons_dropped, identity_gap, gpu_used{p_end}
{synopt:{cmd:r(reps_requested)}, {cmd:r(reps_valid)}, {cmd:r(reps_failed)}}replication counts{p_end}
{synopt:{cmd:r(seed)}, {cmd:r(min_valid_reps)}}reproducibility seed and fail-closed valid-replication threshold{p_end}
{synopt:{cmd:r(method)}, {cmd:r(resampling_unit)}}bootstrap design{p_end}
{synopt:{cmd:r(bootstrap_cluster)}}declared cluster variable, empty for observation pairs{p_end}
{synopt:{cmd:r(ci_method)}, {cmd:r(conf_level)}}interval metadata{p_end}
{synopt:{cmd:r(share_tol)}}absolute denominator threshold used for stored bootstrap shares{p_end}
{synopt:{cmd:r(point_vce)}, {cmd:r(replicate_vce)}}point/refit variance contracts{p_end}
{synopt:{cmd:r(gpu_requested)}, {cmd:r(gpu_required)}}CUDA request/enforcement flags{p_end}
{synopt:{cmd:r(gpu_used_point)}}real CUDA use in the point estimate{p_end}
{synopt:{cmd:r(gpu_used_all_valid)}}real CUDA use in every valid refit{p_end}
{synopt:{cmd:r(point_gpu_backend)}, {cmd:r(point_gpu_status)}}point backend diagnostics{p_end}
{synopt:{cmd:r(interval_status)}}nonregularity qualification{p_end}
{synopt:{cmd:r(rng)}}Stata RNG reproducibility convention{p_end}
{synopt:{cmd:r(version)}}bootstrap-command version and date{p_end}
{synopt:{cmd:r(identity_status)}}point-decomposition accounting identity{p_end}
{synopt:{cmd:r(groups)}, {cmd:r(x1_names)}, {cmd:r(focal_names)}}component and coefficient ordering metadata{p_end}
{synopt:{cmd:r(notes)}}point-estimator warnings and qualifications, when nonempty{p_end}


{title:Examples}

{phang2}{cmd:. xhdfegelbachbootstrap lwage, x1(educ age) focal(educ) x2groups("human = ability : job = tenure exper") commonfes(year) fes(firm) method(pairs) reps(499) seed(42)}{p_end}
{phang2}{cmd:. xhdfegelbachetable, panels(all) format(markdown) saving(gelbach.md) replace}{p_end}

{phang2}{cmd:. xhdfegelbachbootstrap lwage, x1(educ) x2groups("job = tenure exper") fes(firm) method(cluster_pairs) bootcluster(worker) reps(499) seed(42) bootci(basic)}{p_end}
{phang2}{cmd:. matrix list r(bootstrap_ledger)}{p_end}


{title:Also see}

{psee}
{helpb xhdfegelbach}, {helpb xhdfegelbachetable},
{helpb xhdfegelbachcoefplot}
{p_end}
