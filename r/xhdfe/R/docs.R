# Documentation-only file: roxygen2 blocks for the xhdfe package.
#
# This file intentionally contains NO functional code. Every topic is
# documented through the `NULL` + `@name` pattern so that the generated
# man/*.Rd pages mirror, section for section, the Stata help file
# stata/xhdfe.sthlp (version 2.11.0). The NAMESPACE is maintained by hand;
# no @export tags appear here.

# ---------------------------------------------------------------------------
# Package overview
# ---------------------------------------------------------------------------

#' xhdfe: Linear Regression with Multiple High-Dimensional Fixed Effects via a C++ Backend
#'
#' The \pkg{xhdfe} package estimates linear models with multiple
#' high-dimensional fixed effects through a compiled C++ backend -- the very
#' same estimator core (version 2.11.0) behind the Stata \code{xhdfe} command
#' and the Python \code{xhdfe} package. CPU behavior is the reference
#' implementation; the optional CUDA GPU absorber is validated against it and
#' never silently replaces it. Where possible the package mirrors
#' \code{reghdfe}'s reporting conventions and defaults, while also exposing
#' \pkg{fixest}-style small-sample corrections and design ideas drawn from
#' \code{pyfixest} and \code{FixedEffectModels.jl}. Under the default
#' \code{"reghdfe-comparable"} tolerance mode, coefficients match
#' \code{reghdfe} at the same nominal tolerance (down to problem
#' conditioning).
#'
#' @section Features:
#' \itemize{
#'   \item A fast fixed-effect absorber with several iterative methods
#'     (Gauss-Seidel, symmetric Gauss-Seidel, Jacobi, additive-Schwarz /
#'     preconditioned conjugate-gradient, LSMR, and modified-LSMR/within
#'     solvers) plus OpenMP multithreading, with an automatic method
#'     selector.
#'   \item Multiway (two-or-more-way) fixed effects, including combined
#'     fixed effects (\code{f1^f2}) and heterogeneous (group-specific)
#'     slopes (\code{fe[x]}, \code{fe[[x]]}).
#'   \item Unadjusted, heteroskedasticity-robust (HC1) and multiway
#'     clustered standard errors with cluster diagnostics, including the
#'     Cameron-Gelbach-Miller positive-semi-definite adjustment used by
#'     \code{reghdfe}.
#'   \item Instrumental variables / two-stage least squares (2SLS) for
#'     supported standard fixed-effect designs.
#'   \item Analytic and frequency weights (Stata \code{aweight} /
#'     \code{fweight} semantics).
#'   \item \code{reghdfe}-style singleton dropping and degrees-of-freedom
#'     adjustments, plus \pkg{fixest}-style small-sample corrections
#'     (\code{ssc}).
#'   \item Fixed-effect recovery (\code{save_fe = TRUE}, the analogue of
#'     Stata's \code{savefe}), retrieved with \code{fixef()}.
#'   \item Group-level outcomes with individual fixed effects
#'     (\code{group} / \code{individual} / \code{aggregation}), matching
#'     the Stata \code{group()} / \code{individual()} machinery.
#'   \item An optional CUDA GPU absorber with a fail-closed contract:
#'     a requested GPU backend that cannot actually be used raises an
#'     error instead of silently falling back to the CPU.
#' }
#'
#' @section Development process:
#' The package was developed through an AI-assisted optimization workflow
#' (OpenAI Codex, Anthropic Claude Code, and some Google Gemini Code Assist)
#' used to study existing high-dimensional fixed-effects packages, review the
#' literature, inspect the mixed Stata/C++/Python/R codebase, draft and
#' revise patches, instrument slow paths, and improve performance. The human
#' authors did not write the package line by line; they defined the
#' econometric target, set numerical and performance guardrails, chose the
#' benchmark and validation criteria, directed the optimization work, decided
#' which changes to keep, and retain full responsibility for the package. No
#' software tool or AI system is credited as author or co-author.
#'
#' @section The xhdfe ecosystem:
#' \pkg{xhdfe} is built to validate against and interoperate with the
#' established high-dimensional fixed-effects estimators:
#' \itemize{
#'   \item xhdfe (Stata, Python, R): \url{https://github.com/reisportela/xhdfe-xfe}
#'   \item reghdfe (Stata, Sergio Correia):
#'     \url{https://github.com/sergiocorreia/reghdfe}
#'   \item fixest (R, Laurent Berge): \url{https://github.com/lrberge/fixest}
#'   \item pyfixest (Python, Alexander Fischer and collaborators):
#'     \url{https://github.com/py-econometrics/pyfixest}
#'   \item FixedEffectModels.jl (Julia, Matthieu Gomez and collaborators):
#'     \url{https://github.com/FixedEffects/FixedEffectModels.jl}
#' }
#'
#' @section Authors:
#' \itemize{
#'   \item Miguel Portela -- NIPE / Universidade do Minho and BPLIM / Banco
#'     de Portugal -- \email{miguel.portela@@eeg.uminho.pt} --
#'     \url{https://reisportela.github.io}
#'   \item Tiago Tavares -- NIPE / Universidade do Minho --
#'     \email{tgstavares@@eeg.uminho.pt} -- \url{https://www.tgstavares.com}
#' }
#' Only the listed human authors are authors or co-authors of this package;
#' no software tool or AI system is credited as author or co-author.
#'
#' @section Acknowledgements:
#' \pkg{xhdfe} is a reimplementation built to validate against and
#' interoperate with \code{reghdfe} (Sergio Correia), \pkg{fixest} (Laurent
#' Berge), \code{pyfixest} (Alexander Fischer and collaborators), and
#' \code{FixedEffectModels.jl} (Matthieu Gomez and collaborators). The
#' authors thank Paulo Guimaraes (Banco de Portugal), Marta Silva (Banco de
#' Portugal), and Nelson Areal (EEG/Universidade do Minho). An earlier proof
#' of concept was shared at BPLIM's "Speeding Up Empirical Research"
#' workshop, in the hands-on session "Parallel and Cross-Language Computing:
#' A Hands-On Workshop for Empirical Researchers" presented by Nelson Areal
#' and Miguel Portela (BPLIM 2025). Financial support from Universidade do
#' Minho, Banco de Portugal, and FCT (UID/03182/2025) is gratefully
#' acknowledged. The usual disclaimer applies.
#'
#' @references
#' Portela, Miguel, and Tiago Tavares. 2026. "xhdfe: High-dimensional fixed
#' effects regression via a C++ backend." Version 2.18.2.
#' \url{https://github.com/reisportela/xhdfe-xfe}
#'
#' Cornelissen, Thomas. 2008. "The Stata command felsdvreg to fit a linear
#' model with two high-dimensional fixed effects." Stata Journal 8(2):
#' 170-189. \doi{10.1177/1536867X0800800202}
#'
#' Guimaraes, Paulo, and Pedro Portugal. 2010. "A simple feasible procedure
#' to fit models with high-dimensional fixed effects." Stata Journal 10(4):
#' 628-649. \doi{10.1177/1536867X1101000406}
#'
#' Gaure, Simen. 2013. "OLS with multiple high dimensional category
#' variables." Computational Statistics & Data Analysis 66: 8-18.
#' \doi{10.1016/j.csda.2013.03.024}
#'
#' Correia, Sergio. 2016. "reghdfe: Estimating linear models with multi-way
#' fixed effects." 2016 Stata Conference. RePEc:boc:scon16:24.
#'
#' Correia, Sergio, Paulo Guimaraes, and Tom Zylkin. 2020. "Fast Poisson
#' estimation with high-dimensional fixed effects." Stata Journal 20(1):
#' 95-115. \doi{10.1177/1536867X20909691}
#'
#' Berge, Laurent, Kyle Butts, and Grant McDermott. 2026. "Fast and
#' user-friendly econometrics estimations: The R package fixest."
#' arXiv:2601.21749.
#'
#' Fischer, Alexander, and collaborators. "pyfixest: Fast high-dimensional
#' fixed effect estimation in Python." Python package.
#' \url{https://github.com/py-econometrics/pyfixest}
#'
#' FixedEffects.jl contributors. 2025. "FixedEffectModels.jl: Linear models
#' with instrumental variables and high dimensional categorical variables."
#' Julia package. \url{https://github.com/FixedEffects/FixedEffectModels.jl}
#'
#' @seealso \code{\link{xhdfe}} for the formula interface,
#'   \code{\link{xhdfe_fit}} for the matrix interface,
#'   \code{xhdfe_demean()} for stand-alone demeaning,
#'   \code{xhdfe_group_fes()} for group-level fixed-effect aggregation, and
#'   \code{xhdfe_info()} for version and build information.
#'
#' @name xhdfe-package
#' @aliases xhdfe-package
NULL

# ---------------------------------------------------------------------------
# Main estimator: xhdfe()
# ---------------------------------------------------------------------------

#' Linear regression with multiple high-dimensional fixed effects (C++ backend)
#'
#' @description
#' \code{xhdfe()} estimates linear models with any number of high-dimensional
#' fixed effects through the xhdfe C++ backend -- the same compiled estimator
#' behind the Stata \code{xhdfe} command and the Python \code{xhdfe} package.
#' It mirrors \code{reghdfe}'s reporting conventions and defaults where
#' possible (singleton dropping, degrees-of-freedom accounting, multiway
#' clustered inference, constant reporting) while also exposing
#' \pkg{fixest}-style small-sample corrections.
#'
#' The Stata command's six syntax forms translate to R as follows:
#' \itemize{
#'   \item Least-squares regression without fixed effects:
#'     \code{xhdfe(y ~ x1 + x2, data = df)}.
#'   \item Regression with absorbed fixed effects (Stata
#'     \code{absorb(fe1 fe2)}): \code{xhdfe(y ~ x1 + x2 | fe1 + fe2, data = df)}.
#'   \item Group-level outcomes (Stata \code{group(gid)}):
#'     \code{xhdfe(y ~ x, data = df, group = ~ gid)}, optionally with a
#'     fixed-effects part.
#'   \item Group-level outcomes with individual fixed effects (Stata
#'     \code{group(gid) individual(iid)}):
#'     \code{xhdfe(y ~ x | iid, data = df, group = ~ gid, individual = ~ iid)}.
#'   \item Instrumental variables / 2SLS (Stata \code{endogenous()} +
#'     \code{instruments()}):
#'     \code{xhdfe(y ~ exo | fe1 + fe2 | endo ~ inst, data = df)}; without
#'     absorbed fixed effects, \code{xhdfe(y ~ exo | endo ~ inst, data = df)}.
#'   \item Version information (Stata \code{xhdfe, version}):
#'     \code{xhdfe_info()}.
#' }
#'
#' With absorbed fixed effects a \code{reghdfe}-compatible constant
#' \code{"(Intercept)"} is reported after the regressors; remove it with
#' \code{0 +} in the regressor part (the analogue of Stata's
#' \code{noconstant}).
#'
#' Standard S3 methods are available on the returned object:
#' \code{print}, \code{summary}, \code{coef}, \code{vcov}, \code{confint},
#' \code{nobs}, \code{df.residual}, \code{residuals}, \code{fitted},
#' \code{predict} (with Stata's \code{xb}, \code{xbd}, \code{d},
#' \code{residuals}, \code{dresiduals}, \code{score}, and \code{stdp}
#' statistics), \code{logLik}, \code{deviance}, \code{formula},
#' \code{fixef}, and \code{coeftable}. Unlike Stata, no
#' \code{residuals()}-style option is needed before using the fixed-effect
#' prediction types: residuals and prediction caches are always retained
#' (except in grouped fits).
#'
#' @usage
#' xhdfe(fml, data = NULL,
#'       vcov = NULL, cluster = NULL,
#'       weights = NULL, weights_type = c("analytic", "frequency"),
#'       subset = NULL,
#'       group = NULL, individual = NULL, aggregation = "mean",
#'       save_fe = FALSE, groupvar = FALSE,
#'       tol = 1e-8, maxiter = 100000,
#'       tolerance_mode = "reghdfe-comparable",
#'       convergence = "auto", check_interval = 1,
#'       absorption_method = "auto", symmetric_sweep = FALSE,
#'       jacobi_relaxation = 0,
#'       fe_tolerance = 1e-6, fe_recovery_method = "hybrid",
#'       drop_singletons = TRUE, keep_singletons = NULL,
#'       dof = NULL, ssc = NULL, stats_style = "reghdfe",
#'       level = 95,
#'       threads = 0, default_threads = 0, max_threads = 0,
#'       min_parallel_rows = 20000, target_rows_per_thread = 500000,
#'       backend = c("default", "cpu", "cuda", "metal"),
#'       na.action = c("omit", "fail"))
#'
#' @param fml a model formula with up to three \code{|}-separated parts:
#'   \code{y ~ regressors | fixed effects | endogenous ~ instruments}.
#'
#'   \emph{Part 1 -- regressors.} The left-hand side must be numeric. The
#'   regressor part has the full power of R model formulas: factors expand
#'   into dummies (the complete Stata-style stripe with base levels is not
#'   posted; R's usual treatment-contrast columns are used), interactions
#'   (\code{x1:x2}, \code{x1*x2}), and transformations (\code{log(x)},
#'   \code{I(x^2)}, \code{poly(x, 2)}) are accepted. Use \code{0 +} (or
#'   \code{- 1}) to remove the constant, the analogue of Stata's
#'   \code{noconstant}.
#'
#'   \emph{Part 2 -- absorbed fixed effects.} A \code{+}-separated list of
#'   fixed-effect terms (see the \emph{Absorbing fixed effects} section for
#'   the complete grammar):
#'   \itemize{
#'     \item \code{fe} -- plain absorbed intercepts, one per level
#'       (Stata \code{absorb(fe)}).
#'     \item \code{f1^f2} -- combined fixed effect, i.e. the interaction of
#'       two (or more, via nesting) id variables; equivalent to Stata
#'       \code{absorb(f1#f2)}. It is \emph{not} expanded into main effects:
#'       write \code{f1 + f2 + f1^f2} explicitly if all three are intended
#'       (this matches \code{reghdfe}, which canonicalizes categorical-only
#'       \code{f1##f2} to the same joint fixed effect as \code{f1#f2}).
#'     \item \code{fe[x]} -- absorbed intercepts plus group-specific
#'       (heterogeneous) slopes on \code{x}; equivalent to Stata
#'       \code{absorb(fe##c.x)}.
#'     \item \code{fe[[x]]} -- group-specific slopes only, without absorbed
#'       intercepts; equivalent to Stata \code{absorb(fe#c.x)}.
#'     \item \code{fe[x1, x2]} -- multiple slope variables on the same
#'       carrier ids; internally one absorbed dimension is created per slope
#'       variable, with the intercept riding on the first, exactly like
#'       Stata's expansion \code{absorb(fe##c.x1 fe#c.x2)}.
#'     \item Arbitrary expressions are evaluated in \code{data}, e.g.
#'       \code{factor(region)} or \code{interaction(a, b)}.
#'   }
#'
#'   \emph{Part 3 -- IV/2SLS.} An optional \code{endo ~ inst} part naming the
#'   endogenous regressors and their instruments (both sides may contain
#'   several variables joined with \code{+}). The endogenous coefficients are
#'   appended \emph{after} the exogenous ones in the coefficient vector.
#'   Without absorbed fixed effects, use the two-part IV form
#'   \code{y ~ exo | endo ~ inst}.
#' @param data a \code{data.frame} (or environment) in which the formula,
#'   \code{cluster}, \code{weights}, \code{subset}, \code{group}, and
#'   \code{individual} specifications are evaluated. Variables not found in
#'   \code{data} are looked up in the formula's environment.
#' @param vcov the variance-covariance estimator. One of:
#'   \itemize{
#'     \item \code{NULL} (default) -- \code{"unadjusted"} when no clusters
#'       are supplied, \code{"cluster"} otherwise.
#'     \item \code{"unadjusted"} (aliases \code{"iid"}, \code{"ols"},
#'       \code{"homoskedastic"}, \code{"classical"}, \code{"unadj"}) --
#'       conventional homoskedastic standard errors; Stata
#'       \code{vce(unadjusted)}.
#'     \item \code{"robust"} (aliases \code{"hc1"},
#'       \code{"heteroskedastic"}) -- HC1-style heteroskedasticity-robust
#'       standard errors; Stata \code{vce(robust)}.
#'     \item \code{"cluster"} (alias \code{"clustered"}) -- (multiway)
#'       clustered standard errors; requires \code{cluster}. Stata
#'       \code{vce(cluster ...)}.
#'     \item a one-sided formula such as \code{~ id1 + id2} -- shorthand for
#'       \code{vcov = "cluster", cluster = ~ id1 + id2}.
#'   }
#'   Stata's \code{pweight} has no direct R analogue: use analytic weights
#'   together with \code{vcov = "robust"} (which is exactly what Stata's
#'   \code{xhdfe} does by default under \code{pweight}).
#' @param cluster the cluster dimensions, in any of the equivalent forms:
#'   a one-sided formula (\code{~ firm} or \code{~ firm + year}; terms are
#'   evaluated in \code{data} and \code{a^b} combines ids as in the
#'   fixed-effects part), a character vector of column names in \code{data}
#'   (\code{c("firm", "year")}), a bare vector, a \code{data.frame} or
#'   matrix (one dimension per column), or a list of vectors. Supplying
#'   more than one dimension requests multiway (Cameron-Gelbach-Miller)
#'   clustering, as with Stata \code{vce(cluster id1 id2)}. Passing
#'   \code{cluster} without \code{vcov} implies \code{vcov = "cluster"}.
#' @param weights observation weights: a numeric vector, a column name in
#'   \code{data}, or a one-sided formula (\code{~ w}). Interpretation is
#'   controlled by \code{weights_type}.
#' @param weights_type \code{"analytic"} (default) treats \code{weights}
#'   like Stata \code{aweight}s: relative precision weights, with the number
#'   of observations equal to the number of rows. \code{"frequency"} treats
#'   them like Stata \code{fweight}s: each row counts \code{weights} times,
#'   with \emph{exact replicated-rows semantics} -- observation counts,
#'   degrees of freedom, and the \code{*_effective} count fields use the sum
#'   of the weights, matching a physically duplicated dataset. For Stata
#'   \code{pweight} behavior combine analytic weights with
#'   \code{vcov = "robust"}.
#' @param subset an optional expression evaluated in \code{data} (like the
#'   \code{subset} argument of \code{\link[stats]{lm}}): either a logical
#'   vector or a vector of row indices selecting the rows used in
#'   estimation. The analogue of Stata's \code{if}/\code{in}.
#' @param group the group identifier for group-level outcomes (Stata
#'   \code{group(groupvar)}), e.g. a patent id. Accepts a one-sided formula
#'   (\code{~ patent}), a column name, or a vector. When supplied without
#'   \code{individual}, the regression runs on one representative row per
#'   group; see the \emph{Group-level outcomes with individual fixed
#'   effects} section.
#' @param individual the individual identifier whose fixed effect is
#'   absorbed at the group level (Stata \code{individual(indvar)}), e.g. an
#'   inventor id. Requires \code{group}. Same accepted forms as
#'   \code{group}. The individual dimension is automatically appended (last)
#'   to the absorbed dimensions if it is not already among them, mirroring
#'   Stata's behavior.
#' @param aggregation how the individual fixed-effect components are
#'   aggregated within each group: \code{"mean"} (default; Stata aliases
#'   \code{avg}/\code{average}) or \code{"sum"}. Case-insensitive. The
#'   analogue of Stata's \code{aggregation()}.
#' @param save_fe logical; if \code{TRUE}, recover the fixed-effect
#'   contributions (the analogue of Stata's \code{absorb(..., savefe)} /
#'   \code{savefes(prefix)}). The recovered values are stored in the
#'   \code{fe_effects} field (one vector per absorbed dimension, on the
#'   estimation sample) and are most conveniently retrieved with
#'   \code{fixef()}, which expands them to full input length with \code{NA}
#'   off-sample. Where Stata creates variables named \code{__hdfe1__},
#'   \code{__hdfe1__Slope1}, ... , the R interface names the list entries
#'   after the fixed-effect terms. Note that the saved values are
#'   \emph{contributions} (one value per observation), not group ids, and
#'   may not be separately identified in all designs. For heterogeneous
#'   slopes, \code{fe[x]} yields both the level contribution and the slope
#'   contribution while \code{fe[[x]]} yields only the slope contribution,
#'   matching \code{reghdfe}'s layout. With the default
#'   \code{fe_recovery_method = "hybrid"}, the alphas are accumulated during
#'   absorption when feasible so recovery usually avoids a second sweep;
#'   \code{fe_tolerance} applies only to the fallback MAP recovery.
#'   Restrictions: not available together with both \code{group} and
#'   \code{individual}, nor with the LSMR-family absorption methods.
#' @param groupvar logical; if \code{TRUE}, compute the first mobility group
#'   (the connected component of the first two absorbed fixed effects) and
#'   store its 1-based ids in the \code{groupvar} field of the result (the
#'   analogue of Stata's \code{groupvar(newvar)}). Requires at least two
#'   absorbed fixed effects and degrees-of-freedom adjustments that include
#'   the \code{firstpair} or \code{pairwise} logic (the default
#'   \code{dof = "all"} qualifies).
#' @param tol convergence tolerance of the absorber; default \code{1e-8}.
#'   Its exact meaning depends on \code{tolerance_mode}; under the default
#'   \code{"reghdfe-comparable"} mode it carries the same meaning as
#'   \code{reghdfe}'s \code{tolerance()}.
#' @param maxiter maximum number of absorber iterations; default
#'   \code{100000} (the analogue of Stata's \code{maxiter()} /
#'   \code{iterate()}). If the absorber does not converge within
#'   \code{maxiter} iterations a warning is issued and the (untrustworthy)
#'   estimates are still returned with \code{converged = FALSE}.
#' @param tolerance_mode the absorber convergence mode; the mode used is
#'   recorded in the \code{tolerance_mode} field (Stata
#'   \code{e(tolerance_mode)}). One of:
#'   \itemize{
#'     \item \code{"reghdfe-comparable"} (default since 2.7.0; aliases
#'       \code{"reghdfe"}, \code{"comparable"}, \code{"reghdfe_comparable"}):
#'       the accelerated absorber stops when one full sweep moves the
#'       working data by less than \code{tol} in relative norm -- the same
#'       meaning \code{reghdfe} attaches to its tolerance -- so coefficients
#'       match \code{reghdfe} at the same nominal tolerance, down to problem
#'       conditioning. Non-accelerated solver paths use a calibrated
#'       absorber tolerance of \code{min(tol, 1e-9)}.
#'     \item \code{"xhdfe-fast"} (aliases \code{"fast"}, \code{"xhdfe"},
#'       \code{"current"}, \code{"xhdfe_fast"}): the pre-2.7.0 stopping
#'       rule; typically about 1.5-3x fewer absorber iterations. Its
#'       \emph{effective} precision is data-dependent and can be looser
#'       than the nominal tolerance on ill-conditioned (e.g. sparse
#'       bipartite) designs. Appropriate for exploration and speed
#'       benchmarking -- always state the mode when citing timings.
#'     \item \code{"strict-residual"} (aliases \code{"strict"},
#'       \code{"residual-certificate"}, \code{"strict_residual"},
#'       \code{"residual_certificate"}): a heavier audit mode that treats
#'       the final absolute maximum group-mean residual check as
#'       authoritative; it may use extra iterations up to \code{maxiter}
#'       and reports non-convergence if the check is not met. Not
#'       supported with heterogeneous slopes (an error is raised before
#'       estimation).
#'   }
#'   New in 2.11.0: under the default mode, absorbers on ill-conditioned,
#'   poorly connected multiway fixed-effect graphs -- where accelerated
#'   alternating projections would stall into thousands of plain sweeps --
#'   are automatically handed off to a stable per-column conjugate-gradient
#'   solver on the symmetric demeaning operator. The handoff is transparent:
#'   coefficients, standard errors, default output, and convergence behavior
#'   are unchanged (typically \emph{tighter} agreement with \code{reghdfe}),
#'   it applies on CPU and CUDA alike, well-conditioned problems are
#'   unaffected and numerically identical to prior versions, and runtime can
#'   drop by an order of magnitude on the pathological graphs.
#' @param convergence the stopping criterion of the heterogeneous-slope
#'   absorber, active only when the fixed-effects part contains slope terms
#'   (\code{fe[x]} / \code{fe[[x]]}); standard absorption follows
#'   \code{tolerance_mode}. One of \code{"auto"} (default: follows
#'   \code{tolerance_mode} -- under \code{"reghdfe-comparable"} it stops on
#'   the reghdfe-style update criterion at the nominal tolerance, under
#'   \code{"xhdfe-fast"} it uses the historical norm-change trigger),
#'   \code{"normchange"} (aliases \code{"norm"}, \code{"norm-change"},
#'   \code{"norm_change"}; relative change in the transformed-vector norms),
#'   \code{"reghdfe"} (aliases \code{"update"}, \code{"reldif"};
#'   reghdfe-style vector update criterion, i.e. the maximum weighted mean
#'   relative difference between successive transformed y/X columns), or
#'   \code{"both"} (requires both criteria). Note that
#'   \code{tolerance_mode = "strict-residual"} is not supported with
#'   heterogeneous slopes.
#' @param check_interval number of absorber iterations between convergence
#'   checks; default \code{1} (check every iteration).
#' @param absorption_method the fixed-effect absorption method
#'   (case-insensitive; the analogue of Stata's \code{absorptionmethod()}):
#'   \itemize{
#'     \item \code{"auto"} (default): first runs the probe-based auto-MLSMR
#'       selector on eligible CPU standard fixed-effect designs with at most
#'       3 fixed-effect dimensions; it promotes slow-converging large
#'       designs to \code{"mlsmr"} and otherwise resolves to a sweep
#'       fallback (\code{"gauss-seidel"} by default,
#'       \code{"symmetric-gauss-seidel"} only when
#'       \code{symmetric_sweep = TRUE}), avoiding a second survey by the
#'       Schwarz/Jacobi-PCG gate. Under
#'       \code{tolerance_mode = "xhdfe-fast"}, MLSMR promotion is limited to
#'       large, many-right-hand-side, multiway moderate-rho designs.
#'       Designs with 4 or more fixed effects stay on the full-data
#'       gate/sweep path so that the 200,000-row sampling probe cannot
#'       over-promote (raise the cap with the environment variable
#'       \code{XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES}; an explicit
#'       \code{"auto-mlsmr"} is uncapped). Ineligible paths (GPU backends,
#'       \code{save_fe}, heterogeneous slopes, \code{group}/
#'       \code{individual}) resolve directly to the sweep fallback. Gate
#'       promotions to MLSMR can be disabled entirely with
#'       \code{XHDFE_MLSMR_AUTO_GATE=0}.
#'     \item \code{"gauss-seidel"} (aliases \code{"gs"},
#'       \code{"gauss_seidel"}): alternating-projections Gauss-Seidel
#'       sweeps.
#'     \item \code{"symmetric-gauss-seidel"} (aliases \code{"symmetric"},
#'       \code{"sym"}, \code{"symgs"}, \code{"symmetric_gauss_seidel"}):
#'       forward plus backward sweeps.
#'     \item \code{"jacobi"}: Jacobi sweeps (see \code{jacobi_relaxation}).
#'     \item \code{"schwarz"} (aliases \code{"pcg"}, \code{"schwarz-pcg"}):
#'       the additive-Schwarz / preconditioned conjugate-gradient demeaner.
#'     \item \code{"lsmr"} (aliases \code{"plain-lsmr"},
#'       \code{"plain_lsmr"}): a matrix-free LSMR Krylov absorber.
#'     \item \code{"mlsmr"} (aliases \code{"modified-lsmr"},
#'       \code{"modified_lsmr"}, \code{"within"}, \code{"within-additive"},
#'       \code{"within_additive"}): the modified-LSMR (within) absorber.
#'     \item \code{"auto-mlsmr"} (aliases \code{"auto_mlsmr"},
#'       \code{"mlsmr-auto"}, \code{"mlsmr_auto"}): runs the probe-based
#'       MLSMR selector explicitly (uncapped in the number of fixed
#'       effects).
#'   }
#'   The LSMR/MLSMR family are matrix-free additive-Schwarz Krylov
#'   absorbers; they are \emph{CPU-only} for standard fixed-effect designs
#'   and do not support \code{save_fe}, heterogeneous slopes, or
#'   \code{group}/\code{individual}. The method effectively used is
#'   reported in the \code{absorption_method_used} and
#'   \code{absorption_method_code} fields (same code table as Stata's
#'   \code{e(absorption_method_used)}).
#' @param symmetric_sweep logical; if \code{TRUE}, use a forward plus a
#'   backward sweep per iteration for the Gauss-Seidel methods (the analogue
#'   of Stata's \code{symmetricsweep}). It can also influence what
#'   \code{absorption_method = "auto"} selects when several fixed effects
#'   are present.
#' @param jacobi_relaxation relaxation parameter of the Jacobi method.
#'   Values less than or equal to 0 select the default \code{2/(J + 1)},
#'   where \code{J} is the number of fixed-effect dimensions; values above 1
#'   are capped at 1.
#' @param fe_tolerance tolerance of the fixed-effect recovery used by
#'   \code{save_fe} (the analogue of Stata's \code{fetolerance()}); default
#'   \code{1e-6}. It is separate from \code{tol} and is only used when the
#'   MAP fallback recovery path runs.
#' @param fe_recovery_method the fixed-effect recovery strategy for
#'   \code{save_fe}: \code{"hybrid"} (default; reuses alphas cached during
#'   absorption when available, usually avoiding a second recovery sweep) or
#'   \code{"map"} (forces recovery sweeps on the partial residual). The
#'   analogue of Stata's \code{ferecoverymethod()}.
#' @param drop_singletons logical; if \code{TRUE} (default), iteratively
#'   drop singleton observations (observations alone in a level of any
#'   absorbed fixed effect) before estimation, exactly as \code{reghdfe}
#'   does. The counts are reported in \code{num_singletons} /
#'   \code{nobs_full}.
#' @param keep_singletons if \code{TRUE}, keep singleton observations (the
#'   analogue of Stata's \code{keepsingletons}); when supplied it takes
#'   precedence over \code{drop_singletons}. \strong{Warning:} keeping
#'   singletons can bias standard errors when fixed effects are nested
#'   within clusters -- see the note referenced by \code{reghdfe} at
#'   \url{http://scorreia.com/reghdfe/nested_within_cluster.pdf}.
#' @param dof degrees-of-freedom adjustment rules for the absorbed fixed
#'   effects (the analogue of Stata's \code{dofadjustments()}). Default
#'   (\code{NULL}) is the full \code{reghdfe} logic (\code{"all"}). Supply a
#'   character vector of case-insensitive tokens; \strong{when supplied,
#'   only the listed tokens are enabled} (reghdfe semantics):
#'   \describe{
#'     \item{\code{"all"}}{full \code{reghdfe} degrees-of-freedom logic
#'       (implies the cluster and continuous adjustments).}
#'     \item{\code{"none"}}{no fixed-effect degrees-of-freedom correction.}
#'     \item{\code{"firstpair"} / \code{"first"}}{account for redundant
#'       (mobility-group) fixed effects between the first pair of absorbed
#'       dimensions only.}
#'     \item{\code{"pairwise"} / \code{"pair"}}{account for redundancy
#'       across all pairs of absorbed dimensions.}
#'     \item{\code{"clusters"} / \code{"cluster"}}{adjust for absorbed fixed
#'       effects nested within cluster variables.}
#'     \item{\code{"continuous"} / \code{"cont"}}{reserved for
#'       \code{reghdfe}'s continuous-interaction checks (currently a
#'       no-op).}
#'   }
#'   Example: \code{dof = c("firstpair", "clusters")}.
#' @param ssc \pkg{fixest}-style small-sample corrections. Either a named
#'   list with entries among \code{ssc_k_adj} (logical), \code{ssc_k_fixef}
#'   (\code{"full"}, \code{"none"}, \code{"nonnested"}), \code{ssc_k_exact}
#'   (logical), \code{ssc_g_adj} (logical), \code{ssc_g_df} (\code{"min"} or
#'   \code{"conventional"}/\code{"conv"}), and \code{ssc_t_df} (numeric), or
#'   a single Stata-style string such as
#'   \code{"K.adj=1 K.fixef=nonnested G.df=min"} (case-insensitive tokens,
#'   space or comma separated; flag values \code{1}/\code{true}/\code{yes}
#'   or an omitted value mean \code{TRUE}, \code{0}/\code{false}/\code{no}
#'   mean \code{FALSE}; for \code{K.fixef}, \code{true}/\code{yes} map to
#'   \code{"full"} and \code{false}/\code{no} to \code{"none"}). Defaults
#'   follow \code{stats_style}: \code{K.fixef=full} under
#'   \code{stats_style = "reghdfe"} and \code{K.fixef=nonnested} under
#'   \code{stats_style = "legacy"}. Stata's \code{t.df=min} (fix the
#'   residual degrees of freedom at the minimum cluster count minus one) and
#'   \code{t.df=conventional} are expressed in R through \code{ssc_g_df};
#'   in the R interface \code{ssc_t_df} must be numeric and fixes the
#'   residual degrees of freedom at that value.
#' @param stats_style the fit-statistics and small-sample-scaling
#'   convention: \code{"reghdfe"} (default; matches \code{reghdfe}'s RMSE,
#'   adjusted R-squared, and cluster scaling) or \code{"legacy"} (the
#'   pre-change xhdfe statistics). The analogue of Stata's
#'   \code{statstyle()}.
#' @param level the confidence level for the reported intervals, either in
#'   percent (\code{95}, the default) or as a fraction (\code{0.95}). Must
#'   lie in (0, 100).
#' @param threads number of threads used by the absorber; \code{0} (default)
#'   enables automatic threading. The analogue of Stata's
#'   \code{numthreads()}. Parallel absorption requires an OpenMP build of
#'   the package (the default on Linux).
#' @param default_threads default thread count applied when automatic
#'   threading is enabled; \code{0} lets the backend decide (Stata
#'   \code{defaultthreads()}).
#' @param max_threads maximum thread count when automatic threading is
#'   enabled; \code{0} means no explicit cap (Stata \code{maxthreads()}).
#' @param min_parallel_rows minimum number of rows before parallel
#'   absorption is enabled; default \code{20000} (Stata
#'   \code{minparallelrows()}).
#' @param target_rows_per_thread heuristic target of rows per thread used to
#'   size the thread pool; default \code{500000} (Stata
#'   \code{targetrowsperthread()}).
#' @param backend the compute backend: \code{"default"} (follow the
#'   \code{XHDFE_GPU_BACKEND} environment variable if set, otherwise CPU),
#'   \code{"cpu"} (force the CPU path), \code{"cuda"} (request the CUDA GPU
#'   absorber), or \code{"metal"} (reserved for Metal builds; errors where
#'   unavailable). Requesting \code{"cuda"} or \code{"metal"} is
#'   \emph{fail-closed}: if the GPU is not effectively used the fit stops
#'   with an error instead of silently returning CPU results, mirroring the
#'   Stata plugin's error 498 contract. See the \emph{GPU backends} section
#'   for the build recipe and the diagnostics to check.
#' @param na.action \code{"omit"} (default): rows with missing values in any
#'   estimation variable (outcome, regressors, instruments, fixed-effect
#'   ids, slope variables, clusters, weights, group/individual ids) are
#'   dropped before estimation, and full-length outputs (residuals,
#'   prediction caches) are \code{NA} on those rows. \code{"fail"}: raise an
#'   error if any such missing value is present.
#'
#' @section Absorbing fixed effects:
#' The second formula part lists the fixed effects to absorb; each term is
#' equivalent to including one dummy per category, without ever building the
#' dummies. The grammar (fixest-compatible) and its Stata
#' \code{absorb()} equivalents:
#'
#' \tabular{lll}{
#'   \strong{R term} \tab \strong{Stata absorb()} \tab \strong{Meaning} \cr
#'   \code{fe} \tab \code{fe} \tab absorbed intercepts (one per level) \cr
#'   \code{f1^f2} \tab \code{f1#f2} \tab combined (interaction) fixed
#'     effect \cr
#'   \code{fe[x]} \tab \code{fe##c.x} \tab absorbed intercepts plus
#'     group-specific slopes on \code{x} \cr
#'   \code{fe[[x]]} \tab \code{fe#c.x} \tab group-specific slopes only \cr
#'   \code{fe[x1, x2]} \tab \code{fe##c.x1 fe#c.x2} \tab several slopes on
#'     one carrier \cr
#' }
#'
#' Notes:
#' \itemize{
#'   \item \emph{Categorical interactions.} \code{f1^f2} builds the joint
#'     fixed effect of the two id variables. Like \code{reghdfe} -- which
#'     canonicalizes a categorical-only \code{f1##f2} to the same joint
#'     fixed effect as \code{f1#f2} -- it is \emph{not} expanded into main
#'     effects plus interaction; write \code{f1 + f2 + f1^f2} explicitly if
#'     all three dimensions are intended.
#'   \item \emph{Heterogeneous slopes.} Slope variables must be numeric and
#'     of full data length. The core absorbs at most one slope per carrier
#'     dimension; multiple slopes on the same ids (\code{fe[x1, x2]}) are
#'     handled internally by duplicating the carrier dimension -- one
#'     absorbed dimension per slope variable, with the intercept (if
#'     requested) riding on the first -- exactly like Stata expands
#'     \code{absorb(fe##c.x1 fe#c.x2)}.
#'   \item \emph{Id encoding.} Integer id vectors are passed to the core
#'     verbatim (level handling is the core's job); factors use their level
#'     codes; character, logical, and non-integral numeric vectors are
#'     encoded by first appearance. The same rules apply to cluster,
#'     group, and individual ids.
#'   \item \emph{Restrictions.} Heterogeneous slopes are not supported with
#'     \code{group}/\code{individual} or with the LSMR-family absorption
#'     methods (\code{"lsmr"}, \code{"mlsmr"}, \code{"auto-mlsmr"}).
#' }
#'
#' @section Tolerance modes:
#' The nominal tolerance \code{tol} only has a precise meaning together with
#' \code{tolerance_mode}. Under the default \code{"reghdfe-comparable"} mode
#' (the default since 2.7.0) the absorber stops when one full sweep moves
#' the working data by less than \code{tol} in relative norm -- the same
#' meaning \code{reghdfe} attaches to its tolerance -- so coefficients match
#' \code{reghdfe} at the same nominal tolerance, down to problem
#' conditioning. \code{"xhdfe-fast"} restores the pre-2.7.0 stopping rule:
#' typically about 1.5-3x fewer absorber iterations, but its effective
#' precision is data-dependent and can be looser than the nominal tolerance
#' on ill-conditioned (e.g. sparse bipartite) designs; it is appropriate for
#' exploration and speed benchmarking, and any published timing should state
#' the mode used. \code{"strict-residual"} is a heavier audit mode that
#' treats the final absolute maximum group-mean residual check as
#' authoritative, may use extra iterations up to \code{maxiter}, and reports
#' non-convergence when the check is not met; it is not supported with
#' heterogeneous slopes.
#'
#' New in 2.11.0: under the default mode, absorption on ill-conditioned,
#' poorly connected multiway fixed-effect graphs is transparently handed off
#' to a stable per-column conjugate-gradient solver on the symmetric
#' demeaning operator, on CPU and CUDA alike. Results on well-conditioned
#' problems are numerically identical to prior versions; on pathological
#' graphs the handoff can cut runtime by an order of magnitude at equal or
#' better precision.
#'
#' The mode used is recorded in the \code{tolerance_mode} field of the
#' result (Stata \code{e(tolerance_mode)}).
#'
#' @section GPU backends:
#' \code{backend = "cuda"} requests the CUDA absorber. It requires a package
#' build with CUDA support, e.g.
#' \preformatted{XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 R CMD INSTALL xhdfe}
#' where \code{XHDFE_CUDA_ARCH} selects a single target compute capability
#' (default 90, e.g. an NVIDIA H100; the backend requires compute capability
#' 7.5 or newer and the NVCC compiler). The equivalent Stata plugin build is
#' \code{XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash
#' stata/tools/build-plugin.sh --linux --openmp}.
#'
#' \strong{Fail-closed contract.} Requesting \code{"cuda"} (or
#' \code{"metal"}) never silently falls back to the CPU: if the backend is
#' unavailable, fails, does not converge on the GPU, or a CPU cache/profile
#' result would be returned, the fit stops with an error -- mirroring the
#' Stata command, which exits with error 498 in the same situations.
#' After a successful GPU fit, validate real GPU use through the
#' diagnostics on the result object: \code{gpu_used} must be \code{1} and
#' \code{gpu_status} must be \code{"used"} (Stata: \code{e(gpu_used)==1},
#' \code{e(gpu_backend)=="cuda"}, \code{e(gpu_status)=="used"}). The status
#' taxonomy (\code{gpu_status_code} / \code{gpu_status}) is:
#'
#' \tabular{lll}{
#'   \strong{Code} \tab \strong{Label} \tab \strong{Meaning} \cr
#'   0 \tab \code{not_requested} \tab GPU not requested \cr
#'   1 \tab \code{used} \tab GPU backend effectively used \cr
#'   2 \tab \code{backend_unavailable} \tab requested backend unavailable \cr
#'   3 \tab \code{gpu_absorption_not_converged} \tab GPU absorption did not
#'     converge \cr
#'   4 \tab \code{gpu_backend_failed} \tab GPU backend failed \cr
#'   5 \tab \code{cpu_cache_or_profile_result} \tab a CPU cache/profile
#'     result was produced \cr
#' }
#'
#' As an alternative to the \code{backend} argument, the environment
#' variable \code{XHDFE_GPU_BACKEND} (\code{cpu}|\code{cuda}|\code{metal})
#' selects the backend; the \code{backend} argument temporarily sets that
#' variable around the fit and is the recommended interface.
#' \code{backend = "cpu"} forces the CPU path even when the environment
#' variable is set. \code{"metal"} is reserved for Metal builds and errors
#' where unavailable. CPU behavior is the reference implementation; the
#' LSMR-family absorption methods are CPU-only, and GPU-ineligible paths
#' under \code{absorption_method = "auto"} resolve to the sweep fallback.
#'
#' @section Group-level outcomes with individual fixed effects:
#' \code{xhdfe()} supports individual fixed effects with group-level
#' outcomes (e.g. patent-level citations with inventor fixed effects). The
#' data must be in long format where the group and individual ids together
#' uniquely identify observations (the Stata help's
#' \code{isid patent_id inventor_id} requirement). The combining rules are:
#' \enumerate{
#'   \item no fixed effects, no \code{group}, no \code{individual}:
#'     ordinary least squares with a constant;
#'   \item only absorbed fixed effects: standard fixed-effects regression;
#'   \item \code{group} without \code{individual}: the same as (1)/(2) run
#'     on one observation per group (the data are collapsed to a
#'     representative row per group);
#'   \item \code{group} and \code{individual} (plus, optionally, absorbed
#'     fixed effects): a group-level fixed-effects regression in which each
#'     group's outcome loads on the \code{aggregation} (mean or sum) of its
#'     individuals' fixed-effect components.
#' }
#' The individual dimension is appended (last) to the absorbed dimensions
#' for stability if not already present, exactly as in Stata. In
#' group/individual mode the core estimates on the collapsed group-level
#' rows: the \code{residuals} field is then at the group level,
#' \code{sample} is empty, \code{grouped_fit} is \code{TRUE}, and the
#' prediction caches are unavailable. \strong{Note:} singleton bookkeeping
#' happens \emph{after} the representative-row construction, so
#' \code{num_singletons} can differ from \code{reghdfe}'s pre-collapse row
#' count even when the effective estimation sample is identical.
#'
#' IV/2SLS, heterogeneous slopes, and (with both \code{group} and
#' \code{individual}) \code{save_fe} are not supported in group mode.
#'
#' @section Missing features and restrictions:
#' \itemize{
#'   \item Heterogeneous slopes are not supported together with
#'     \code{group}/\code{individual}, nor with the LSMR-family absorption
#'     methods (\code{"lsmr"}, \code{"mlsmr"}, \code{"auto-mlsmr"}).
#'   \item \code{save_fe} is not supported when both \code{group} and
#'     \code{individual} are specified.
#'   \item IV/2SLS is not supported in \code{group} mode.
#'   \item \code{tolerance_mode = "strict-residual"} is not supported with
#'     heterogeneous slopes (an error is raised before estimation).
#'   \item The LSMR-family absorption methods are CPU-only and do not
#'     support \code{save_fe}.
#'   \item No HAC or Driscoll-Kraay standard errors (see \code{ivreghdfe}
#'     in the \code{reghdfe} ecosystem).
#'   \item With IV/2SLS and absorbed fixed effects, the reported
#'     \code{"(Intercept)"} is \emph{not identified}: this is documented
#'     upstream behavior (Stata prints the same degenerate \code{_cons},
#'     with a t-statistic near 0 and a p-value near 1), and users should
#'     ignore that row.
#' }
#'
#' @section Environment variables:
#' The C++ core reads the following runtime knobs:
#' \describe{
#'   \item{\code{XHDFE_GPU_BACKEND}}{backend request
#'     (\code{cpu}|\code{cuda}|\code{metal}); the \code{backend} argument
#'     sets it temporarily around the fit and restores the previous value.}
#'   \item{\code{XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES}}{raises the default cap
#'     (3) on the number of fixed-effect dimensions eligible for the
#'     probe-based auto-MLSMR selector under
#'     \code{absorption_method = "auto"}.}
#'   \item{\code{XHDFE_MLSMR_AUTO_GATE}}{set to \code{0} to disable MLSMR
#'     promotions by the automatic gate entirely.}
#'   \item{\code{XHDFE_MOBILITY_PROFILE}, \code{XHDFE_MOBILITY_MODE}}{path
#'     and mode (\code{off}|\code{read}|\code{write}|\code{auto}) of the
#'     mobility-diagnostics profile -- the R route to the Stata
#'     \code{mobilityprofile}/\code{mobfile()} options.}
#'   \item{\code{XHDFE_ABSORPTION_CACHE},
#'     \code{XHDFE_ABSORPTION_CACHE_MODE}}{path and mode of the absorption
#'     cache that stores transformed outcome/regressors for reuse across
#'     identical specifications (Stata \code{absorptioncache()} /
#'     \code{absorptioncachemode()}); can be large.}
#'   \item{\code{XHDFE_FE_STRUCTURE_CACHE},
#'     \code{XHDFE_FE_STRUCTURE_MODE}}{path and mode of the fixed-effect
#'     structure cache (normalized group ids and singleton filters) reused
#'     across runs with the same absorb structure (Stata
#'     \code{festructurecache}/\code{fescache()}/\code{fescachemode()}).}
#'   \item{\code{XHDFE_FE_DIAG_FILE}}{appends per-run fixed-effect
#'     diagnostics (partial-residual statistics, fixed-effect totals,
#'     residual consistency, recovery method) to the given file.}
#'   \item{\code{XHDFE_PROFILE_SAVEFE}}{nonzero enables fixed-effect
#'     recovery timing traces on stderr (partial-residual construction,
#'     cached-hybrid reconstruction and checks, MAP-fallback stages);
#'     disabled by default with no output or behavior change when unset.}
#'   \item{\code{XHDFE_PROFILE_CPU}}{nonzero enables CPU-stage timing traces
#'     (absorption and OLS stage profiling) on stderr.}
#'   \item{\code{XHDFE_FE_NORMALIZE}}{normalization style of recovered
#'     fixed-effect contributions: \code{component} (default; per-component
#'     normalization) or \code{reghdfe} (aliases \code{mean},
#'     \code{mean_only}; reghdfe-style mean-only normalization).}
#' }
#'
#' @section Warnings and notes:
#' \itemize{
#'   \item \emph{Saturated designs.} If the absorbed model fits the data
#'     perfectly (zero residual degrees of freedom), a note is emitted,
#'     \code{saturated} is set to 1, adjusted R-squared statistics are
#'     forced to 1, and standard errors are not identified. The reported
#'     coefficients are still the exact within-OLS solution.
#'   \item \emph{Multiway-cluster PSD adjustment.} If the multiway-cluster
#'     variance matrix requires the Cameron-Gelbach-Miller
#'     positive-semi-definite adjustment, a warning is printed (matching
#'     \code{reghdfe}) and \code{vcv_psd_fixed} is 1.
#'   \item \emph{Singletons.} \code{keep_singletons = TRUE} can bias
#'     inference when fixed effects are nested within clusters; see
#'     \url{http://scorreia.com/reghdfe/nested_within_cluster.pdf}.
#'   \item \emph{Group-mode singleton bookkeeping.} In
#'     \code{group}/\code{individual} mode the singleton counts are computed
#'     after the representative-row construction and can differ from
#'     \code{reghdfe}'s pre-collapse counts.
#'   \item \emph{Combined fixed effects.} \code{f1^f2} (like
#'     \code{reghdfe}'s canonicalized categorical \code{##}) creates only
#'     the joint fixed effect; list \code{f1 + f2 + f1^f2} for main effects
#'     plus interaction.
#'   \item \emph{Recovered fixed effects are contributions}, not group ids,
#'     and may not be separately identified in all designs.
#'   \item \emph{Non-convergence.} If the absorber hits \code{maxiter}, the
#'     estimates are returned with a warning and \code{converged = FALSE};
#'     they should not be trusted.
#' }
#'
#' @return An object of class \code{"xhdfe"}: a list whose fields are the R
#' analogue of Stata's stored results (the corresponding \code{e()} names
#' are given where applicable).
#'
#' \strong{Coefficients and inference}
#' \describe{
#'   \item{\code{coefficients}}{named numeric vector of point estimates
#'     (\code{e(b)}). With absorbed fixed effects a reghdfe-compatible
#'     \code{"(Intercept)"} is appended after the regressors; IV
#'     coefficients follow the exogenous ones.}
#'   \item{\code{se}, \code{tvalues}, \code{pvalues}}{standard errors,
#'     t-statistics, and p-values aligned with \code{coefficients}
#'     (from \code{e(V)} and \code{e(df_r)}).}
#'   \item{\code{conf_int}}{two-column matrix of confidence bounds at
#'     \code{level} (columns labelled e.g. \code{"2.5 \%"} and
#'     \code{"97.5 \%"}).}
#'   \item{\code{vcov}}{variance-covariance matrix of the estimates
#'     (\code{e(V)}).}
#'   \item{\code{omitted}}{named integer codes per coefficient
#'     (\code{e(omit_reason)}): 0 kept, 1 collinear with the absorbed fixed
#'     effects, 2 other collinearity.}
#'   \item{\code{se_type}}{\code{"unadjusted"}, \code{"robust"}, or
#'     \code{"cluster"} (cf. \code{e(vce)}).}
#'   \item{\code{level}}{confidence level in percent.}
#' }
#'
#' \strong{Sample and observation counts}
#' \describe{
#'   \item{\code{residuals}}{full-length residual vector (\code{n_input}
#'     entries) with \code{NA} on rows outside the estimation sample; in
#'     grouped fits the residuals are at the group level instead.}
#'   \item{\code{sample}}{integer indices of the input rows used in
#'     estimation, after \code{subset}, missing-value filtering, and
#'     singleton dropping -- the analogue of \code{e(sample)}. Empty in
#'     grouped fits.}
#'   \item{\code{nobs}}{number of observations used (\code{e(N)}).}
#'   \item{\code{nobs_full}}{observations before singleton dropping
#'     (\code{e(N_full)}).}
#'   \item{\code{num_singletons}}{singleton observations dropped
#'     (\code{e(num_singletons)}).}
#'   \item{\code{nobs_effective}, \code{nobs_full_effective},
#'     \code{num_singletons_effective}}{frequency-weight effective counts:
#'     with \code{weights_type = "frequency"} these are sums of weights
#'     (replicated-rows semantics); otherwise they equal the row counts.}
#'   \item{\code{sumweights}}{sum of weights, or \code{nobs} if unweighted
#'     (\code{e(sumweights)}).}
#'   \item{\code{n_input}}{number of rows of the input data before any
#'     filtering.}
#'   \item{\code{grouped_fit}}{\code{TRUE} for \code{group}/
#'     \code{individual} collapsed fits.}
#' }
#'
#' \strong{Degrees of freedom}
#' \describe{
#'   \item{\code{df_r}}{residual degrees of freedom after small-sample
#'     corrections, used for t-statistics, p-values, and confidence
#'     intervals (\code{e(df_r)}).}
#'   \item{\code{df_r_unadj}}{unadjusted residual degrees of freedom,
#'     N - k - df_a (\code{e(df_r_unadj)}).}
#'   \item{\code{df_m}}{model degrees of freedom (\code{e(df_m)}).}
#'   \item{\code{rank}}{effective model rank, omitted coefficients excluded
#'     (\code{e(rank)}).}
#'   \item{\code{df_a}}{absorbed degrees of freedom, using the exact
#'     fixed-effect rank (\code{e(df_a)}).}
#'   \item{\code{df_a_levels}}{absorbed levels (\code{e(df_a_levels)}).}
#'   \item{\code{df_a_exact}}{absorbed levels adjusted for collinearity
#'     (\code{e(df_a_exact)}).}
#'   \item{\code{df_a_nested}}{absorbed levels nested within clusters, if
#'     any (\code{e(df_a_nested)}).}
#'   \item{\code{df_a_initial}}{absorbed levels before redundancy
#'     adjustments (\code{e(df_a_initial)}).}
#'   \item{\code{df_a_redundant}}{redundant absorbed degrees of freedom
#'     (\code{e(df_a_redundant)}).}
#' }
#'
#' \strong{Fit statistics}
#' \describe{
#'   \item{\code{r2}, \code{r2_within}}{overall and within R-squared
#'     (\code{e(r2)}, \code{e(r2_within)}).}
#'   \item{\code{r2_a}, \code{r2_a_within}}{adjusted (within) R-squared,
#'     computed with the unadjusted residual degrees of freedom, as in
#'     \code{reghdfe} (\code{e(r2_a)}, \code{e(r2_a_within)}); forced to 1
#'     in saturated fits.}
#'   \item{\code{rss}, \code{tss}, \code{tss_within}, \code{mss}}{residual,
#'     total, within-total, and model (tss - rss) sums of squares
#'     (\code{e(rss)}, \code{e(tss)}, \code{e(tss_within)}, \code{e(mss)}).
#'     When the fixed-effect part contains only slope terms
#'     (\code{fe[[x]]}, no level dimension), the model has no intercept and
#'     \code{tss}/\code{r2} use the \emph{uncentered} total sum of squares,
#'     matching \code{reghdfe} and \code{regress, noconstant}.}
#'   \item{\code{sigma2}}{estimated residual variance (\code{e(sigma2)}).}
#'   \item{\code{rmse}}{root mean squared error,
#'     sqrt(rss / df_r_unadj-style denominator), matching \code{reghdfe}
#'     (\code{e(rmse)}).}
#'   \item{\code{F_stat}, \code{F_p}}{model F-statistic and p-value
#'     (\code{e(F)}, \code{e(p)}): a Wald test that all kept non-intercept
#'     coefficients are zero, using a generalized inverse of their
#'     covariance block.}
#'   \item{\code{ll}, \code{ll_0}}{log-likelihood and log-likelihood at
#'     zeroed regressors, within (\code{e(ll)}, \code{e(ll_0)}).}
#'   \item{\code{saturated}}{1 if the fitted design is saturated /
#'     perfect-fit, 0 otherwise (\code{e(saturated)}).}
#'   \item{\code{has_intercept}}{whether a constant is reported
#'     (\code{e(report_constant)}).}
#' }
#'
#' \strong{Absorbed-fixed-effect diagnostics} (one entry per absorbed
#' dimension, mirroring \code{reghdfe}'s degrees-of-freedom footnote table,
#' \code{e(dof_table)})
#' \describe{
#'   \item{\code{fe_num_levels}}{number of categories after singleton
#'     dropping.}
#'   \item{\code{fe_base_levels}}{raw number of levels before adjustments.}
#'   \item{\code{fe_redundant}}{redundant coefficients detected in the
#'     dimension.}
#'   \item{\code{fe_num_coefs}}{number of coefficients contributed by the
#'     dimension.}
#'   \item{\code{fe_inexact}}{flag: the redundancy count is an inexact
#'     (pairwise) estimate.}
#'   \item{\code{fe_nested}}{flag: the dimension is nested within a cluster
#'     variable.}
#'   \item{\code{fe_labels}}{labels of the absorbed dimensions (deparsed
#'     fixed-effect terms; cf. \code{e(absorb_labels)}).}
#' }
#'
#' \strong{Solver and backend diagnostics}
#' \describe{
#'   \item{\code{iterations}}{absorber iterations (\code{e(iterations)}).}
#'   \item{\code{converged}}{1 if converged, 0 otherwise
#'     (\code{e(converged)}).}
#'   \item{\code{absorption_method_used}}{name of the method effectively
#'     used (\code{"auto"}, \code{"gauss-seidel"},
#'     \code{"symmetric-gauss-seidel"}, \code{"jacobi"}, \code{"schwarz"},
#'     \code{"lsmr"}, \code{"mlsmr"}, \code{"auto-mlsmr"}).}
#'   \item{\code{absorption_method_code}}{numeric code with the same table
#'     as Stata's \code{e(absorption_method_used)}: 0 auto, 1 gauss-seidel,
#'     2 symmetric-gauss-seidel, 3 jacobi, 4 schwarz, 5 lsmr, 6 mlsmr,
#'     7 auto-mlsmr.}
#'   \item{\code{threads_used}}{threads used by the backend
#'     (\code{e(threads_used)}).}
#'   \item{\code{tolerance_mode}}{tolerance mode used
#'     (\code{e(tolerance_mode)}).}
#'   \item{\code{gpu_used}}{1 if the GPU backend was effectively used, 0 if
#'     the CPU path was used (\code{e(gpu_used)}).}
#'   \item{\code{gpu_status_code}, \code{gpu_status}}{status code 0-5 and
#'     its label (\code{e(gpu_status_code)}, \code{e(gpu_status)}); see the
#'     \emph{GPU backends} section for the taxonomy.}
#'   \item{\code{gpu_attempted}}{1 if GPU absorption was attempted
#'     (\code{e(gpu_attempted)}).}
#'   \item{\code{gpu_absorption_converged},
#'     \code{gpu_absorption_iterations}}{convergence flag and iteration
#'     count of the attempted GPU absorption; missing if not attempted
#'     (\code{e(gpu_absorption_converged)},
#'     \code{e(gpu_absorption_iterations)}).}
#'   \item{\code{backend_requested}}{the \code{backend} argument value
#'     (cf. \code{e(gpu_backend_requested)}).}
#' }
#'
#' \strong{Fixed-effect recovery} (populated with \code{save_fe = TRUE})
#' \describe{
#'   \item{\code{fe_effects}}{named list with one numeric vector of
#'     per-observation fixed-effect contributions per absorbed dimension,
#'     on the estimation sample; use \code{fixef()} for full-length,
#'     NA-padded vectors.}
#'   \item{\code{fe_recovery_converged}, \code{fe_recovery_iterations},
#'     \code{fe_recovery_max_delta}}{recovery convergence flag, iteration
#'     count, and maximum change in the final recovery sweep
#'     (\code{e(fe_recovery_converged)}, \code{e(fe_recovery_iterations)},
#'     \code{e(fe_recovery_max_delta)}).}
#'   \item{\code{groupvar}}{with \code{groupvar = TRUE}: 1-based ids of the
#'     first mobility group (connected component), the analogue of Stata's
#'     \code{groupvar(newvar)}; otherwise \code{NULL}.}
#' }
#'
#' \strong{Cluster diagnostics} (populated when clustered)
#' \describe{
#'   \item{\code{num_clusters}}{minimum number of clusters across
#'     dimensions (\code{e(N_clust)}).}
#'   \item{\code{cluster_counts}}{cluster counts by dimension
#'     (\code{e(N_clust#)}).}
#'   \item{\code{cluster_combo_counts}}{counts for each cluster combination
#'     in bitmask order (\code{e(cluster_diag)}).}
#'   \item{\code{cluster_scale}}{cluster scaling factor from the
#'     small-sample corrections (\code{e(cluster_scale)}).}
#'   \item{\code{cluster_names}}{names of the cluster dimensions
#'     (\code{e(clustvar)}).}
#'   \item{\code{vcv_psd_fixed}}{1 if the multiway-cluster variance matrix
#'     required the Cameron-Gelbach-Miller positive-semi-definite
#'     adjustment (\code{e(vcv_psd_fixed)}); a warning is printed, matching
#'     \code{reghdfe}.}
#' }
#'
#' \strong{Prediction caches and bookkeeping}
#' \describe{
#'   \item{\code{xb_cache}, \code{y_cache}, \code{stdp_cache}}{full-length
#'     caches (NA off-sample) of the linear prediction, the outcome, and
#'     the standard error of the linear prediction; they back
#'     \code{predict()} and \code{fitted()} and are absent in grouped
#'     fits.}
#'   \item{\code{call}}{the matched call.}
#'   \item{\code{version}}{the package version string (\code{e(version)}).}
#' }
#'
#' \strong{Formula-interface extras} (set by \code{xhdfe()}, not by
#' \code{xhdfe_fit()})
#' \describe{
#'   \item{\code{fml}}{the model formula.}
#'   \item{\code{terms}}{the \code{terms} object of the regressor part
#'     (used by \code{predict(..., newdata = )}).}
#'   \item{\code{terms_endo}}{the \code{terms} object of the endogenous
#'     part for IV fits (\code{NULL} otherwise); \code{predict} uses it to
#'     rebuild the endogenous columns on new data.}
#'   \item{\code{xlevels}, \code{contrasts}}{factor levels and contrasts
#'     recorded at fit time so \code{predict(..., newdata = )} expands
#'     factors exactly as during estimation.}
#'   \item{\code{data_name}}{deparsed name of the \code{data} argument.}
#'   \item{\code{weights_type}}{\code{"analytic"} or \code{"frequency"}
#'     when weights were supplied, otherwise \code{NULL}.}
#'   \item{\code{group_name}, \code{individual_name},
#'     \code{aggregation}}{deparsed group/individual arguments and the
#'     aggregation method when group mode is used (\code{e(group)},
#'     \code{e(individual)}, \code{e(aggregation)}).}
#' }
#'
#' @references
#' Portela, Miguel, and Tiago Tavares. 2026. "xhdfe: High-dimensional fixed
#' effects regression via a C++ backend." Version 2.18.2.
#' \url{https://github.com/reisportela/xhdfe-xfe}
#'
#' Correia, Sergio. 2016. "reghdfe: Estimating linear models with multi-way
#' fixed effects." 2016 Stata Conference. RePEc:boc:scon16:24.
#'
#' Gaure, Simen. 2013. "OLS with multiple high dimensional category
#' variables." Computational Statistics & Data Analysis 66: 8-18.
#' \doi{10.1016/j.csda.2013.03.024}
#'
#' Guimaraes, Paulo, and Pedro Portugal. 2010. "A simple feasible procedure
#' to fit models with high-dimensional fixed effects." Stata Journal 10(4):
#' 628-649. \doi{10.1177/1536867X1101000406}
#'
#' Berge, Laurent, Kyle Butts, and Grant McDermott. 2026. "Fast and
#' user-friendly econometrics estimations: The R package fixest."
#' arXiv:2601.21749.
#'
#' See \code{\link{xhdfe-package}} for the complete reference list.
#'
#' @seealso \code{\link{xhdfe_fit}} for the matrix interface;
#'   \code{xhdfe_demean()} for stand-alone demeaning;
#'   \code{xhdfe_group_fes()} for group-level fixed-effect aggregation;
#'   \code{xhdfe_info()} for version/build information; \code{fixef()} for
#'   recovered fixed effects; \code{predict()} (types \code{xb},
#'   \code{xbd}, \code{d}, \code{residuals}, \code{dresiduals},
#'   \code{score}, \code{stdp}) for Stata-compatible predictions. In the
#'   wider ecosystem: \code{fixest::feols}, \code{lfe::felm}, Stata's
#'   \code{reghdfe}, Python's \code{pyfixest}, and Julia's
#'   \code{FixedEffectModels.jl}.
#'
#' @author Miguel Portela (\email{miguel.portela@@eeg.uminho.pt}) and
#'   Tiago Tavares (\email{tgstavares@@eeg.uminho.pt}).
#'
#' @examples
#' ## A small synthetic worker-firm panel ------------------------------------
#' set.seed(2026)
#' n <- 600
#' d <- data.frame(
#'   worker = sample(80, n, replace = TRUE),
#'   firm   = sample(30, n, replace = TRUE),
#'   year   = sample(2015:2020, n, replace = TRUE),
#'   x1     = rnorm(n),
#'   x2     = rnorm(n),
#'   w      = runif(n, 0.5, 1.5)
#' )
#' d$z  <- rnorm(n)                            # instrument
#' d$xe <- 0.6 * d$z + 0.3 * d$x1 + rnorm(n)   # endogenous regressor
#' d$y  <- 1 + 0.5 * d$x1 - 0.2 * d$x2 + 0.7 * d$xe +
#'   0.05 * d$worker + 0.03 * d$firm + 0.1 * (d$year - 2015) + rnorm(n)
#'
#' ## 1. One absorbed fixed effect (Stata: xhdfe y x1 x2, absorb(firm))
#' m1 <- xhdfe(y ~ x1 + x2 | firm, data = d)
#' summary(m1)
#'
#' ## 2. Clustered standard errors (Stata: ..., vce(cluster firm))
#' m2 <- xhdfe(y ~ x1 + x2 | firm + year, data = d, cluster = ~ firm)
#' ## equivalent: vcov = ~ firm, cluster = "firm", or cluster = d$firm
#'
#' ## 3. Three-way fixed effects
#' m3 <- xhdfe(y ~ x1 + x2 | worker + firm + year, data = d)
#'
#' ## 4. Fixed-effect recovery (Stata: absorb(..., savefe))
#' m4 <- xhdfe(y ~ x1 + x2 | firm + year, data = d, save_fe = TRUE)
#' alpha <- fixef(m4)   # list: one full-length vector per dimension
#' head(alpha$firm)
#'
#' ## 5. Residuals and predictions (Stata: predict, xb / xbd / d / stdp)
#' head(residuals(m1))
#' head(predict(m1, type = "xb"))
#' head(predict(m1, type = "xbd"))   # xb + absorbed effects
#' head(predict(m1, type = "d"))     # absorbed effects only
#' head(predict(m1, type = "stdp"))  # SE of the linear prediction
#'
#' ## 6. Heterogeneous slopes (Stata: absorb(firm##c.x2 year))
#' m5 <- xhdfe(y ~ x1 | firm[x2] + year, data = d)
#' ## slopes only, no firm intercepts (Stata: absorb(firm#c.x2 year)):
#' m6 <- xhdfe(y ~ x1 | firm[[x2]] + year, data = d)
#'
#' ## 7. Analytic weights (Stata: [aw = w])
#' m7 <- xhdfe(y ~ x1 + x2 | firm, data = d, weights = ~ w)
#'
#' ## 8. IV/2SLS (Stata: ..., endogenous(xe) instruments(z))
#' m8 <- xhdfe(y ~ x1 | firm + year | xe ~ z, data = d)
#'
#' ## 9. Combined fixed effect (Stata: absorb(firm#year))
#' m9 <- xhdfe(y ~ x1 + x2 | firm^year, data = d)
#'
#' ## 10. Group-level outcomes with individual fixed effects
#' ## (Stata: xhdfe citations funding, absorb(inventor_id)
#' ##         group(patent_id) individual(inventor_id))
#' pat <- data.frame(patent = rep(1:20, each = 2))
#' pat$inventor <- ((pat$patent +
#'   ave(pat$patent, pat$patent, FUN = seq_along)) %% 10) + 1
#' pat$year <- 2020 + pat$patent %% 3
#' pat$funding <- pat$patent %% 2
#' pat$citations <- 5 + 0.4 * pat$funding + pat$patent / 10
#' g1 <- xhdfe(citations ~ funding | inventor, data = pat,
#'             group = ~ patent, individual = ~ inventor)
#' g2 <- xhdfe(citations ~ funding | inventor, data = pat,
#'             group = ~ patent, individual = ~ inventor,
#'             aggregation = "sum")
#' g3 <- xhdfe(citations ~ funding | year, data = pat, group = ~ patent)
#'
#' \donttest{
#' ## 11. Side-by-side with fixest (coefficients agree to reghdfe-level
#' ## precision under the default tolerance mode)
#' if (requireNamespace("fixest", quietly = TRUE)) {
#'   mf <- fixest::feols(y ~ x1 + x2 | firm + year, data = d)
#'   mx <- xhdfe(y ~ x1 + x2 | firm + year, data = d, cluster = ~ firm)
#'   print(cbind(fixest = coef(mf), xhdfe = coef(mx)[c("x1", "x2")]))
#' }
#' }
#'
#' \dontrun{
#' ## 12. CUDA GPU absorption (requires a CUDA build of the package; the
#' ## request fails closed instead of silently falling back to the CPU)
#' mg <- xhdfe(y ~ x1 + x2 | worker + firm, data = d, backend = "cuda")
#' stopifnot(mg$gpu_used == 1, mg$gpu_status == "used")
#' }
#'
#' @name xhdfe
NULL

# ---------------------------------------------------------------------------
# Matrix interface: xhdfe_fit()
# ---------------------------------------------------------------------------

#' Matrix interface to the xhdfe estimator
#'
#' @description
#' \code{xhdfe_fit()} is the low-level matrix interface to the xhdfe C++
#' backend, mirroring the Python package's
#' \code{HdfeRegressor().fit(y, X, fes, ...)} call. It takes the outcome,
#' design matrix, and id vectors directly -- no formula, no
#' \code{data.frame} evaluation -- and returns exactly the same
#' \code{"xhdfe"} object as \code{\link{xhdfe}}. Use it when the design is
#' already materialized (simulations, benchmark harnesses, code ported from
#' Python) or when formula processing overhead matters.
#'
#' Unlike \code{\link{xhdfe}}, this interface performs \strong{no missing
#' value handling}: any \code{NA} in \code{y} or \code{X} is an error, and
#' no sample marking against an original data frame takes place (filter
#' beforehand, or use the formula interface). There are also no
#' \code{data}, \code{subset}, or \code{na.action} arguments.
#'
#' @usage
#' xhdfe_fit(y, X, fes = NULL,
#'           weights = NULL, weights_type = c("analytic", "frequency"),
#'           cluster = NULL, vcov = NULL,
#'           instruments = NULL, endogenous = NULL,
#'           slopes = NULL,
#'           group = NULL, individual = NULL, aggregation = "mean",
#'           save_fe = FALSE, groupvar = FALSE,
#'           tol = 1e-8, maxiter = 100000,
#'           tolerance_mode = "reghdfe-comparable",
#'           convergence = "auto", check_interval = 1,
#'           absorption_method = "auto", symmetric_sweep = FALSE,
#'           jacobi_relaxation = 0,
#'           fit_intercept = TRUE,
#'           fe_tolerance = 1e-6, fe_recovery_method = "hybrid",
#'           drop_singletons = TRUE, keep_singletons = NULL,
#'           dof = NULL, ssc = NULL, stats_style = "reghdfe",
#'           level = 95,
#'           threads = 0, default_threads = 0, max_threads = 0,
#'           min_parallel_rows = 20000, target_rows_per_thread = 500000,
#'           backend = c("default", "cpu", "cuda", "metal"))
#'
#' @param y numeric outcome vector (no \code{NA}s).
#' @param X numeric design matrix (or \code{data.frame}/vector coerced to
#'   one) with one row per observation and no \code{NA}s. Column names
#'   become the coefficient names (defaults \code{x1}, \code{x2}, ... when
#'   absent). Do \emph{not} include an intercept column; the constant is
#'   controlled by \code{fit_intercept}. Endogenous regressors (IV) are
#'   columns of \code{X} flagged through \code{endogenous}.
#' @param fes the absorbed fixed effects: a list with one id vector per
#'   absorbed dimension (list names become the fixed-effect labels used in
#'   \code{fe_labels} / \code{fixef()}), a matrix (one dimension per
#'   column), or a single vector. Ids are encoded like in
#'   \code{\link{xhdfe}}: integer vectors pass through verbatim, factors
#'   use their level codes, and character/logical/non-integral numeric
#'   vectors are encoded by first appearance. \code{NULL} means no absorbed
#'   fixed effects (plain OLS).
#' @param weights optional numeric weight vector, one entry per
#'   observation; interpreted according to \code{weights_type}.
#' @param weights_type \code{"analytic"} (Stata \code{aweight}) or
#'   \code{"frequency"} (Stata \code{fweight}, exact replicated-rows
#'   semantics); see \code{\link{xhdfe}}.
#' @param cluster the cluster dimensions: a vector, a matrix (one dimension
#'   per column), a \code{data.frame}, or a list of vectors; several
#'   dimensions request multiway clustering. No formula form here (there is
#'   no \code{data} to evaluate it in). Supplying \code{cluster} without
#'   \code{vcov} implies \code{vcov = "cluster"}.
#' @param vcov \code{NULL}, \code{"unadjusted"} (aliases \code{"iid"},
#'   \code{"ols"}, \code{"homoskedastic"}, \code{"classical"},
#'   \code{"unadj"}), \code{"robust"} (aliases \code{"hc1"},
#'   \code{"heteroskedastic"}), or \code{"cluster"} (alias
#'   \code{"clustered"}); see \code{\link{xhdfe}}.
#' @param instruments numeric matrix of instruments (one row per
#'   observation); requires \code{endogenous}. The instrument count must be
#'   at least the number of endogenous regressors.
#' @param endogenous which columns of \code{X} are endogenous: a vector of
#'   1-based column positions or of column names. Required together with
#'   \code{instruments}; supplying it without \code{instruments} is an
#'   error. (In the formula interface the endogenous variables instead
#'   form the third formula part and are appended to the design
#'   automatically.)
#' @param slopes heterogeneous (group-specific) slope specifications: a
#'   list of entries, each itself a list with elements
#'   \describe{
#'     \item{\code{fe}}{1-based index into \code{fes} naming the carrier
#'       dimension (alias \code{fe_index}).}
#'     \item{\code{values}}{the numeric slope variable, one entry per
#'       observation (alias \code{x}).}
#'     \item{\code{include_intercept}}{logical (alias \code{intercept};
#'       default \code{FALSE}): \code{TRUE} absorbs the carrier's
#'       intercepts as well (Stata \code{fe##c.x}); \code{FALSE} absorbs
#'       the slopes only (Stata \code{fe#c.x}).}
#'   }
#'   A single specification may be passed unwrapped (a plain
#'   \code{list(fe = , values = )}). The core absorbs at most one slope per
#'   carrier dimension: to attach several slope variables to the same ids,
#'   repeat the id vector as separate entries of \code{fes} and point one
#'   slope entry at each, letting the intercept ride on the first -- the
#'   same expansion the formula interface performs for
#'   \code{fe[x1, x2]} and Stata performs for
#'   \code{absorb(fe##c.x1 fe#c.x2)}.
#' @param group,individual,aggregation group-level-outcome machinery, as in
#'   \code{\link{xhdfe}} but with plain id vectors (no formula or column
#'   name forms). \code{individual} requires \code{group} and is appended
#'   to \code{fes} automatically when not already present.
#' @param save_fe logical; recover fixed-effect contributions (Stata
#'   \code{savefe}); see \code{\link{xhdfe}}.
#' @param groupvar logical; store first-mobility-group ids; see
#'   \code{\link{xhdfe}}.
#' @param tol,maxiter,tolerance_mode absorber tolerance, iteration cap, and
#'   tolerance mode; see \code{\link{xhdfe}} and its \emph{Tolerance modes}
#'   section.
#' @param convergence heterogeneous-slope stopping rule (\code{"auto"},
#'   \code{"normchange"}, \code{"reghdfe"}, \code{"both"}); see
#'   \code{\link{xhdfe}}.
#' @param check_interval iterations between convergence checks; see
#'   \code{\link{xhdfe}}.
#' @param absorption_method absorption method (\code{"auto"},
#'   \code{"gauss-seidel"}, \code{"symmetric-gauss-seidel"},
#'   \code{"jacobi"}, \code{"schwarz"}, \code{"lsmr"}, \code{"mlsmr"},
#'   \code{"auto-mlsmr"}, with aliases); see \code{\link{xhdfe}}.
#' @param symmetric_sweep forward-plus-backward Gauss-Seidel sweeps; see
#'   \code{\link{xhdfe}}.
#' @param jacobi_relaxation Jacobi relaxation parameter; see
#'   \code{\link{xhdfe}}.
#' @param fit_intercept logical (default \code{TRUE}): report the
#'   reghdfe-style constant (and, without fixed effects, include an
#'   intercept in the regression). \code{FALSE} is the analogue of Stata's
#'   \code{noconstant}; the formula interface controls this with \code{0 +}
#'   in the regressor part instead.
#' @param fe_tolerance,fe_recovery_method fixed-effect recovery controls
#'   for \code{save_fe}; see \code{\link{xhdfe}}.
#' @param drop_singletons,keep_singletons singleton handling (reghdfe
#'   semantics); see \code{\link{xhdfe}}.
#' @param dof degrees-of-freedom adjustment tokens (\code{"all"},
#'   \code{"none"}, \code{"firstpair"}, \code{"pairwise"},
#'   \code{"clusters"}, \code{"continuous"}); see \code{\link{xhdfe}}.
#' @param ssc small-sample corrections (named list or Stata-style string);
#'   see \code{\link{xhdfe}}.
#' @param stats_style \code{"reghdfe"} (default) or \code{"legacy"}; see
#'   \code{\link{xhdfe}}.
#' @param level confidence level in percent or as a fraction; see
#'   \code{\link{xhdfe}}.
#' @param threads,default_threads,max_threads number of absorber threads
#'   and auto-threading bounds; see \code{\link{xhdfe}}.
#' @param min_parallel_rows,target_rows_per_thread parallel-absorption
#'   heuristics; see \code{\link{xhdfe}}.
#' @param backend \code{"default"}, \code{"cpu"}, \code{"cuda"}, or
#'   \code{"metal"}, with the same fail-closed GPU contract as
#'   \code{\link{xhdfe}}; see its \emph{GPU backends} section.
#'
#' @return The same \code{"xhdfe"} object documented in
#'   \code{\link{xhdfe}} (see its \emph{Value} section for every field),
#'   with two differences: the formula-interface extras (\code{fml},
#'   \code{terms}, \code{data_name}, \code{group_name},
#'   \code{individual_name}, \code{aggregation}) are absent, and
#'   consequently \code{predict(..., newdata = )} is unavailable (the
#'   cache-based prediction types \code{xb}, \code{xbd}, \code{d},
#'   \code{residuals}, \code{dresiduals}, \code{score}, and \code{stdp}
#'   still work for non-grouped fits). The \code{sample} field indexes into
#'   the rows of \code{y}/\code{X}.
#'
#' @seealso \code{\link{xhdfe}} for the formula interface and the full
#'   documentation of the shared arguments, sections, and returned object;
#'   \code{xhdfe_info()} for version/build information.
#'
#' @author Miguel Portela (\email{miguel.portela@@eeg.uminho.pt}) and
#'   Tiago Tavares (\email{tgstavares@@eeg.uminho.pt}).
#'
#' @examples
#' set.seed(1)
#' n <- 300
#' X <- cbind(x1 = rnorm(n), x2 = rnorm(n))
#' firm <- sample(25, n, replace = TRUE)
#' year <- sample(6, n, replace = TRUE)
#' y <- 0.5 * X[, 1] - 0.2 * X[, 2] + 0.1 * firm + 0.2 * year + rnorm(n)
#'
#' ## Two-way fixed effects with clustered standard errors
#' m <- xhdfe_fit(y, X, fes = list(firm = firm, year = year),
#'                cluster = firm)
#' coef(m)
#'
#' ## Heterogeneous slope: firm intercepts + firm-specific slopes on z
#' ## (Stata: absorb(firm##c.z year))
#' z <- rnorm(n)
#' m2 <- xhdfe_fit(y, X, fes = list(firm = firm, year = year),
#'                 slopes = list(list(fe = 1, values = z,
#'                                    include_intercept = TRUE)))
#'
#' ## IV/2SLS: the "xe" column of X is endogenous, instrumented by zi
#' zi <- rnorm(n)
#' xe <- 0.6 * zi + rnorm(n)
#' Xiv <- cbind(X, xe = xe)
#' yiv <- y + 0.7 * xe
#' miv <- xhdfe_fit(yiv, Xiv, fes = list(firm = firm),
#'                  endogenous = "xe", instruments = cbind(z1 = zi))
#' coef(miv)
#'
#' @name xhdfe_fit
NULL
