# AKM + leave-out (KSS) variance decomposition front-end.
#
# R-idiom layer over the compiled hdfe::akm module; mirrors the Python
# py_hdfe_v11.akm_kss / akm_leave_out_set free functions so that the R,
# Python and (future) Stata surfaces sit on the identical compiled estimator.

.akm_id_codes <- function(x, label) {
  if (is.factor(x)) {
    codes <- as.integer(x)
    if (anyNA(codes)) {
      stop(sprintf("%s contains missing ids", label), call. = FALSE)
    }
    return(codes)
  }
  if (is.character(x)) {
    codes <- as.integer(factor(x))
    if (anyNA(codes)) {
      stop(sprintf("%s contains missing ids", label), call. = FALSE)
    }
    return(codes)
  }
  if (any(!is.finite(x))) {
    stop(sprintf("%s contains missing or non-finite ids", label), call. = FALSE)
  }
  # Numeric ids: keep the caller's integer codes when they fit int32; otherwise
  # (e.g. NISS/NIF person codes larger than 2^31, or non-integer labels) compact
  # to dense 1..N codes. The compiled core relabels ids to dense indices
  # internally, so the recoding never changes the estimates.
  if (all(x == floor(x)) && max(abs(x)) <= .Machine$integer.max) {
    return(as.integer(x))
  }
  as.integer(factor(x))
}

#' Largest leave-one-out connected set (KSS)
#'
#' Computes the largest connected set of the worker-firm bipartite graph such
#' that removing any single worker keeps it connected (the Kline-Saggio-
#' Soelvsten (2020) leave-out sample, following Saggio's LeaveOutTwoWay:
#' articulation workers are removed iteratively and workers observed only
#' once are dropped).
#'
#' @param worker Worker identifiers (integer, factor or character), length n.
#' @param firm Firm identifiers (integer, factor or character), length n.
#' @return A list with a logical \code{keep} mask over the input rows and the
#'   sample counts (\code{n_obs}, \code{n_workers}, \code{n_firms},
#'   \code{n_matches}, \code{n_movers}, \code{n_stayers}, ...).
#' @seealso \code{\link{xhdfe_akm_kss}}
#' @export
xhdfe_akm_leave_out_set <- function(worker, firm) {
  if (length(worker) != length(firm)) {
    stop("worker and firm must have the same length", call. = FALSE)
  }
  .xhdfe_cpp_akm_leave_out_set(.akm_id_codes(worker, "worker"),
                               .akm_id_codes(firm, "firm"))
}

#' AKM estimation with leave-out (KSS) variance decomposition
#'
#' Estimates the two-way AKM model \eqn{y = \alpha_{worker} + \psi_{firm} +
#' X\beta + \epsilon} on the leave-out connected set and reports the variance
#' decomposition in three flavours: plug-in (biased baseline), AGSU
#' (Andrews et al. 2008, homoskedastic correction) and KSS (Kline, Saggio &
#' Soelvsten 2020, heteroskedastic leave-out correction). Numerical semantics
#' follow Saggio's LeaveOutTwoWay: leave-a-match-out by default (collapse to
#' match means with sqrt(match-length) FGLS weighting), person-year weighted
#' components with denominator (n - 1), and the JLA leverage approximation
#' with deterministic seeding for large samples.
#'
#' @param y Numeric outcome, length n.
#' @param worker,firm Identifiers (integer, factor or character), length n.
#' @param X Optional numeric matrix of controls (n rows); partialled out at
#'   the person-year level with the xhdfe absorber (FWL) before the two-way
#'   machinery runs.
#' @param leave_out_level \code{"match"} (default, Saggio's default) or
#'   \code{"obs"}.
#' @param leverages \code{"auto"} (exact when the input has at most
#'   \code{exact_max_rows} rows, else JLA), \code{"exact"} or \code{"jla"}.
#' @param jla_draws Rademacher simulations for the JLA path (default 200).
#' @param seed Seed for the JLA draws; results are reproducible for any
#'   thread count.
#' @param prune Compute the leave-out connected set (default TRUE). Set to
#'   FALSE only when the input is already a leave-out sample.
#' @param exact_max_rows,direct_max_firms,direct_max_nnz,cg_tol,cg_max_iter
#'   Solver knobs; see the package vignette sources.
#' @param num_threads OpenMP threads (0 = library default).
#' @param fwl_tol,fwl_max_iter Absorber controls for the covariate step.
#' @param compute_se Component standard errors (KSS leave-out inference).
#' @param se_nsim Simulation draws for the SE quadratic part (default 1000).
#' @param eigen_diagnostics Weak-identification diagnostics and the
#'   Andrews-Mikusheva q = 1 confidence intervals (leave_out_COMPLETE
#'   eigen_diagno path): top eigenvalue/shares of Atilde, Lindeberg
#'   condition, gamma^2, F statistic, theta_1 and the AM CI per component.
#'   Implies \code{compute_se = TRUE}.
#' @param eig_trace_nsim Hutchinson draws for tr(Atilde^2) (default 100,
#'   the oracle default).
#' @param se_sigma_lowess Use the llr_fit mode-0 lowess surface fit of
#'   sigma_i on (Pii, Bii) instead of the binned mode-4 fit (O(n^2);
#'   sensitivity analysis on small/medium samples).
#' @param fweights Optional positive-integer frequency weights (row i stands
#'   for fweights[i] identical person-year observations). Match-level point
#'   decomposition only; equals the row-expanded run exactly.
#' @section Advanced performance environment variables:
#'   Defaults are tuned and none changes the default numeric output.
#'   \code{XHDFE_AKM_TEAM} caps the OpenMP team size of the per-iteration
#'   solver regions (the default caps it by the edge work so a large thread
#'   pool does not oversubscribe small/medium graphs \emph{--} the dominant
#'   speed lever below ~10M rows; \code{0} = uncapped, \code{k} forces
#'   \code{k}). \code{XHDFE_AKM_JLA_BLOCK} / \code{XHDFE_AKM_SE_BLOCK} set the
#'   multi-RHS block size for the JLA leverage and the SE/eigen/lincom solves
#'   (default 8; \code{0} = pre-2.14 sequential). \code{XHDFE_AKM_SCATTER_CSR}
#'   (default on) selects the parallel CSR-ordered Rademacher scatter at scale.
#'   The solves are batched so results are identical for any block size and
#'   thread count. A \code{warning()} fires on non-convergence
#'   (check \code{$converged}).
#' @param gpu Solve the two-way systems on the CUDA backend when available.
#' @return An object of class \code{xhdfe_akm_kss}: a list with the sample
#'   summary, observation-level \code{alpha}/\code{psi} on the kept rows,
#'   control coefficients \code{beta}, and the \code{plugin}/\code{agsu}/
#'   \code{kss} component tables. Each component table carries
#'   \code{var_alpha}, \code{var_psi}, \code{cov_alpha_psi} plus the derived
#'   \code{corr_alpha_psi}, \code{var_alpha_plus_psi} and the shares of
#'   \code{var_y} (\code{share_var_alpha}, \code{share_var_psi},
#'   \code{share_2cov}). Also returned: \code{var_y}, \code{sigma2_ho},
#'   row-level leverages \code{pii} and \code{sigma_i}, and solver
#'   diagnostics (\code{leverages_exact}, \code{gpu_used}, \code{converged},
#'   ...). With \code{compute_se = TRUE} a \code{component_se} list holds the
#'   KSS component standard errors and corrected point estimates; with
#'   \code{eigen_diagnostics = TRUE} a \code{weak_id} list holds, per
#'   component, the top eigenvalue and shares of Atilde, the Lindeberg
#'   condition, \code{gamma_sq}, the F statistic, \code{theta_1}, the
#'   Andrews-Mikusheva q = 1 confidence bounds (\code{ci_lb}/\code{ci_ub})
#'   and the curvature.
#' @references Kline P., Saggio R., Soelvsten M. (2020), Econometrica;
#'   Andrews M. et al. (2008), JRSS-A; Abowd J., Kramarz F., Margolis D.
#'   (1999), Econometrica.
#' @examples
#' \donttest{
#' n_w <- 100; n_f <- 12; reps <- 4
#' worker <- rep(seq_len(n_w), each = reps)
#' firm <- sample(seq_len(n_f), n_w * reps, replace = TRUE)
#' y <- rnorm(n_w)[worker] + rnorm(n_f)[firm] + rnorm(n_w * reps)
#' fit <- xhdfe_akm_kss(y, worker, firm)
#' print(fit)
#' }
#' @export
xhdfe_akm_kss <- function(y, worker, firm, X = NULL,
                          leave_out_level = "match",
                          leverages = "auto",
                          jla_draws = 200L,
                          seed = 20260705,
                          prune = TRUE,
                          exact_max_rows = 10000L,
                          direct_max_firms = 50000L,
                          direct_max_nnz = 4e7,
                          cg_tol = 1e-10,
                          cg_max_iter = 0L,
                          num_threads = 0L,
                          fwl_tol = 1e-10,
                          fwl_max_iter = 100000L,
                          compute_se = FALSE,
                          se_nsim = 1000L,
                          eigen_diagnostics = FALSE,
                          eig_trace_nsim = 100L,
                          se_sigma_lowess = FALSE,
                          gpu = FALSE,
                          fweights = NULL) {
  y <- as.numeric(y)
  if (length(worker) != length(y) || length(firm) != length(y)) {
    stop("y, worker and firm must have the same length", call. = FALSE)
  }
  if (anyNA(y)) {
    stop("y contains missing values; supply complete cases", call. = FALSE)
  }
  if (!is.null(X)) {
    X <- as.matrix(X)
    storage.mode(X) <- "double"
    if (nrow(X) != length(y)) {
      stop("X must have the same number of rows as y", call. = FALSE)
    }
    if (anyNA(X)) {
      stop("X contains missing values; supply complete cases", call. = FALSE)
    }
  }
  opts <- list(leave_out_level = as.character(leave_out_level),
               leverages = as.character(leverages),
               jla_draws = as.integer(jla_draws),
               seed = as.double(seed),
               prune = isTRUE(prune),
               exact_max_rows = as.integer(exact_max_rows),
               direct_max_firms = as.integer(direct_max_firms),
               direct_max_nnz = as.double(direct_max_nnz),
               cg_tol = as.double(cg_tol),
               cg_max_iter = as.integer(cg_max_iter),
               num_threads = as.integer(num_threads),
               fwl_tol = as.double(fwl_tol),
               fwl_max_iter = as.integer(fwl_max_iter),
               compute_se = isTRUE(compute_se),
               se_nsim = as.integer(se_nsim),
               eigen_diagnostics = isTRUE(eigen_diagnostics),
               eig_trace_nsim = as.integer(eig_trace_nsim),
               se_sigma_lowess = isTRUE(se_sigma_lowess),
               gpu = isTRUE(gpu))
  if (!is.null(fweights)) {
    fweights <- as.numeric(fweights)
    if (length(fweights) != length(y)) {
      stop("fweights must have the same length as y", call. = FALSE)
    }
  }
  out <- .xhdfe_cpp_akm_kss(y,
                            .akm_id_codes(worker, "worker"),
                            .akm_id_codes(firm, "firm"),
                            X, opts, fweights)
  class(out) <- "xhdfe_akm_kss"
  # Surface non-convergence loudly, matching the Stata front-end and the
  # Gelbach treatment (a silently returned non-converged decomposition is a
  # footgun); the flag stays on the object for programmatic checks.
  if (isFALSE(out$converged)) {
    warning("xhdfe_akm_kss: the AKM/KSS decomposition did not converge - ",
            "results are unreliable (see $notes).", call. = FALSE)
  }
  out
}

#' @export
print.xhdfe_akm_kss <- function(x, digits = 6, ...) {
  s <- x$sample
  cat("AKM + leave-out (KSS) variance decomposition\n")
  cat(sprintf("Leave-out sample: %s obs, %s workers (%s movers), %s firms, %s matches\n",
              format(s$n_obs, big.mark = ","), format(s$n_workers, big.mark = ","),
              format(s$n_movers, big.mark = ","), format(s$n_firms, big.mark = ","),
              format(s$n_matches, big.mark = ",")))
  tab <- rbind(plugin = unlist(x$plugin),
               agsu = unlist(x$agsu),
               kss = unlist(x$kss))
  print(round(tab[, c("var_alpha", "var_psi", "cov_alpha_psi")], digits))
  cat(sprintf("var(y) = %s; leverages: %s%s; converged: %s\n",
              format(x$var_y, digits = digits),
              if (isTRUE(x$leverages_exact)) "exact" else "JLA",
              if (isTRUE(x$leverages_exact)) ""
              else sprintf(" (%d draws, seed %.0f)", x$jla_draws_used, x$seed),
              x$converged))
  if (nzchar(x$notes)) cat("notes:", x$notes, "\n")
  invisible(x)
}
