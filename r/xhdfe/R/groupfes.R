# Group/individual fixed-effect decomposition (extractfes parity).

#' Decompose group/individual fixed effects
#'
#' Recovers individual-level and FE-level effect estimates for the
#' group-level-outcome estimator (the analogue of the Python module's
#' \code{extract_group_individual_fes}). It refits the grouped model
#' internally and then runs the iterative decomposition with the same
#' tuning knobs exposed by the Python binding.
#'
#' @param y,X,fes,group,individual,weights,aggregation the model inputs, in
#'   the matrix form of \code{\link{xhdfe_fit}}. \code{group} and
#'   \code{individual} are required; \code{individual} must also appear in
#'   \code{fes} (it is appended automatically by the high-level interface).
#' @param tol_main,tol_start,tol_final main-loop and solver tolerances
#'   (defaults \code{1e-9}, \code{1e-3}, \code{1e-9}).
#' @param max_iter_main,max_iter_solver iteration caps (defaults
#'   \code{100000} and \code{1000}).
#' @param verbose \code{0} for silent.
#' @param accel acceleration scheme: \code{0} none, \code{1} linear,
#'   \code{2} geometric (default).
#' @param start_accel,every_accel when acceleration starts and how often it
#'   is applied (defaults \code{5}, \code{5}).
#' @param factor tighten the solver tolerance when the MSE falls below
#'   \code{factor * tol} (default \code{1}).
#' @param a1p1,a2p1,a2p2 acceleration safeguards (defaults \code{0.75},
#'   \code{1e-8}, \code{5}).
#' @param threads absorber thread count (0 = auto).
#' @param weights_type,fit_intercept,drop_singletons,keep_singletons,tol,maxiter,tolerance_mode,convergence,absorption_method,symmetric_sweep
#'   options for the internal grouped fit that precedes the decomposition;
#'   pass the SAME values used for the model being decomposed so the
#'   extractor conditions on the identical estimation state (defaults match
#'   \code{\link{xhdfe_fit}}).
#' @return An object of class \code{xhdfe_group_fes}: a list with
#'   \code{individual_ids}, \code{individual_effects}, \code{fe_level_ids},
#'   \code{fe_level_effects}, \code{iterations}, \code{converged} and
#'   \code{mse}.
#' @seealso \code{\link{xhdfe}} (arguments \code{group},
#'   \code{individual}, \code{aggregation})
#' @name xhdfe_group_fes
xhdfe_group_fes <- function(y, X, fes, group, individual,
                            weights = NULL, aggregation = "mean",
                            weights_type = c("analytic", "frequency"),
                            fit_intercept = TRUE,
                            drop_singletons = TRUE, keep_singletons = NULL,
                            tol = 1e-8, maxiter = 100000,
                            tolerance_mode = "reghdfe-comparable",
                            convergence = "auto",
                            absorption_method = "auto",
                            symmetric_sweep = FALSE,
                            tol_main = 1e-9, tol_start = 1e-3,
                            tol_final = 1e-9,
                            max_iter_main = 100000, max_iter_solver = 1000,
                            verbose = 0, accel = 2, start_accel = 5,
                            every_accel = 5, factor = 1.0, a1p1 = 0.75,
                            a2p1 = 1e-8, a2p2 = 5,
                            threads = 0) {
  weights_type <- match.arg(weights_type)
  y <- as.numeric(y)
  if (is.data.frame(X)) X <- as.matrix(X)
  if (!is.matrix(X)) X <- matrix(as.numeric(X), ncol = 1L)
  storage.mode(X) <- "double"

  if (is.matrix(fes)) fes <- lapply(seq_len(ncol(fes)), function(j) fes[, j])
  if (!is.list(fes)) fes <- list(fes)
  fes_use <- lapply(fes, to_ids, label = "fixed effect")

  gi <- list(tol_main = tol_main, tol_start = tol_start, tol_final = tol_final,
             max_iter_main = as.integer(max_iter_main),
             max_iter_solver = as.integer(max_iter_solver),
             verbose = as.integer(verbose), accel = as.integer(accel),
             start_accel = as.integer(start_accel),
             every_accel = as.integer(every_accel),
             factor = factor, a1p1 = a1p1, a2p1 = a2p1,
             a2p2 = as.integer(a2p2))
  opts <- list(se_type = "unadjusted", num_threads = as.integer(threads),
               fit_intercept = isTRUE(fit_intercept),
               drop_singletons = isTRUE(drop_singletons),
               tol = tol, max_iter = as.integer(maxiter),
               tolerance_mode = tolerance_mode, convergence = convergence,
               absorption_method = absorption_method,
               symmetric_sweep = isTRUE(symmetric_sweep),
               weights_are_frequencies = identical(weights_type, "frequency"))
  if (!is.null(keep_singletons)) opts$keepsingletons <- isTRUE(keep_singletons)

  res <- .xhdfe_cpp_extract_group_fes(
    y, X, fes_use,
    to_ids(group, "group"), to_ids(individual, "individual"),
    if (is.null(weights)) NULL else as.numeric(weights),
    aggregation, gi, opts)

  structure(res, class = "xhdfe_group_fes")
}

print.xhdfe_group_fes <- function(x, ...) {
  cat("xhdfe group/individual FE decomposition\n")
  cat(sprintf("Individuals: %d | FE dimensions: %d\n",
              length(x$individual_ids), length(x$fe_level_ids)))
  cat(sprintf("%s in %d iterations (mse %.3g)\n",
              ifelse(x$converged, "Converged", "NOT converged"),
              x$iterations, x$mse))
  invisible(x)
}
