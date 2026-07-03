# Within-transform (partialling out) without running the regression.

#' Within-transform variables (partialling out fixed effects)
#'
#' Applies the xhdfe absorber to a set of variables without running the
#' final regression, exposing the core's \code{partial_out} entry point.
#' This is the analogue of \code{fixest::demean()}, with the full xhdfe
#' grammar: multiple dimensions, combined effects \code{f1^f2}, and
#' heterogeneous slopes \code{fe[x]} / \code{fe[[x]]}.
#'
#' @param fml a formula \code{y ~ x1 + x2 | fe1 + fe2}; every variable to
#'   the left of the \code{|} (response and regressors alike) is transformed.
#'   No IV part is accepted.
#' @param data,weights as in \code{\link{xhdfe}}.
#' @param tol,maxiter,tolerance_mode,absorption_method,symmetric_sweep,threads,backend
#'   absorber controls, as in \code{\link{xhdfe}}.
#' @param drop_singletons defaults to \code{FALSE} here (transform-only use
#'   normally wants all rows kept), unlike \code{xhdfe()}.
#' @return An object of class \code{xhdfe_demean}: a list with
#'   \code{demeaned} (matrix of transformed columns, one per variable),
#'   \code{rows} (the input rows present in \code{demeaned}, after NA
#'   filtering and, when \code{drop_singletons = TRUE}, singleton dropping),
#'   \code{num_singletons}, \code{fe_labels}, \code{fe_num_levels},
#'   \code{iterations}, \code{converged}, \code{absorption_method_used},
#'   \code{schwarz_used}, \code{mlsmr_used}, \code{gpu_used} and
#'   \code{threads_used}.
#' @examples
#' d <- data.frame(f1 = rep(1:25, 20), f2 = rep(1:20, each = 25))
#' d$x <- rnorm(500); d$y <- d$x + d$f1 / 10 + d$f2 / 5 + rnorm(500)
#' dm <- xhdfe_demean(y ~ x | f1 + f2, d)
#' # regression on the transformed data reproduces the within estimate:
#' coef(lm(dm$demeaned[, "y"] ~ 0 + dm$demeaned[, "x"]))
#' @seealso \code{\link{xhdfe}}
#' @name xhdfe_demean
xhdfe_demean <- function(fml, data = NULL, weights = NULL,
                         tol = 1e-8, maxiter = 100000,
                         tolerance_mode = "reghdfe-comparable",
                         absorption_method = "auto",
                         symmetric_sweep = FALSE,
                         drop_singletons = FALSE,
                         threads = 0,
                         backend = c("default", "cpu", "cuda", "metal")) {
  backend <- match.arg(backend)
  env <- environment(fml)
  if (is.null(env)) env <- parent.frame()
  spec <- split_xhdfe_formula(fml)
  if (!is.null(spec$endogenous)) {
    stop("xhdfe_demean() takes no IV part", call. = FALSE)
  }
  if (is.null(spec$fe)) {
    stop("xhdfe_demean() requires a fixed-effects part: vars ~ 1 | fe1 + fe2 ",
         "or y ~ x | fe", call. = FALSE)
  }

  # All variables on the left of `|` (LHS and RHS alike) are transformed.
  lhs_vars <- eval(spec$lhs, data, env)
  rhs_fml <- stats::as.formula(call("~", spec$regressors), env = env)
  tf <- stats::terms(rhs_fml, data = data)
  mf <- stats::model.frame(tf, data = data, na.action = stats::na.pass)
  X <- stats::model.matrix(tf, mf)
  int_col <- match("(Intercept)", colnames(X))
  if (!is.na(int_col)) X <- X[, -int_col, drop = FALSE]

  y <- as.numeric(lhs_vars)
  n_input <- length(y)

  fe_raw <- list(); slope_meta <- list(); fe_labels <- character(0)
  for (term in split_plus_terms(spec$fe)) {
    parsed <- parse_fe_term(term)
    ids <- eval_fe_expr(parsed$fe_expr, data, env)
    if (length(parsed$slopes) == 0L) {
      fe_raw[[length(fe_raw) + 1L]] <- ids
      fe_labels <- c(fe_labels, parsed$label)
    } else {
      for (j in seq_along(parsed$slopes)) {
        values <- eval(parsed$slopes[[j]], data, env)
        fe_raw[[length(fe_raw) + 1L]] <- ids
        slope_meta[[length(slope_meta) + 1L]] <-
          list(fe_index = length(fe_raw),
               values = as.numeric(values),
               include_intercept = parsed$include_intercept && j == 1L)
        fe_labels <- c(fe_labels, parsed$label)
      }
    }
  }

  if (inherits(weights, "formula")) {
    wenv <- environment(weights)
    if (is.null(wenv)) wenv <- env
    weights <- eval(weights[[length(weights)]], data, wenv)
  } else if (is.character(weights) && length(weights) == 1L &&
             !is.null(data) && weights %in% names(data)) {
    weights <- data[[weights]]
  }
  if (!is.null(weights)) {
    if (!is.numeric(weights) || length(weights) != n_input) {
      stop("`weights` must be a numeric vector with one entry per observation",
           call. = FALSE)
    }
    weights <- as.numeric(weights)
  }

  ok <- !is.na(y) & stats::complete.cases(X)
  for (v in fe_raw) ok <- ok & !is.na(v)
  for (s in slope_meta) ok <- ok & !is.na(s$values)
  if (!is.null(weights)) ok <- ok & !is.na(weights)
  rows_used <- which(ok)

  fes_use <- lapply(fe_raw, function(v) to_ids(v[rows_used], "fixed effect"))
  slopes_use <- lapply(slope_meta, function(s) {
    list(fe_index = s$fe_index - 1L, values = s$values[rows_used],
         include_intercept = s$include_intercept)
  })

  opts <- list(tol = tol, max_iter = as.integer(maxiter),
               tolerance_mode = tolerance_mode,
               absorption_method = absorption_method,
               symmetric_sweep = isTRUE(symmetric_sweep),
               drop_singletons = isTRUE(drop_singletons),
               num_threads = as.integer(threads),
               se_type = "unadjusted")

  res <- run_with_backend(backend, function() {
    .xhdfe_cpp_partial_out(y[rows_used], X[rows_used, , drop = FALSE],
                           fes_use,
                           if (is.null(weights)) NULL
                           else as.numeric(weights)[rows_used],
                           slopes_use, opts)
  })

  out_mat <- cbind(res$y_tilde, res$X_tilde)
  colnames(out_mat) <- c(deparse1(spec$lhs), colnames(X))
  # With drop_singletons = TRUE the core removes rows; realign `rows` to the
  # rows actually present in `demeaned` (identity map otherwise).
  if (length(res$sample_index0) == nrow(out_mat)) {
    rows_used <- rows_used[res$sample_index0 + 1L]
  }
  structure(list(demeaned = out_mat,
                 rows = rows_used,
                 num_singletons = res$num_singletons,
                 fe_labels = fe_labels,
                 fe_num_levels = res$fe_num_levels,
                 iterations = res$iterations,
                 converged = res$converged,
                 absorption_method_used = res$absorption_method_used,
                 schwarz_used = res$schwarz_used,
                 mlsmr_used = res$mlsmr_used,
                 gpu_used = res$gpu_used,
                 threads_used = res$threads_used),
            class = "xhdfe_demean")
}

print.xhdfe_demean <- function(x, ...) {
  cat("xhdfe within-transform:", ncol(x$demeaned), "columns,",
      nrow(x$demeaned), "rows\n")
  cat(sprintf("Absorbed: %s | %s in %d iterations%s\n",
              paste(x$fe_labels, collapse = ", "),
              ifelse(x$converged, "converged", "NOT converged"),
              x$iterations,
              ifelse(x$gpu_used, " (GPU)", "")))
  invisible(x)
}
