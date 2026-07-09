# Gelbach (2016) conditional decomposition, HDFE-aware (M9B front-end).

#' Gelbach conditional decomposition (HDFE-aware)
#'
#' Decomposes the change in the base-specification coefficients when
#' covariate groups and/or absorbed fixed-effect blocks are added, following
#' Gelbach (2016). Inference reproduces Gelbach's \code{b1x2} exactly
#' (homoskedastic, robust and cluster; \code{gamma0}/\code{cov0} options);
#' absorbed FE blocks receive the gamma0 (aux-regression-only) variance.
#' Same compiled implementation as the Python \code{xhdfe.gelbach} module.
#'
#' @param y Numeric outcome, length n.
#' @param x1 Base covariates (matrix or vector; constant implicit).
#' @param x2_groups Named list of observed covariate groups (each a matrix
#'   or vector with n rows).
#' @param fes Named list of absorbed fixed-effect id vectors.
#' @param vce \code{"unadjusted"} (default), \code{"robust"} or
#'   \code{"cluster"} (requires \code{cluster}).
#' @param cluster Cluster ids (integer/factor/character), length n.
#' @param gamma0,cov0 b1x2's variance options.
#' @param weights Optional Stata-style weights: analytic by default,
#'   frequency when \code{fweights = TRUE}; matches b1x2's weighted
#'   estimators exactly.
#' @param fweights Treat \code{weights} as frequency weights.
#' @param num_threads OpenMP threads (0 = library default).
#' @return An object of class \code{xhdfe_gelbach}: per-group contribution
#'   vectors over \code{[x1..., _cons]} with standard errors, the total
#'   (= b_base - b_full), the full covariance, and diagnostics.
#' @export
xhdfe_gelbach <- function(y, x1, x2_groups = NULL, fes = NULL,
                          vce = "unadjusted", cluster = NULL,
                          gamma0 = FALSE, cov0 = FALSE, num_threads = 0L,
                          weights = NULL, fweights = FALSE) {
  y <- as.numeric(y)
  x1 <- as.matrix(x1)
  storage.mode(x1) <- "double"
  if (nrow(x1) != length(y)) {
    stop("y and x1 must have the same number of rows", call. = FALSE)
  }
  x2_groups <- as.list(x2_groups %||% list())
  fes <- as.list(fes %||% list())
  if (!length(x2_groups) && !length(fes)) {
    stop("provide at least one x2 group or fixed-effect dimension", call. = FALSE)
  }
  sizes <- integer(0)
  X2 <- NULL
  for (g in x2_groups) {
    g <- as.matrix(g)
    storage.mode(g) <- "double"
    if (nrow(g) != length(y)) stop("x2 group has wrong length", call. = FALSE)
    sizes <- c(sizes, ncol(g))
    X2 <- if (is.null(X2)) g else cbind(X2, g)
  }
  fe_list <- lapply(fes, function(ids) .akm_id_codes(ids, "fe"))
  if (!is.null(cluster) && !identical(vce, "cluster")) {
    stop("cluster ids supplied but vce != \"cluster\"", call. = FALSE)
  }
  if (identical(vce, "cluster") && is.null(cluster)) {
    stop("vce = \"cluster\" requires cluster ids", call. = FALSE)
  }
  cl <- if (is.null(cluster)) NULL else .akm_id_codes(cluster, "cluster")
  w <- if (is.null(weights)) NULL else as.numeric(weights)
  out <- .xhdfe_cpp_gelbach(y, x1, X2, as.integer(sizes), fe_list, cl,
                            as.character(vce), isTRUE(gamma0), isTRUE(cov0),
                            as.integer(num_threads), w, isTRUE(fweights))
  k1 <- ncol(x1) + 1L
  nm <- c(names(x2_groups), names(fes))
  delta <- out$delta
  colnames(delta) <- nm
  rownames(delta) <- c(paste0("x1_", seq_len(ncol(x1))), "_cons")
  se <- vapply(seq_along(nm), function(g) {
    idx <- ((g - 1L) * k1 + 1L):(g * k1)
    sqrt(diag(out$cov[idx, idx, drop = FALSE]))
  }, numeric(k1))
  dimnames(se) <- dimnames(delta)
  out$delta <- delta
  out$se <- se
  out$total_se <- sqrt(diag(out$total_cov))
  out$names <- nm
  out$vce <- vce
  class(out) <- "xhdfe_gelbach"
  out
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' @export
print.xhdfe_gelbach <- function(x, digits = 6, ...) {
  cat("Gelbach conditional decomposition (xhdfe backend)\n")
  cat(sprintf("n = %s, vce = %s%s\n", format(x$n_obs, big.mark = ","),
              x$vce, if (isTRUE(x$gamma0)) " (gamma0)" else ""))
  cat("\nContributions (delta):\n")
  print(round(x$delta, digits))
  cat("\nStandard errors:\n")
  print(round(x$se, digits))
  cat(sprintf("\nTotal (= b_base - b_full): %s\n",
              paste(round(x$total, digits), collapse = " ")))
  if (!isTRUE(x$converged)) {
    warning("xhdfe_gelbach: the decomposition did not converge or failed a ",
            "convergence cross-check - results are unreliable (see notes).",
            call. = FALSE)
  }
  if (nzchar(x$notes)) cat("notes:", x$notes, "\n")
  invisible(x)
}
