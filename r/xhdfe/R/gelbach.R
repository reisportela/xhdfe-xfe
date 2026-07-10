# Gelbach (2016) conditional decomposition, HDFE-aware (M9B front-end).

#' Gelbach coefficient-movement decomposition (HDFE-aware)
#'
#' Decomposes the change in the base-specification coefficients when
#' covariate groups and/or absorbed fixed-effect blocks are added, following
#' Gelbach (2016). This is specification accounting, not causal mediation;
#' causal interpretation requires a separately justified research design.
#' Inference reproduces Gelbach's \code{b1x2} exactly
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
#' @param tol Positive fixed-effect absorption tolerance. The default
#'   \code{1e-8} preserves the historical effective tolerance.
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
                          gamma0 = FALSE, cov0 = FALSE, tol = 1e-8,
                          num_threads = 0L,
                          weights = NULL, fweights = FALSE) {
  y <- as.numeric(y)
  x1 <- as.matrix(x1)
  storage.mode(x1) <- "double"
  if (nrow(x1) != length(y)) {
    stop("y and x1 must have the same number of rows", call. = FALSE)
  }
  if (any(!is.finite(y)) || any(!is.finite(x1))) {
    stop("y and x1 must contain only finite values", call. = FALSE)
  }
  if (length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("tol must be finite and strictly positive", call. = FALSE)
  }
  x2_groups <- as.list(x2_groups %||% list())
  fes <- as.list(fes %||% list())
  if (!length(x2_groups) && !length(fes)) {
    stop("provide at least one x2 group or fixed-effect dimension", call. = FALSE)
  }
  nm <- c(names(x2_groups), names(fes))
  if (length(nm) != length(x2_groups) + length(fes) ||
      anyNA(nm) || any(!nzchar(trimws(nm)))) {
    stop("every x2/FE block must have a non-empty name", call. = FALSE)
  }
  if (anyDuplicated(nm)) {
    stop("x2 and FE block names must be unique", call. = FALSE)
  }
  if (!vce %in% c("unadjusted", "robust", "cluster")) {
    stop("vce must be \"unadjusted\", \"robust\" or \"cluster\"", call. = FALSE)
  }
  sizes <- integer(0)
  X2 <- NULL
  for (g in x2_groups) {
    g <- as.matrix(g)
    storage.mode(g) <- "double"
    if (nrow(g) != length(y)) stop("x2 group has wrong length", call. = FALSE)
    if (ncol(g) == 0L) stop("x2 groups must contain at least one column", call. = FALSE)
    if (any(!is.finite(g))) stop("x2 groups must contain only finite values", call. = FALSE)
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
  if (!is.null(cl) && length(unique(cl)) < 2L) {
    stop("vce = \"cluster\" requires at least two clusters", call. = FALSE)
  }
  w <- if (is.null(weights)) NULL else as.numeric(weights)
  if (!is.null(w)) {
    if (length(w) != length(y) || any(!is.finite(w)) || any(w <= 0)) {
      stop("weights must be finite, strictly positive and have length n", call. = FALSE)
    }
    if (isTRUE(fweights) && any(w != round(w))) {
      stop("frequency weights must be integers", call. = FALSE)
    }
  } else if (isTRUE(fweights)) {
    stop("fweights = TRUE requires weights", call. = FALSE)
  }
  out <- .xhdfe_cpp_gelbach(y, x1, X2, as.integer(sizes), fe_list, cl,
                            as.character(vce), isTRUE(gamma0), isTRUE(cov0),
                            as.numeric(tol), as.integer(num_threads), w,
                            isTRUE(fweights))
  k1 <- ncol(x1) + 1L
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
  out$group_kinds <- setNames(c(rep("x2", length(x2_groups)),
                                rep("fe", length(fes))), nm)
  out$se_type <- setNames(ifelse(out$group_kinds == "fe", "conditional_gamma0",
                                 if (isTRUE(gamma0)) "gamma0" else "full"), nm)
  if (length(fes)) {
    fe_cols <- (length(x2_groups) + 1L):length(nm)
    fe_cov <- matrix(0, k1, k1)
    for (g in fe_cols) for (h in fe_cols) {
      gi <- ((g - 1L) * k1 + 1L):(g * k1)
      hi <- ((h - 1L) * k1 + 1L):(h * k1)
      fe_cov <- fe_cov + out$cov[gi, hi, drop = FALSE]
    }
    dimnames(fe_cov) <- list(rownames(delta), rownames(delta))
    out$fe_total <- list(
      members = nm[fe_cols],
      coef = rowSums(delta[, fe_cols, drop = FALSE]),
      cov = fe_cov,
      se = sqrt(diag(fe_cov)),
      se_type = "conditional_gamma0"
    )
  } else {
    out$fe_total <- NULL
  }
  out$vce <- vce
  out$tol <- as.numeric(tol)
  out$estimand <- "coefficient_movement"
  out$causal_interpretation <- FALSE
  class(out) <- "xhdfe_gelbach"
  out
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' @export
print.xhdfe_gelbach <- function(x, digits = 6, ...) {
  cat("Gelbach coefficient-movement decomposition (xhdfe backend)\n")
  cat("Specification accounting; not causal mediation.\n")
  cat(sprintf("n = %s, vce = %s%s\n", format(x$n_obs, big.mark = ","),
              x$vce, if (isTRUE(x$gamma0)) " (gamma0)" else ""))
  cat("\nContributions (delta):\n")
  print(round(x$delta, digits))
  cat("\nStandard errors:\n")
  print(round(x$se, digits))
  if (!is.null(x$fe_total)) {
    cat("Absorbed-FE aggregate (conditional/gamma0 SE):\n")
    print(round(cbind(coef = x$fe_total$coef, se = x$fe_total$se), digits))
  }
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
