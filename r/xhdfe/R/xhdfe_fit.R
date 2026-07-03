# Low-level matrix interface (mirrors the Python HdfeRegressor().fit() call).

xhdfe_fit <- function(y, X, fes = NULL,
                      weights = NULL, weights_type = c("analytic", "frequency"),
                      cluster = NULL, vcov = NULL,
                      instruments = NULL, endogenous = NULL,
                      slopes = NULL,
                      group = NULL, individual = NULL, aggregation = "mean",
                      save_fe = FALSE, groupvar = FALSE,
                      tol = 1e-8, maxiter = 100000,
                      tolerance_mode = "reghdfe-comparable",
                      convergence = "auto", check_interval = 1,
                      absorption_method = "auto", symmetric_sweep = FALSE,
                      jacobi_relaxation = 0,
                      fit_intercept = TRUE,
                      fe_tolerance = 1e-6, fe_recovery_method = "hybrid",
                      drop_singletons = TRUE, keep_singletons = NULL,
                      dof = NULL, ssc = NULL, stats_style = "reghdfe",
                      level = 95,
                      threads = 0, default_threads = 0, max_threads = 0,
                      min_parallel_rows = 20000, target_rows_per_thread = 500000,
                      backend = c("default", "cpu", "cuda", "metal")) {
  cl <- match.call()
  backend <- match.arg(backend)
  weights_type <- match.arg(weights_type)

  y <- as.numeric(y)
  n <- length(y)
  if (is.data.frame(X)) X <- as.matrix(X)
  if (!is.matrix(X)) X <- matrix(as.numeric(X), ncol = 1L)
  storage.mode(X) <- "double"
  if (nrow(X) != n) stop("X must have the same number of rows as y", call. = FALSE)
  coef_names <- colnames(X)
  if (is.null(coef_names)) coef_names <- paste0("x", seq_len(ncol(X)))

  if (anyNA(y) || anyNA(X)) {
    stop("y/X contain missing values; the matrix interface does not drop NAs ",
         "-- use xhdfe() or filter beforehand", call. = FALSE)
  }

  if (is.null(fes)) fes <- list()
  if (is.matrix(fes)) fes <- lapply(seq_len(ncol(fes)), function(j) fes[, j])
  if (!is.list(fes)) fes <- list(fes)
  fe_labels <- names(fes)
  if (is.null(fe_labels) || any(!nzchar(fe_labels))) {
    fe_labels <- paste0("fe", seq_along(fes))
    if (!is.null(names(fes))) {
      keep <- nzchar(names(fes))
      fe_labels[keep] <- names(fes)[keep]
    }
  }
  fes_use <- lapply(seq_along(fes), function(i) to_ids(fes[[i]], fe_labels[i]))

  cluster_list <- if (is.null(cluster)) list() else {
    if (is.matrix(cluster)) {
      lapply(seq_len(ncol(cluster)), function(j) cluster[, j])
    } else if (is.data.frame(cluster)) {
      as.list(cluster)
    } else if (is.list(cluster)) cluster else list(cluster)
  }
  cluster_names <- names(cluster_list)
  cluster_use <- lapply(cluster_list, to_ids, label = "cluster")
  se_type <- resolve_vcov(vcov, length(cluster_use) > 0L)

  if (!is.null(weights)) {
    weights <- as.numeric(weights)
    if (length(weights) != n) stop("weights length must match nobs", call. = FALSE)
    if (anyNA(weights)) {
      stop("weights contain missing values; the matrix interface does not ",
           "drop NAs -- use xhdfe() or filter beforehand", call. = FALSE)
    }
  }

  Z <- NULL
  endogenous_idx <- integer(0)
  if (!is.null(instruments)) {
    Z <- as.matrix(instruments)
    storage.mode(Z) <- "double"
    if (anyNA(Z)) {
      stop("instruments contain missing values; the matrix interface does ",
           "not drop NAs -- use xhdfe() or filter beforehand", call. = FALSE)
    }
    if (is.null(endogenous)) {
      stop("`endogenous` (column positions or names in X) is required with ",
           "`instruments`", call. = FALSE)
    }
    if (is.character(endogenous)) {
      endogenous_idx <- match(endogenous, coef_names)
      if (anyNA(endogenous_idx)) {
        stop("endogenous names not found among the columns of X", call. = FALSE)
      }
    } else {
      endogenous_idx <- as.integer(endogenous)
    }
  } else if (!is.null(endogenous)) {
    stop("`endogenous` supplied without `instruments`", call. = FALSE)
  }

  slopes_use <- list()
  if (!is.null(slopes)) {
    if (!is.list(slopes)) stop("`slopes` must be a list", call. = FALSE)
    if (!is.null(slopes$fe) || !is.null(slopes$fe_index)) slopes <- list(slopes)
    slopes_use <- lapply(slopes, function(s) {
      fe_index <- if (!is.null(s$fe)) s$fe else s$fe_index
      if (is.null(fe_index)) {
        stop("each slope entry needs `fe` (1-based index into fes)", call. = FALSE)
      }
      values <- if (!is.null(s$values)) s$values else s$x
      if (is.null(values)) {
        stop("each slope entry needs `values` (the continuous slope variable)",
             call. = FALSE)
      }
      list(fe_index = as.integer(fe_index) - 1L,
           values = as.numeric(values),
           include_intercept = isTRUE(s$include_intercept) || isTRUE(s$intercept))
    })
  }

  if (!is.null(individual) && is.null(group)) {
    stop("`individual` requires `group` (group-level-outcome mode); see the ",
         "Group-level outcomes section in ?xhdfe", call. = FALSE)
  }
  group_use <- if (is.null(group)) NULL else to_ids(group, "group")
  individual_use <- if (is.null(individual)) NULL else to_ids(individual, "individual")
  if (!is.null(individual_use)) {
    already <- any(vapply(fes_use, function(f) identical(f, individual_use), TRUE))
    if (!already) {
      fes_use[[length(fes_use) + 1L]] <- individual_use
      fe_labels <- c(fe_labels, "individual")
    }
  }

  opts <- build_opts(se_type, tol, maxiter, check_interval, convergence,
                     fit_intercept, threads, default_threads, max_threads,
                     min_parallel_rows, target_rows_per_thread,
                     drop_singletons, keep_singletons, save_fe,
                     symmetric_sweep, absorption_method, jacobi_relaxation,
                     level, dof, ssc, tolerance_mode, fe_tolerance,
                     fe_recovery_method, stats_style, weights_type, groupvar)

  res <- run_with_backend(backend, function() {
    .xhdfe_cpp_fit(y, X, fes_use, weights, cluster_use, Z,
                   as.integer(endogenous_idx - 1L), slopes_use,
                   group_use, individual_use, aggregation, opts)
  })

  out <- finalize_xhdfe(res, coef_names, n, seq_len(n), cl, level,
                        backend, se_type, cluster_names, fe_labels,
                        tolerance_mode,
                        weights_sum = if (is.null(weights) ||
                                          length(res$sample_index0) == 0L) NULL
                                      else sum(weights[res$sample_index0 + 1L]),
                        X_used = X, y_used = y)
  out$weights_type <- if (is.null(weights)) NULL else weights_type
  out
}
