# Main user-facing estimator: formula interface.

xhdfe <- function(fml, data = NULL,
                  vcov = NULL, cluster = NULL,
                  weights = NULL, weights_type = c("analytic", "frequency"),
                  subset = NULL,
                  group = NULL, individual = NULL, aggregation = "mean",
                  save_fe = FALSE, groupvar = FALSE,
                  tol = 1e-8, maxiter = 100000,
                  tolerance_mode = "reghdfe-comparable",
                  convergence = "auto", check_interval = 1,
                  absorption_method = "auto", symmetric_sweep = FALSE,
                  jacobi_relaxation = 0,
                  fe_tolerance = 1e-6, fe_recovery_method = "hybrid",
                  drop_singletons = TRUE, keep_singletons = NULL,
                  dof = NULL, ssc = NULL, stats_style = "reghdfe",
                  level = 95,
                  threads = 0, default_threads = 0, max_threads = 0,
                  min_parallel_rows = 20000, target_rows_per_thread = 500000,
                  backend = c("default", "cpu", "cuda", "metal"),
                  na.action = c("omit", "fail")) {
  cl <- match.call()
  subset_expr <- substitute(subset)
  backend <- match.arg(backend)
  weights_type <- match.arg(weights_type)
  na.action <- match.arg(na.action)
  env <- environment(fml)
  if (is.null(env)) env <- parent.frame()

  spec <- split_xhdfe_formula(fml)

  # ---- dependent variable -------------------------------------------------
  y <- eval(spec$lhs, data, env)
  if (!is.numeric(y)) stop("the dependent variable must be numeric", call. = FALSE)
  n_input <- length(y)
  y <- as.numeric(y)

  # ---- regressor design ---------------------------------------------------
  rhs_fml <- stats::as.formula(call("~", spec$regressors), env = env)
  tf <- stats::terms(rhs_fml, data = data)
  fit_intercept <- attr(tf, "intercept") == 1L
  mf <- stats::model.frame(tf, data = data, na.action = stats::na.pass)
  X <- stats::model.matrix(tf, mf)
  xlevels <- stats::.getXlevels(tf, mf)
  contrasts_fit <- attr(X, "contrasts")
  int_col <- match("(Intercept)", colnames(X))
  if (!is.na(int_col)) X <- X[, -int_col, drop = FALSE]
  coef_names <- colnames(X)

  # ---- IV parts -----------------------------------------------------------
  endogenous_idx <- integer(0)
  Z <- NULL
  endo_tf <- NULL
  endo_xlevels <- NULL
  if (!is.null(spec$endogenous)) {
    endo_fml <- stats::as.formula(call("~", spec$endogenous), env = env)
    endo_tf <- stats::terms(endo_fml, data = data)
    endo_mf <- stats::model.frame(endo_tf, data = data,
                                  na.action = stats::na.pass)
    Xe <- stats::model.matrix(endo_tf, endo_mf)
    endo_xlevels <- stats::.getXlevels(endo_tf, endo_mf)
    ic <- match("(Intercept)", colnames(Xe))
    if (!is.na(ic)) Xe <- Xe[, -ic, drop = FALSE]
    endogenous_idx <- ncol(X) + seq_len(ncol(Xe))  # 1-based, converted below
    X <- cbind(X, Xe)
    coef_names <- c(coef_names, colnames(Xe))

    inst_fml <- stats::as.formula(call("~", spec$instruments), env = env)
    inst_tf <- stats::terms(inst_fml, data = data)
    Z <- stats::model.matrix(inst_tf,
                             stats::model.frame(inst_tf, data = data,
                                                na.action = stats::na.pass))
    ic <- match("(Intercept)", colnames(Z))
    if (!is.na(ic)) Z <- Z[, -ic, drop = FALSE]
  }
  if (nrow(X) != n_input) {
    stop("regressors and dependent variable have different lengths", call. = FALSE)
  }

  # ---- fixed effects and heterogeneous slopes -----------------------------
  fe_raw <- list()        # raw id vectors, one per absorbed dimension
  slope_meta <- list()    # list(fe_index (1-based), values, include_intercept)
  fe_labels <- character(0)
  if (!is.null(spec$fe)) {
    for (term in split_plus_terms(spec$fe)) {
      parsed <- parse_fe_term(term)
      ids <- eval_fe_expr(parsed$fe_expr, data, env)
      if (length(ids) != n_input) {
        stop("fixed effect `", deparse1(parsed$fe_expr),
             "` has the wrong length", call. = FALSE)
      }
      if (length(parsed$slopes) == 0L) {
        fe_raw[[length(fe_raw) + 1L]] <- ids
        fe_labels <- c(fe_labels, parsed$label)
      } else {
        # One absorbed dimension per slope variable, all on the same carrier
        # ids; the intercept (if any) rides on the first term. This matches
        # Stata's absorb(fe##c.x1 fe#c.x2 ...) expansion.
        for (j in seq_along(parsed$slopes)) {
          values <- eval(parsed$slopes[[j]], data, env)
          if (!is.numeric(values) || length(values) != n_input) {
            stop("slope variable `", deparse1(parsed$slopes[[j]]),
                 "` must be numeric with the same length as the data",
                 call. = FALSE)
          }
          fe_raw[[length(fe_raw) + 1L]] <- ids
          include_int <- parsed$include_intercept && j == 1L
          slope_meta[[length(slope_meta) + 1L]] <-
            list(fe_index = length(fe_raw), values = as.numeric(values),
                 include_intercept = include_int)
          fe_labels <- c(fe_labels,
                         paste0(deparse1(parsed$fe_expr),
                                if (include_int) "[" else "[[",
                                deparse1(parsed$slopes[[j]]),
                                if (include_int) "]" else "]]"))
        }
      }
    }
  }

  # ---- clusters -----------------------------------------------------------
  if (inherits(vcov, "formula")) {
    if (!is.null(cluster)) {
      stop("supply the clusters either through `vcov` or `cluster`, not both",
           call. = FALSE)
    }
    cluster <- vcov
    vcov <- "cluster"
  }
  cluster_list <- normalize_cluster_spec(cluster, data, env)
  se_type <- resolve_vcov(vcov, length(cluster_list) > 0L)
  cluster_names <- names(cluster_list)
  for (v in cluster_list) {
    if (length(v) != n_input) stop("cluster variable has the wrong length", call. = FALSE)
  }

  # ---- weights ------------------------------------------------------------
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

  # ---- group / individual -------------------------------------------------
  eval_id_arg <- function(arg, label) {
    if (is.null(arg)) return(NULL)
    if (inherits(arg, "formula")) {
      aenv <- environment(arg)
      if (is.null(aenv)) aenv <- env
      arg <- eval(arg[[length(arg)]], data, aenv)
    }
    else if (is.character(arg) && length(arg) == 1L && !is.null(data) &&
             arg %in% names(data)) arg <- data[[arg]]
    if (length(arg) != n_input) {
      stop("`", label, "` has the wrong length", call. = FALSE)
    }
    arg
  }
  group_raw <- eval_id_arg(group, "group")
  individual_raw <- eval_id_arg(individual, "individual")
  if (!is.null(individual_raw) && is.null(group_raw)) {
    stop("`individual` requires `group`", call. = FALSE)
  }

  # ---- subset / NA handling ----------------------------------------------
  keep <- rep(TRUE, n_input)
  if (!is.null(subset_expr)) {
    sub <- eval(subset_expr, data, parent.frame())
    if (is.logical(sub)) {
      keep <- keep & !is.na(sub) & sub
    } else {
      keep <- rep(FALSE, n_input)
      keep[sub] <- TRUE
    }
  }
  ok <- !is.na(y) & stats::complete.cases(X)
  if (!is.null(Z)) ok <- ok & stats::complete.cases(Z)
  for (v in fe_raw) ok <- ok & !is.na(v)
  for (s in slope_meta) ok <- ok & !is.na(s$values)
  for (v in cluster_list) ok <- ok & !is.na(v)
  if (!is.null(weights)) ok <- ok & !is.na(weights)
  if (!is.null(group_raw)) ok <- ok & !is.na(group_raw)
  if (!is.null(individual_raw)) ok <- ok & !is.na(individual_raw)
  if (na.action == "fail" && any(keep & !ok)) {
    stop("missing values in the estimation variables (na.action = \"fail\")",
         call. = FALSE)
  }
  keep <- keep & ok
  rows_used <- which(keep)
  if (length(rows_used) == 0L) stop("no observations", call. = FALSE)

  subset_vec <- function(v) if (is.null(v)) NULL else v[rows_used]

  y_use <- y[rows_used]
  X_use <- X[rows_used, , drop = FALSE]
  Z_use <- if (is.null(Z)) NULL else Z[rows_used, , drop = FALSE]
  fes_use <- lapply(fe_raw, function(v) to_ids(subset_vec(v), "fixed effect"))
  clusters_use <- lapply(cluster_list, function(v) to_ids(subset_vec(v), "cluster"))
  weights_use <- subset_vec(weights)
  slopes_use <- lapply(slope_meta, function(s) {
    list(fe_index = s$fe_index - 1L,  # 0-based for the C++ layer
         values = s$values[rows_used],
         include_intercept = s$include_intercept)
  })
  group_use <- if (is.null(group_raw)) NULL else to_ids(subset_vec(group_raw), "group")
  individual_use <- if (is.null(individual_raw)) NULL else
    to_ids(subset_vec(individual_raw), "individual")

  # The core requires the individual ids to also appear among the absorbed
  # dimensions; Stata appends them (last) automatically -- mirror that.
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
    .xhdfe_cpp_fit(y_use, X_use, fes_use, weights_use, clusters_use,
                   Z_use, as.integer(endogenous_idx - 1L), slopes_use,
                   group_use, individual_use, aggregation, opts)
  })

  out <- finalize_xhdfe(res, coef_names, n_input, rows_used, cl, level,
                        backend, se_type, cluster_names, fe_labels,
                        tolerance_mode,
                        weights_sum = if (is.null(weights_use) ||
                                          length(res$sample_index0) == 0L) NULL
                                      else sum(weights_use[res$sample_index0 + 1L]),
                        X_used = X_use, y_used = y_use)
  out$fml <- fml
  out$terms <- tf
  out$terms_endo <- endo_tf
  out$xlevels <- c(xlevels, endo_xlevels)
  out$contrasts <- contrasts_fit
  out$data_name <- deparse1(substitute(data))
  out$weights_type <- if (is.null(weights)) NULL else weights_type
  out$group_name <- if (!is.null(group)) deparse1(substitute(group)) else NULL
  out$individual_name <- if (!is.null(individual)) deparse1(substitute(individual)) else NULL
  out$aggregation <- if (!is.null(group)) aggregation else NULL
  out
}
