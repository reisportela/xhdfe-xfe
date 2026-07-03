# Internal estimation engine shared by xhdfe() and xhdfe_fit().

# Assemble the options list handed to the C++ binding. `se_type` is resolved
# beforehand; everything else arrives via the user-facing arguments.
build_opts <- function(se_type, tol, maxiter, check_interval, convergence,
                       fit_intercept, threads, default_threads, max_threads,
                       min_parallel_rows, target_rows_per_thread,
                       drop_singletons, keep_singletons, save_fe,
                       symmetric_sweep, absorption_method, jacobi_relaxation,
                       level, dof, ssc, tolerance_mode, fe_tolerance,
                       fe_recovery_method, stats_style, weights_type,
                       groupvar) {
  ssc <- parse_ssc(ssc)
  opts <- list(
    se_type = se_type,
    tol = tol,
    max_iter = as.integer(maxiter),
    check_interval = as.integer(check_interval),
    convergence = convergence,
    fit_intercept = isTRUE(fit_intercept),
    num_threads = as.integer(threads),
    default_threads = as.integer(default_threads),
    max_threads = as.integer(max_threads),
    min_parallel_rows = as.integer(min_parallel_rows),
    target_rows_per_thread = as.integer(target_rows_per_thread),
    drop_singletons = isTRUE(drop_singletons),
    retain_fes = isTRUE(save_fe),
    symmetric_sweep = isTRUE(symmetric_sweep),
    absorption_method = absorption_method,
    jacobi_relaxation = jacobi_relaxation,
    level = level,
    tolerance_mode = tolerance_mode,
    fe_tolerance = fe_tolerance,
    fe_recovery_method = fe_recovery_method,
    stats_style = stats_style,
    weights_are_frequencies = identical(weights_type, "frequency"),
    groupvar = isTRUE(groupvar)
  )
  if (!is.null(keep_singletons)) opts$keepsingletons <- isTRUE(keep_singletons)
  if (!is.null(dof)) opts$dofadjustments <- as.character(dof)
  opts[names(ssc)] <- ssc
  opts
}

# Parse fixest/Stata-style small-sample-correction spec into ssc_* options.
# Accepts a named list (ssc_k_adj = TRUE, ...) or a Stata-style string such
# as "K.adj=1 K.fixef=nonnested G.df=min".
parse_ssc <- function(ssc) {
  if (is.null(ssc)) return(list())
  if (is.list(ssc)) {
    allowed <- c("ssc_k_adj", "ssc_k_fixef", "ssc_k_exact", "ssc_g_adj",
                 "ssc_g_df", "ssc_t_df")
    bad <- setdiff(names(ssc), allowed)
    if (length(bad)) {
      stop("Unknown ssc entries: ", paste(bad, collapse = ", "), call. = FALSE)
    }
    return(ssc)
  }
  if (!is.character(ssc) || length(ssc) != 1L) {
    stop("`ssc` must be a named list or a single string", call. = FALSE)
  }
  out <- list()
  tokens <- strsplit(gsub(",", " ", ssc), "\\s+")[[1L]]
  tokens <- tokens[nzchar(tokens)]
  as_flag <- function(v) {
    if (v %in% c("", "1", "true", "yes")) TRUE
    else if (v %in% c("0", "false", "no")) FALSE
    else stop("Invalid ssc flag value: ", v, call. = FALSE)
  }
  for (tok in tokens) {
    kv <- strsplit(tok, "=", fixed = TRUE)[[1L]]
    key <- tolower(kv[1L])
    val <- if (length(kv) > 1L) tolower(kv[2L]) else ""
    if (key == "k.adj") out$ssc_k_adj <- as_flag(val)
    else if (key == "k.fixef") {
      out$ssc_k_fixef <- if (val %in% c("true", "yes")) "full"
                         else if (val %in% c("false", "no")) "none" else val
    }
    else if (key == "k.exact") out$ssc_k_exact <- as_flag(val)
    else if (key == "g.adj") out$ssc_g_adj <- as_flag(val)
    else if (key == "g.df") out$ssc_g_df <- val
    else if (key == "t.df") {
      out$ssc_t_df <- suppressWarnings(as.numeric(val))
      if (is.na(out$ssc_t_df)) {
        stop("t.df must be numeric in the R interface (min/conventional are ",
             "resolved by the core via G.df)", call. = FALSE)
      }
    }
    else stop("Unknown ssc token: ", tok, call. = FALSE)
  }
  out
}

# Run the fit with the requested backend env override and fail-closed GPU
# semantics (mirrors Stata's error 498 contract for gpubackend(cuda|metal)).
run_with_backend <- function(backend, fun) {
  backend <- match.arg(backend, c("default", "cpu", "cuda", "metal"))
  if (backend != "default") {
    old <- Sys.getenv("XHDFE_GPU_BACKEND", unset = NA)
    Sys.setenv(XHDFE_GPU_BACKEND = backend)
    on.exit({
      if (is.na(old)) Sys.unsetenv("XHDFE_GPU_BACKEND")
      else Sys.setenv(XHDFE_GPU_BACKEND = old)
    }, add = TRUE)
  }
  res <- fun()
  if (backend %in% c("cuda", "metal") && !isTRUE(res$gpu_used)) {
    stop("backend = \"", backend, "\" was requested but the GPU was not used (status: ",
         gpu_status_label(res$gpu_status_code),
         "). xhdfe refuses to return silent CPU-fallback results; rebuild the ",
         "package with XHDFE_ENABLE_CUDA=ON and ensure a CUDA device is ",
         "available, or drop the backend request.", call. = FALSE)
  }
  res
}

# Post-process the raw C++ result list into the user-facing "xhdfe" object.
finalize_xhdfe <- function(res, coef_names, n_input, rows_used, call, level,
                           backend, se_type, cluster_names, fe_labels,
                           tolerance_mode, weights_sum = NULL,
                           X_used = NULL, y_used = NULL) {
  k <- length(coef_names)
  ncoef <- length(res$coefficients)
  has_cons <- FALSE
  if (ncoef == k + 1L) {
    coef_names <- c(coef_names, "(Intercept)")
    has_cons <- TRUE
  } else if (ncoef != k) {
    stop(sprintf("internal error: %d coefficients for %d regressors", ncoef, k),
         call. = FALSE)
  }

  coefficients <- stats::setNames(res$coefficients, coef_names)
  se <- stats::setNames(res$se, coef_names)
  tvalues <- stats::setNames(res$tvalues, coef_names)
  pvalues <- stats::setNames(res$pvalues, coef_names)
  conf_int <- res$conf_int
  dimnames(conf_int) <- list(coef_names,
                             c(sprintf("%g %%", (100 - res_level(level)) / 2),
                               sprintf("%g %%", 100 - (100 - res_level(level)) / 2)))
  covariance <- res$covariance
  dimnames(covariance) <- list(coef_names, coef_names)

  omitted <- res$omitted_reason
  if (length(omitted) == k && has_cons) omitted <- c(omitted, 0L)
  if (length(omitted) == length(coef_names)) {
    omitted <- stats::setNames(as.integer(omitted), coef_names)
  } else {
    omitted <- stats::setNames(integer(length(coef_names)), coef_names)
  }

  # Rows of the original data that survived NA filtering AND singleton
  # dropping (0-based core indices refer to the post-NA input arrays). In
  # group()/individual() mode the core estimates on collapsed group-level
  # rows and does not report a sample index; residuals then stay at the
  # group level.
  grouped_fit <- length(res$sample_index0) == 0L && res$nobs > 0L
  if (grouped_fit) {
    sample_rows <- integer(0)
    residuals_full <- res$residuals
  } else {
    sample_rows <- rows_used[res$sample_index0 + 1L]
    residuals_full <- rep(NA_real_, n_input)
    residuals_full[sample_rows] <- res$residuals
  }

  nobs <- res$nobs
  df_r <- res$df_resid
  df_m <- res$df_m
  # Effective N (weighted): the Stata command uses the plugin's effective
  # count for r2_a/ll; identical to nobs when unweighted or aweighted.
  n_eff <- if (!is.null(res$nobs_effective) && res$nobs_effective > 0) {
    res$nobs_effective
  } else {
    nobs
  }

  # Derived statistics, mirroring stata/xhdfe.ado.
  mss <- res$tss - res$rss
  used_df_r <- max(res$df_resid_unadj - res$df_a_nested, 0)
  rmse <- if (used_df_r > 0) sqrt(res$rss / used_df_r) else sqrt(res$rss)
  r2_a <- r2_a_within <- NA_real_
  if (res$tss > 0 && used_df_r > 0) {
    r2_a <- 1 - (res$rss / used_df_r) / (res$tss / max(n_eff - has_cons, 1))
  }
  if (res$tss_within > 0 && used_df_r + df_m > 0 && used_df_r > 0) {
    r2_a_within <- 1 - (res$rss / used_df_r) /
      (res$tss_within / (used_df_r + df_m))
  }
  if (isTRUE(res$saturated)) {
    r2_a <- 1
    r2_a_within <- 1
  }
  ll <- ll0 <- NA_real_
  if (res$rss > 0 && n_eff > 0) {
    ll <- -0.5 * n_eff * (log(2 * pi) + log(res$rss / n_eff) + 1)
  }
  if (res$tss_within > 0 && n_eff > 0) {
    ll0 <- -0.5 * n_eff * (log(2 * pi) + log(res$tss_within / n_eff) + 1)
  }

  # Model F statistic (Wald on kept, non-intercept coefficients).
  f_stat <- f_p <- NA_real_
  keep <- which(omitted == 0L & coef_names != "(Intercept)" &
                  is.finite(se) & se > 0)
  if (length(keep) > 0 && df_r > 0) {
    V1 <- covariance[keep, keep, drop = FALSE]
    b1 <- coefficients[keep]
    g <- sym_ginv(V1)
    if (g$rank > 0) {
      f_stat <- drop(t(b1) %*% g$inv %*% b1) / g$rank
      f_p <- stats::pf(f_stat, g$rank, df_r, lower.tail = FALSE)
    }
  }

  fe_effects <- res$fe_effects
  if (length(fe_effects) && length(fe_labels) == length(fe_effects)) {
    names(fe_effects) <- fe_labels
  }

  out <- list(
    coefficients = coefficients,
    se = se,
    tvalues = tvalues,
    pvalues = pvalues,
    conf_int = conf_int,
    vcov = covariance,
    residuals = residuals_full,
    omitted = omitted,
    nobs = nobs,
    nobs_full = res$nobs_full,
    num_singletons = res$num_singletons,
    nobs_effective = res$nobs_effective,
    nobs_full_effective = res$nobs_full_effective,
    num_singletons_effective = res$num_singletons_effective,
    sample = sample_rows,
    df_r = df_r,
    df_r_unadj = res$df_resid_unadj,
    df_m = df_m,
    rank = df_m,
    df_a = res$df_a,
    df_a_levels = res$df_a_levels,
    df_a_exact = res$df_a_exact,
    df_a_nested = res$df_a_nested,
    df_a_initial = res$df_a_levels,
    df_a_redundant = max(res$df_a_levels - res$df_a, 0),
    r2 = res$r2,
    r2_within = res$r2_within,
    r2_a = r2_a,
    r2_a_within = r2_a_within,
    rss = res$rss,
    tss = res$tss,
    tss_within = res$tss_within,
    mss = mss,
    sigma2 = res$sigma2,
    rmse = rmse,
    F_stat = f_stat,
    F_p = f_p,
    ll = ll,
    ll_0 = ll0,
    saturated = res$saturated,
    fe_num_levels = res$fe_num_levels,
    fe_base_levels = res$fe_base_levels,
    fe_redundant = res$fe_redundant,
    fe_num_coefs = res$fe_num_coefs,
    fe_inexact = res$fe_inexact,
    fe_nested = res$fe_nested,
    fe_labels = fe_labels,
    iterations = res$num_iterations,
    converged = res$converged,
    absorption_method_used = res$absorption_method_used,
    absorption_method_code = res$absorption_method_code,
    threads_used = res$threads_used,
    gpu_used = res$gpu_used,
    gpu_status_code = res$gpu_status_code,
    gpu_status = gpu_status_label(res$gpu_status_code),
    gpu_attempted = res$gpu_attempted,
    gpu_absorption_converged = res$gpu_absorption_converged,
    gpu_absorption_iterations = res$gpu_absorption_iterations,
    backend_requested = backend,
    fe_effects = fe_effects,
    fe_recovery_iterations = res$fe_recovery_iterations,
    fe_recovery_max_delta = res$fe_recovery_max_delta,
    fe_recovery_converged = res$fe_recovery_converged,
    groupvar = if (length(res$groupvar)) res$groupvar else NULL,
    num_clusters = res$num_clusters,
    cluster_counts = res$cluster_counts,
    cluster_combo_counts = res$cluster_combo_counts,
    cluster_scale = res$cluster_scale,
    cluster_names = cluster_names,
    vcv_psd_fixed = res$vcv_psd_fixed,
    se_type = se_type,
    level = res_level(level),
    tolerance_mode = tolerance_mode,
    has_intercept = has_cons,
    sumweights = if (is.null(weights_sum)) nobs else weights_sum,
    n_input = n_input,
    call = call,
    version = as.character(utils::packageVersion("xhdfe"))
  )

  out$grouped_fit <- grouped_fit

  # Prediction caches (n-vectors; the full design is not retained).
  if (!is.null(X_used) && !is.null(y_used) && !grouped_fit) {
    kept <- res$sample_index0 + 1L
    Xs <- X_used[kept, , drop = FALSE]
    b_slopes <- coefficients[seq_len(ncol(Xs))]
    b_slopes[!is.finite(b_slopes)] <- 0
    xb <- if (ncol(Xs) > 0) drop(Xs %*% b_slopes) else numeric(length(kept))
    if (has_cons) xb <- xb + coefficients[["(Intercept)"]]
    xb_full <- y_full <- stdp_full <- rep(NA_real_, n_input)
    xb_full[sample_rows] <- xb
    y_full[sample_rows] <- y_used[kept]
    Vok <- covariance
    Vok[!is.finite(Vok)] <- 0
    design <- if (has_cons) cbind(Xs, 1) else Xs
    vv <- rowSums((design %*% Vok) * design)
    stdp_full[sample_rows] <- sqrt(pmax(vv, 0))
    out$xb_cache <- xb_full
    out$y_cache <- y_full
    out$stdp_cache <- stdp_full
  }
  class(out) <- "xhdfe"

  if (!isTRUE(res$converged)) {
    warning("absorption did not converge within maxiter iterations; ",
            "estimates are reported but should not be trusted", call. = FALSE)
  }
  if (isTRUE(res$vcv_psd_fixed)) {
    warning("the multiway-cluster VCV required the Cameron-Gelbach-Miller ",
            "positive-semi-definite adjustment (as in reghdfe)", call. = FALSE)
  }
  if (isTRUE(res$saturated)) {
    message("note: the absorbed model fits the data perfectly (saturated ",
            "design); standard errors are not identified")
  }
  out
}

# Normalize a level given either as percent (95) or fraction (0.95).
res_level <- function(level) {
  if (level > 0 && level <= 1) level * 100 else level
}
