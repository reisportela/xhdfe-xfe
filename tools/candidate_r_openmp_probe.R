# Gate actual OpenMP work in an xhdfe package installed in a private R library.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("usage: candidate_r_openmp_probe.R PRIVATE_LIB FORBID_PREFIX VERSION OUT")
}
private_lib <- normalizePath(args[[1L]], mustWork = TRUE)
forbid_prefix <- normalizePath(args[[2L]], mustWork = TRUE)
expected_version <- args[[3L]]
output <- normalizePath(args[[4L]], mustWork = FALSE)
if (file.exists(output)) stop("output already exists: ", output)

Sys.setenv(
  XHDFE_GPU_BACKEND = "cpu", XHDFE_CERTIFY = "0",
  XHDFE_ABSORPTION_CACHE_MODE = "off", XHDFE_MOBILITY_MODE = "off",
  OMP_DYNAMIC = "FALSE", OMP_NUM_THREADS = "2",
  OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1"
)
.libPaths(c(private_lib, .libPaths()))

inside <- function(path, root) {
  identical(path, root) || startsWith(path, paste0(root, .Platform$file.sep))
}

receipt <- tryCatch({
  suppressPackageStartupMessages(library(xhdfe))
  package_path <- normalizePath(find.package("xhdfe"), mustWork = TRUE)
  if (!inside(package_path, private_lib)) stop("xhdfe was not loaded from the private R library")
  if (inside(package_path, forbid_prefix)) stop("xhdfe was loaded from the checkout")
  dll_path <- getLoadedDLLs()[["xhdfe"]][["path"]]
  if (is.null(dll_path)) stop("xhdfe native DLL is not loaded")
  dll_path <- normalizePath(dll_path, mustWork = TRUE)
  if (!inside(dll_path, private_lib)) stop("xhdfe DLL was not loaded from the private R library")
  if (inside(dll_path, forbid_prefix)) stop("xhdfe DLL was loaded from the checkout")
  if (as.character(packageVersion("xhdfe")) != expected_version) {
    stop("installed R package version differs")
  }

  # Exact fixture and limits transcribed from tools/plugin_openmp_probe.cpp.
  n <- 64L * 64L * 32L
  row <- 0:(n - 1L)
  replicate <- row %% 32L
  fe2 <- as.integer((row %/% 32L) %% 64L)
  fe1 <- as.integer(row %/% (32L * 64L))
  x1 <- ifelse(bitwAnd(replicate, 1L) != 0L, 1, -1)
  x2 <- ifelse(bitwAnd(replicate, 2L) != 0L, 1, -1)
  noise <- ifelse(bitwAnd(replicate, 4L) != 0L, 0.125, -0.125)
  X <- cbind(x1, x2)
  y <- 1.25 * x1 - 0.5 * x2 + 3 + 0.25 * fe1 - 0.125 * fe2 + noise
  expected_beta <- c(1.25, -0.5)
  expected_rss <- n * 0.015625
  beta_atol <- 1e-10
  v_scaled_atol <- 1e-8
  rss_rtol <- 1e-10

  rows <- list()
  reference <- NULL
  for (requested in c(1L, 2L)) {
    fit <- xhdfe_fit(
      y, X, list(fe1, fe2), vcov = "unadjusted",
      threads = requested, maxiter = 1000L, tol = 1e-8,
      drop_singletons = FALSE, fit_intercept = FALSE,
      absorption_method = "gauss-seidel",
      tolerance_mode = "reghdfe-comparable", backend = "cpu"
    )
    diagnostics <- as.integer(c(
      fit$threads_requested, fit$threads_effective,
      fit$threads_used, fit$parallel_workers_active
    ))
    if (!identical(diagnostics, rep(requested, 4L))) {
      stop("thread diagnostics differ: ", paste(diagnostics, collapse = ","))
    }
    if (!isTRUE(fit$openmp_enabled) || fit$thread_capacity < 2L) {
      stop("OpenMP is disabled or capacity is below two")
    }
    if (!isTRUE(fit$converged) || !isTRUE(fit$precision_certified)) {
      stop("fit is not converged/certified")
    }
    if (isTRUE(fit$gpu_used) || fit$gpu_status_code != 0L) stop("CPU probe used a GPU")
    if (fit$nobs != n) stop("fit changed the analytic sample")
    if (!is.finite(fit$df_r) || fit$df_r < 1 || fit$df_r > n) stop("invalid residual degrees of freedom")
    if (fit$absorption_method_code != 1L) stop("explicit Gauss-Seidel method changed")

    beta <- as.numeric(fit$coefficients)
    covariance <- unname(fit$vcov)
    variance <- 0.015625 / fit$df_r
    expected_v <- diag(c(variance, variance))
    beta_error <- max(abs(beta - expected_beta))
    v_error <- max(abs(covariance - expected_v)) / variance
    rss_error <- abs(fit$rss - expected_rss) / expected_rss
    if (beta_error > beta_atol) stop("analytic beta mismatch: ", beta_error)
    if (v_error > v_scaled_atol) stop("analytic covariance mismatch: ", v_error)
    if (rss_error > rss_rtol) stop("analytic RSS mismatch: ", rss_error)

    if (is.null(reference)) {
      parity_b <- parity_v <- parity_rss <- 0
      reference <- list(beta = beta, covariance = covariance, rss = fit$rss)
    } else {
      parity_b <- max(abs(beta - reference$beta))
      parity_v <- max(abs(covariance - reference$covariance)) / variance
      parity_rss <- abs(fit$rss - reference$rss) / expected_rss
      if (parity_b > beta_atol) stop("thread beta disagreement: ", parity_b)
      if (parity_v > v_scaled_atol) stop("thread covariance disagreement: ", parity_v)
      if (parity_rss > rss_rtol) stop("thread RSS disagreement: ", parity_rss)
    }
    rows[[length(rows) + 1L]] <- list(
      requested = requested,
      threads_effective = diagnostics[[2L]],
      threads_used = diagnostics[[3L]],
      parallel_workers_active = diagnostics[[4L]],
      openmp_enabled = isTRUE(fit$openmp_enabled),
      beta_error = beta_error,
      covariance_scaled_error = v_error,
      rss_relative_error = rss_error,
      thread_beta_difference = parity_b,
      thread_covariance_scaled_difference = parity_v,
      thread_rss_relative_difference = parity_rss
    )
  }
  list(
    schema = "xhdfe-candidate-r-openmp-v1", status = "PASS",
    package_path = package_path, dll_path = dll_path,
    package_version = expected_version,
    fixture_rows = n,
    limits = list(beta_atol = beta_atol,
                  covariance_diagonal_scaled = v_scaled_atol,
                  rss_rtol = rss_rtol),
    rows = rows
  )
}, error = function(error) {
  list(schema = "xhdfe-candidate-r-openmp-v1", status = "FAIL",
       reason = conditionMessage(error))
})

dput(receipt, file = output)
cat("R_OPENMP_PROBE_", receipt$status, " ", output, "\n", sep = "")
if (!identical(receipt$status, "PASS")) quit(status = 1L)
