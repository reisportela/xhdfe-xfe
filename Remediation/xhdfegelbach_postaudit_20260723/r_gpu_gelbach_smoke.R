args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[[1L]] else "used"
stopifnot(mode %in% c("used", "fallback"))

suppressPackageStartupMessages(library(xhdfe))
set.seed(20260723)
n <- if (identical(mode, "used")) 100000L else 5000L
firm <- sample.int(if (identical(mode, "used")) 1000L else 100L, n,
                   replace = TRUE)
year <- sample.int(20L, n, replace = TRUE)
target <- rnorm(n)
common <- rnorm(n)
z <- 0.35 * target - 0.2 * common + rnorm(n)
firm_effect <- rnorm(max(firm), sd = 0.5)
year_effect <- rnorm(max(year), sd = 0.2)
y <- (0.8 * target - 0.3 * common + 0.6 * z +
      firm_effect[firm] + year_effect[year] + rnorm(n))
x1 <- cbind(target = target, common = common)

if (identical(mode, "used")) {
  Sys.setenv(XHDFE_GPU_BACKEND = "cpu")
  cpu <- xhdfe_gelbach(
    y, x1, x2_groups = list(observed = z),
    fes = list(firm = firm, year = year),
    vce = "cluster", cluster = firm, num_threads = 8L, gpu = FALSE
  )
  gpu <- xhdfe_gelbach(
    y, x1, x2_groups = list(observed = z),
    fes = list(firm = firm, year = year),
    vce = "cluster", cluster = firm, num_threads = 8L, gpu = TRUE
  )
  max_b <- max(abs(cpu$b_full - gpu$b_full))
  max_delta <- max(abs(cpu$delta - gpu$delta))
  max_cov <- max(abs(cpu$cov - gpu$cov))
  max_total_cov <- max(abs(cpu$total_cov - gpu$total_cov))
  cat(sprintf(
    paste0("mode=used gpu_requested=%s gpu_used=%s backend=%s status=%s ",
           "attempted=%s converged=%s iterations=%d threads=%d ",
           "max_b=%.17g max_delta=%.17g max_cov=%.17g ",
           "max_total_cov=%.17g env_after=%s\n"),
    gpu$gpu_requested, gpu$gpu_used, gpu$gpu_backend, gpu$gpu_status,
    gpu$gpu_attempted, gpu$gpu_absorption_converged,
    gpu$gpu_absorption_iterations, gpu$threads_used,
    max_b, max_delta, max_cov, max_total_cov,
    Sys.getenv("XHDFE_GPU_BACKEND")
  ))
  stopifnot(
    isTRUE(gpu$converged),
    isTRUE(gpu$gpu_requested),
    isTRUE(gpu$gpu_used),
    identical(gpu$gpu_backend, "cuda"),
    identical(gpu$gpu_status, "used"),
    isTRUE(gpu$gpu_attempted),
    isTRUE(gpu$gpu_absorption_converged),
    gpu$gpu_absorption_iterations > 0L,
    identical(Sys.getenv("XHDFE_GPU_BACKEND"), "cpu"),
    max_b < 1e-8,
    max_delta < 1e-8,
    max_cov < 1e-8,
    max_total_cov < 1e-8
  )
} else {
  fit <- xhdfe_gelbach(
    y, x1, x2_groups = list(observed = z),
    fes = list(firm = firm, year = year),
    vce = "robust", num_threads = 4L, gpu = TRUE
  )
  cat(sprintf(
    paste0("mode=fallback gpu_requested=%s gpu_used=%s backend=%s ",
           "status=%s status_code=%d attempted=%s converged=%s ",
           "iterations=%d notes=%s\n"),
    fit$gpu_requested, fit$gpu_used, fit$gpu_backend, fit$gpu_status,
    fit$gpu_status_code, fit$gpu_attempted, fit$converged,
    fit$gpu_absorption_iterations, fit$notes
  ))
  stopifnot(
    isTRUE(fit$converged),
    isTRUE(fit$gpu_requested),
    !isTRUE(fit$gpu_used),
    identical(fit$gpu_backend, "cpu"),
    identical(fit$gpu_status, "unavailable"),
    identical(fit$gpu_status_code, 2L),
    !isTRUE(fit$gpu_attempted),
    !isTRUE(fit$gpu_absorption_converged),
    identical(fit$gpu_absorption_iterations, 0L),
    grepl("GPU requested but unavailable", fit$notes, fixed = TRUE)
  )
}
