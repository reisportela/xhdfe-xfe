# R bridge and independent scalar two-firm oracle, including estimated controls.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1], mustWork = TRUE)
.libPaths(c(normalizePath(args[3], mustWork = TRUE), .libPaths()))
library(Rcpp)
api <- new.env(parent = globalenv())
dll <- dyn.load(normalizePath(args[2], mustWork = TRUE))
for (symbol in getDLLRegisteredRoutines(dll)[[".Call"]]) assign(symbol$name, symbol, envir = api)
for (file in c("RcppExports.R", "akm.R")) sys.source(file.path(root, "r/xhdfe/R", file), envir = api)
worker <- rep(0:63, each = 4L)
firm <- rep(c(0L, 0L, 1L, 1L), 64L)
f <- ifelse(firm == 0, -.5, .5)
s <- 1 + worker %% 3
v <- s * rep(c(1, -1, 1, -1), 64L)
x <- 5 * f + v
y <- .3 * sin(.37 * (worker + 1)) + .7 * f + .35 * x + .1 * sin(.73 * seq_along(worker))
vv <- sum(v^2)
raw <- sum(f * y) / 64
gamma <- sum(v * y) / vv
fy <- as.numeric(rowsum(f * y, worker))
vy <- as.numeric(rowsum(v * y, worker))
local_vv <- 4 * (1 + (0:63) %% 3)^2
lf <- 1 / 64
lv <- local_vv / vv
geometric <- function(a) 1 + a + a^2 + a^3 + a^4
score <- -.5 / 64 * (geometric(lf) * (fy - raw) + lf^5 * fy) +
  2.5 * local_vv / vv * (geometric(lv) * (vy / local_vv - gamma) + lv^5 * vy / local_vv)
variance <- sum(score^2)
centre <- abs(raw - 5 * gamma) / 2
radius <- sqrt(-2 * log(.05) * variance)
expected <- c(2 * 256 / 255 * centre * sqrt(variance),
              256 / 255 * max(0, centre - radius)^2,
              256 / 255 * (centre + radius)^2)
ols <- lm(y ~ x + factor(worker) + factor(firm))
for (method in c("exact", "jla")) {
  point <- suppressWarnings(api$xhdfe_akm_kss(y, worker, firm, X = matrix(x),
    sigma_centering = "person_year", leverages = method, num_threads = 2L))
  fit <- suppressWarnings(api$xhdfe_akm_kss(y, worker, firm, X = matrix(x),
    sigma_centering = "person_year", leverages = method, num_threads = 2L, compute_se = TRUE))
  p <- fit$projection_inference
  actual <- c(fit$component_se$se_var_psi, p$var_psi$ci_lb, p$var_psi$ci_ub)
  stopifnot(fit$converged, identical(point$kss, fit$kss), max(abs(actual - expected)) < 1e-9,
    abs(fit$beta[1] - coef(ols)["x"]) < 1e-10, p$method == "worker_projection",
    p$cluster == "worker", p$backend == "cpu", p$dimension == 2L,
    p$clusters == 64L, p$threads_used == 2L, p$parallel_workers_active == 2L,
    is.null(fit$weak_id), is.na(fit$component_se$se_var_alpha))
  shifted <- suppressWarnings(api$xhdfe_akm_kss(y + 1e6, worker, firm, X = matrix(x),
    sigma_centering = "person_year", leverages = method, num_threads = 2L,
    compute_se = TRUE, eigen_diagnostics = TRUE))
  other <- c(shifted$component_se$se_var_psi, shifted$projection_inference$var_psi$ci_lb,
             shifted$projection_inference$var_psi$ci_ub)
  stopifnot(max(abs(actual - other)) < 1e-8, is.null(shifted$weak_id))
}
tools::checkRd(file.path(root, "r/xhdfe/man/xhdfe_akm_kss.Rd"))
cat("AKM_PROJECTION_R_CONTRACT_PASS\n")
