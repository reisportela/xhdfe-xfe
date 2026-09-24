args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 7L)
xhdfe_library <- normalizePath(args[1], mustWork = TRUE)
dependency_libraries <- strsplit(args[2], .Platform$path.sep, fixed = TRUE)[[1L]]
dependency_libraries <- dependency_libraries[nzchar(dependency_libraries)]
dependency_libraries <- vapply(dependency_libraries, normalizePath, character(1), mustWork = TRUE)
backend <- args[3]
output <- args[4]
expected_package <- normalizePath(args[5], mustWork = TRUE)
expected_dll <- normalizePath(args[6], mustWork = TRUE)
scratch <- normalizePath(args[7], mustWork = TRUE)
stopifnot(backend %in% c("cpu", "cuda"), !file.exists(output), dir.exists(dirname(output)))

for (key in c("TMPDIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "R_USER_CACHE_DIR")) {
  value <- normalizePath(Sys.getenv(key), mustWork = TRUE)
  stopifnot(startsWith(value, scratch))
}
stopifnot(tolower(Sys.getenv("XHDFE_ABSORPTION_CACHE_MODE")) == "off",
          tolower(Sys.getenv("XHDFE_MOBILITY_MODE")) == "off")
audit <- Sys.getenv("XHDFE_CERTIFY", unset = "0")
stopifnot(tolower(audit) %in% c("0", "1", "false", "true", "no", "yes"))

.libPaths(c(xhdfe_library, dependency_libraries, .libPaths()))
stopifnot(identical(normalizePath(find.package("xhdfe", lib.loc = xhdfe_library), mustWork = TRUE),
                    expected_package))
library(xhdfe, lib.loc = xhdfe_library)
library(jsonlite)
loaded_dll <- normalizePath(getLoadedDLLs()[["xhdfe"]][["path"]], mustWork = TRUE)
stopifnot(identical(loaded_dll, expected_dll))

index <- 0:4095
cell <- index %/% 256
data <- data.frame(fe_a = cell %/% 4, fe_b = cell %% 4,
                   sign_a = 2 * (index %% 2) - 1,
                   sign_b = 2 * ((index %/% 2) %% 2) - 1)
data$leverage <- data$fe_a == 0 & data$fe_b == 1
data$x <- data$sign_a * (1 + data$sign_b) / 2 * data$leverage
data$weight <- 1 + (3 * data$fe_a + data$fe_b) %% 7

run_one <- function(exponent, offset, interface, save_fe = FALSE) {
  small <- 2^-exponent
  noise <- ifelse(data$leverage, small, 1) * data$sign_b
  data$y <- .75 * data$x + offset * (data$fe_b - data$fe_a) + noise
  options <- list(weights = data$weight, weights_type = "analytic", vcov = "robust",
                  tol = 1e-8, tolerance_mode = "reghdfe-comparable", maxiter = 1000,
                  threads = 2, backend = backend, drop_singletons = FALSE,
                  save_fe = save_fe)
  if (interface == "matrix") {
    options$y <- data$y
    options$X <- as.matrix(data["x"])
    options$fes <- data[c("fe_a", "fe_b")]
    options$fit_intercept <- save_fe
    estimator <- xhdfe_fit
  } else {
    options$fml <- if (save_fe) y ~ x | fe_a + fe_b else y ~ 0 + x | fe_a + fe_b
    options$data <- data
    estimator <- xhdfe
  }
  record <- list(exponent = exponent, offset = offset, interface = interface,
                 role = if (save_fe) "savefe" else "default")
  result <- tryCatch({
    model <- do.call(estimator, options)
    expected_variance <- 32 * small^2 / 4088
    errors <- c(beta = abs(model$coefficients[[1L]] - .75),
                variance = abs(model$vcov[1L, 1L] / expected_variance - 1),
                residual = max(abs(model$residuals - noise)))
    if (save_fe) {
      fitted <- model$coefficients[[1L]] * data$x +
        tail(model$coefficients, 1L) + Reduce(`+`, model$fe_effects)
      errors <- c(errors, reconstruction = max(abs(data$y - fitted - noise)))
    }
    gpu_ok <- isTRUE(model$gpu_used) == (backend == "cuda") &&
      (backend != "cuda" || model$gpu_status_code == 1L)
    passed <- errors[["beta"]] <= 1e-9 && errors[["variance"]] <= 1e-8 &&
      errors[["residual"]] <= 1e-9 && model$df_r == 4088 && model$converged &&
      model$precision_certified && gpu_ok &&
      (!save_fe || (model$fe_recovery_converged && length(model$fe_effects) == 2L &&
                    errors[["reconstruction"]] <= 1e-6))
    list(verdict = if (passed) "PASS" else "FAIL", errors = as.list(errors),
         df_r = model$df_r, gpu_used = isTRUE(model$gpu_used),
         gpu_status = model$gpu_status_code, threads_used = model$threads_used)
  }, error = function(error) list(verdict = "FAIL", error = conditionMessage(error)))
  c(record, result)
}

rows <- list()
for (exponent in c(0, 10, 20, 30, 40)) for (offset in c(1, 16, 64))
  for (interface in c("matrix", "formula"))
    rows[[length(rows) + 1L]] <- run_one(exponent, offset, interface)
for (interface in c("matrix", "formula"))
  rows[[length(rows) + 1L]] <- run_one(40, 16, interface, save_fe = TRUE)

passed <- sum(vapply(rows, function(row) row$verdict == "PASS", logical(1)))
write_json(list(backend = backend, audit = audit, cases = 15L,
                passed = passed, total = length(rows), rows = rows,
                package = expected_package, dll = loaded_dll),
           output, auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("LOW_NOISE_R", backend, passed, "OF", length(rows), "\n")
quit(status = as.integer(passed != length(rows)))
