args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 6L)
xhdfe_library <- normalizePath(args[1], mustWork = TRUE)
dependency_libraries <- strsplit(args[2], .Platform$path.sep, fixed = TRUE)[[1L]]
dependency_libraries <- dependency_libraries[nzchar(dependency_libraries)]
dependency_libraries <- vapply(dependency_libraries, normalizePath, character(1), mustWork = TRUE)
.libPaths(c(xhdfe_library, dependency_libraries, .libPaths()))
expected_package <- normalizePath(args[5], mustWork = TRUE)
expected_dll <- normalizePath(args[6], mustWork = TRUE)
stopifnot(identical(normalizePath(find.package("xhdfe", lib.loc = xhdfe_library), mustWork = TRUE),
                    expected_package))
library(xhdfe, lib.loc = xhdfe_library)
library(jsonlite)
manifest <- fromJSON(args[3], simplifyVector = FALSE)
backend <- args[4]
loaded_dll <- normalizePath(getLoadedDLLs()[["xhdfe"]][["path"]], mustWork = TRUE)
stopifnot(identical(loaded_dll, expected_dll))
for (job in manifest$jobs) {
  if (job$backend != backend) next
  output <- file.path(manifest$output, job$id)
  stopifnot(dir.create(output, recursive = FALSE, showWarnings = FALSE))
  meta <- fromJSON(file.path(job$fixture, "case.json"), simplifyVector = FALSE)
  data <- read.csv(file.path(manifest$csv, paste0(job$case, ".csv")), check.names = FALSE)
  fes <- unlist(meta$fes)
  slopes <- meta$slopes
  fit_args <- list(tol = if (job$precision == "strict") 1e-12 else 1e-8,
                   maxiter = 100000, threads = 2, backend = backend,
                   absorption_method = job$method, save_fe = isTRUE(job$savefe),
                   vcov = meta$vce)
  fit_args$tolerance_mode <- switch(job$precision,
    strict = if (length(slopes)) "reghdfe-comparable" else "strict-residual",
    fast = "xhdfe-fast", `invalid-strict-slope` = "strict-residual",
    "reghdfe-comparable")
  if (job$precision == "strict" && length(slopes)) fit_args$convergence <- "both"
  if (meta$weight != "") {
    fit_args$weights <- data$weight
    fit_args$weights_type <- if (meta$weight == "fw") "frequency" else "analytic"
  }
  if (length(meta$clusters)) fit_args$cluster <- data[unlist(meta$clusters)]
  if (meta$kind != "standard") fit_args$group <- data$group
  if (meta$kind == "group_individual") {
    fit_args$individual <- data$individual
    fit_args$aggregation <- meta$aggregation
  }
  if (job$interface == "r-matrix") {
    fit_args$y <- data$y
    fit_args$X <- as.matrix(data[c("x1", "x2", "x3")])
    fit_args$fes <- data[fes]
    fit_args$fit_intercept <- meta$intercept
    if (length(slopes)) {
      fit_args$slopes <- lapply(slopes, function(s)
        list(fe = s[[1]] + 1L, values = data[[s[[2]]]], include_intercept = s[[3]]))
    }
    estimate <- xhdfe_fit
  } else {
    fe_terms <- fes
    for (s in slopes) {
      bracket <- if (s[[3]]) c("[", "]") else c("[[", "]]")
      fe_terms[s[[1]] + 1L] <- paste0(fes[s[[1]] + 1L], bracket[1], s[[2]], bracket[2])
    }
    formula <- paste0("y ~ ", if (!meta$intercept) "0 + " else "", "x1 + x2 + x3")
    if (length(fe_terms)) formula <- paste(formula, "|", paste(fe_terms, collapse = " + "))
    fit_args$fml <- as.formula(formula)
    fit_args$data <- data
    estimate <- xhdfe
  }
  start <- proc.time()[["elapsed"]]
  result <- tryCatch({
    model <- do.call(estimate, fit_args)
    selected <- model$sample
    raw <- list(rc = 0L, beta = unname(model$coefficients[1:3]),
                cons = if (length(model$coefficients) > 3L) unname(model$coefficients[4]) else 0,
                V = unname(model$vcov[1:3, 1:3]), N = model$nobs,
                rss = model$rss, df_r = model$df_r, df_a = model$df_a,
                singletons = model$num_singletons, iterations = model$iterations,
                converged = as.integer(model$converged), certified = as.integer(model$precision_certified),
                method_used = model$absorption_method_code, gpu_used = as.integer(model$gpu_used),
                gpu_status = model$gpu_status_code, threads = model$threads_used,
                groups = unname(data$group[selected]), residuals = unname(model$residuals[selected]),
                module = loaded_dll, package = expected_package)
    if (isTRUE(job$savefe)) {
      raw$recovered_fe <- unname(Reduce(`+`, model$fe_effects))
      raw$recovery_converged <- model$fe_recovery_converged
    }
    raw
  }, error = function(e) list(rc = 1L, error = conditionMessage(e), converged = 0L,
                              module = loaded_dll, package = expected_package))
  result$seconds <- proc.time()[["elapsed"]] - start
  write_json(result, file.path(output, "raw.json"), auto_unbox = TRUE, digits = NA, na = "null", pretty = TRUE)
  cat("R_ATTEMPT", job$id, "rc=", result$rc, "\n")
}
