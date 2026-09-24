args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
library(jsonlite)
data <- read.csv(args[3])
reference <- read.csv(args[4], check.names = FALSE)
backend <- args[5]
data$audit_cluster <- (data$group*13 + floor(data$group/5)) %% 23
records <- list()
for (row in which(reference$engine == "regress")) {
  ref <- reference[row, ]
  kind <- ref$weight_case
  w <- if (kind == "iw_fractional") 0.73 + (data$g1+2*data$g2) %% 7/13 else data$weight
  weight_type <- if (startsWith(kind, "iw_")) "importance" else
    switch(kind, aw = "analytic", fw = "frequency", pw = "probability")
  for (interface in c("matrix", "formula")) {
    options <- list(weights = w, weights_type = weight_type, vcov = ref$vce,
                    tol = 1e-12, tolerance_mode = "strict-residual", threads = 2,
                    backend = backend)
    if (ref$vce == "cluster") options$cluster <- data$audit_cluster
    if (interface == "matrix") {
      options$y <- data$y
      options$X <- as.matrix(data[c("x1", "x2", "x3")])
      options$fes <- data[c("g1", "g2")]
      model <- do.call(xhdfe_fit, options)
    } else {
      options$fml <- y ~ x1+x2+x3 | g1+g2
      options$data <- data
      model <- do.call(xhdfe, options)
    }
    values <- c(N = model$nobs, df_r = model$df_r, rss = model$rss)
    for (prefix in c("b", "se", "p", "lo", "hi")) {
      v <- switch(prefix, b = model$coefficients, se = model$se, p = model$pvalues,
                  lo = model$conf_int[, 1], hi = model$conf_int[, 2])
      values[paste0(prefix, 1:3)] <- unname(v[1:3])
    }
    difference <- abs(values - unlist(ref[names(values)]))
    stopifnot(difference[["N"]] == 0, difference[["df_r"]] == 0,
              max(difference) < 1e-8, model$converged, model$precision_certified)
    if (backend == "cuda") stopifnot(model$gpu_used, model$gpu_status_code == 1)
    records[[length(records)+1L]] <- list(weight_case = kind, vce = ref$vce,
      interface = interface, max_error = max(difference), differences = as.list(difference))
  }
}
write_json(list(backend = backend, PASS = TRUE, rows = records), args[6],
           auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("R_WEIGHTS_CONTRACT_PASS", backend, length(records), "\n")
